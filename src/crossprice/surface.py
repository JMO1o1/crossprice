"""Offline SPX quote screening, BSM inversion and reproducible smile evidence.

The input is an immutable one-time snapshot, not a live data service. Midpoint
shape violations are screening findings, not proof of executable arbitrage.
Pandas and matplotlib belong to the existing plots extra / dev environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from crossprice.analytic import OptionKind, bsm_greeks, bsm_price
from crossprice.implied_vol import implied_vol


@dataclass(frozen=True)
class SurfaceConfig:
    """Rules fixed before evaluating the snapshot; rates/yields are decimals."""

    min_days: float = 7.0
    max_relative_spread: float = 0.5
    shape_tolerance: float = 1e-8
    rate_bump: float = 0.0025
    div_bump: float = 0.005
    plot_min_moneyness: float = 0.75
    plot_max_moneyness: float = 1.25
    atm_distance: float = 0.01
    wing_distance: float = 0.03

    def __post_init__(self) -> None:
        if any(
            not np.isfinite(value) or value <= 0
            for value in (
                self.min_days,
                self.max_relative_spread,
                self.shape_tolerance,
                self.rate_bump,
                self.div_bump,
                self.plot_min_moneyness,
                self.plot_max_moneyness,
                self.atm_distance,
                self.wing_distance,
            )
        ):
            raise ValueError("surface settings must be finite and positive")
        if self.plot_min_moneyness >= 0.9 or self.plot_max_moneyness <= 1:
            raise ValueError("plot range must contain 90% and ATM moneyness")


DEFAULT_CONFIG = SurfaceConfig()


def treasury_rate(tau: float, curve: pd.DataFrame) -> float:
    """Linearly interpolate par percent, then 2*log1p(y/200), a zero-rate proxy.

    No extrapolation; a par curve is not a bootstrapped discount curve. This
    semiannual bond-equivalent conversion is an explicitly approximate model
    input for both bill and coupon maturities, tested through sensitivity bumps.
    """
    ordered = curve.sort_values("tenor_years")
    tenors = ordered["tenor_years"].to_numpy(dtype=float)
    yields = ordered["par_yield_pct"].to_numpy(dtype=float)
    if (
        len(tenors) < 2
        or not np.all(np.isfinite(tenors))
        or not np.all(np.isfinite(yields))
        or np.any(np.diff(tenors) <= 0)
        or np.any(yields <= -200)
        or not np.isfinite(tau)
        or not tenors[0] <= tau <= tenors[-1]
    ):
        raise ValueError("invalid Treasury curve or maturity outside its tenor range")
    return float(2 * np.log1p(np.interp(tau, tenors, yields) / 200))


def clean_quotes(chain: pd.DataFrame, config: SurfaceConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    """Retain every input row with one first-failure reason and a midpoint.

    Zero volume OR zero open interest is excluded, a conservative activity
    filter, not a no-arbitrage theorem. Duplicate contracts are all excluded,
    so input ordering cannot choose a preferred quote. Source IV is not a
    cleaning criterion: missing published IV must not hide our own inversion.
    """
    numeric = ("spot", "strike", "bid", "ask", "bid_size", "ask_size", "volume", "open_interest")
    required = (
        *numeric,
        "option_symbol",
        "underlying",
        "settlement",
        "expiry",
        "kind",
        "valuation_timestamp",
        "expiry_timestamp",
        "cboe_iv",
    )
    missing = set(required) - set(chain)
    if missing:
        raise ValueError(f"missing snapshot columns: {sorted(missing)}")
    table = chain.copy(deep=True).reset_index(drop=True)
    table["quote_id"] = np.arange(len(table))
    for column in (*numeric, "cboe_iv"):
        table[column] = pd.to_numeric(table[column], errors="coerce")
    expiry = pd.to_datetime(table["expiry_timestamp"], format="ISO8601", utc=True, errors="coerce")
    valuation = pd.to_datetime(
        table["valuation_timestamp"], format="ISO8601", utc=True, errors="coerce"
    )
    table["tau"] = (expiry - valuation).dt.total_seconds() / (365 * 86400)
    table["mid"] = (table["bid"] + table["ask"]) / 2
    table["relative_spread"] = (table["ask"] - table["bid"]) / table["mid"]
    table["moneyness"] = table["strike"] / table["spot"]
    table["status"], table["reason"] = "candidate", ""
    rules = (
        ("nonfinite_quote", ~np.isfinite(table[list(numeric)]).all(axis=1)),
        (
            "invalid_contract",
            (table.spot <= 0) | (table.strike <= 0) | ~table.kind.isin(["call", "put"]),
        ),
        ("unsupported_contract", (table.underlying != "SPX") | (table.settlement != "AM")),
        ("invalid_timestamp", ~np.isfinite(table.tau)),
        ("under_seven_days", table.tau * 365 < config.min_days),
        ("crossed_quote", table.ask < table.bid),
        ("nonpositive_bid", table.bid <= 0),
        ("zero_size", (table.bid_size <= 0) | (table.ask_size <= 0)),
        ("zero_volume", table.volume <= 0),
        ("zero_open_interest", table.open_interest <= 0),
        ("wide_spread", table.relative_spread > config.max_relative_spread),
        ("duplicate_contract", table.duplicated(["expiry", "kind", "strike"], keep=False)),
    )
    for reason, rejected in rules:
        mask = (table.status == "candidate") & rejected
        table.loc[mask, ["status", "reason"]] = ["dropped", reason]
    return table


def infer_carry(table: pd.DataFrame, curve: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use the nearest-to-spot eligible matching call/put strike per expiry.

    F=K+exp(r*T)*(C-P), q=r-log(F/S)/T. Bid/ask pair extremes give a forward
    interval; it is a quote range, not a statistical CI. Fit before static
    pruning and record the exact pair even if subsequently screened out.
    """
    output = table.copy(deep=True)
    for column in ("rate", "div", "forward"):
        output[column] = np.nan
    carry_rows = []
    for expiry, group in output.groupby("expiry", sort=True):
        eligible = group[group.status == "candidate"]
        if eligible.empty:
            carry_rows.append(dict(expiry=expiry, status="no_eligible_quotes"))
            continue
        if eligible.spot.nunique() != 1 or eligible.tau.nunique() != 1:
            raise ValueError("one expiry must share a single spot and valuation time")
        spot, tau = float(eligible.spot.iloc[0]), float(eligible.tau.iloc[0])
        rate = treasury_rate(tau, curve)
        calls, puts = eligible[eligible.kind == "call"], eligible[eligible.kind == "put"]
        pairs = calls.merge(puts, on="strike", suffixes=("_call", "_put"))
        if pairs.empty:
            output.loc[eligible.index, ["status", "reason"]] = [
                "inversion_failed",
                "no_atm_parity_pair",
            ]
            carry_rows.append(dict(expiry=expiry, status="no_atm_parity_pair", rate=rate, tau=tau))
            continue
        pairs["distance"] = abs(pairs.strike / spot - 1)
        pair = pairs.sort_values(["distance", "strike"]).iloc[0]
        forward = float(pair.strike + np.exp(rate * tau) * (pair.mid_call - pair.mid_put))
        forward_low = float(pair.strike + np.exp(rate * tau) * (pair.bid_call - pair.ask_put))
        forward_high = float(pair.strike + np.exp(rate * tau) * (pair.ask_call - pair.bid_put))
        if forward_low <= 0 or not np.isfinite(forward_high):
            output.loc[eligible.index, ["status", "reason"]] = [
                "inversion_failed",
                "invalid_parity_forward",
            ]
            carry_rows.append(
                dict(expiry=expiry, status="invalid_parity_forward", rate=rate, tau=tau)
            )
            continue
        div = float(rate - np.log(forward / spot) / tau)
        output.loc[group.index, ["rate", "div", "forward"]] = [rate, div, forward]
        carry_rows.append(
            dict(
                expiry=expiry,
                status="ok",
                spot=spot,
                tau=tau,
                rate=rate,
                div=div,
                forward=forward,
                forward_low=forward_low,
                forward_high=forward_high,
                div_low=rate - np.log(forward_high / spot) / tau,
                div_high=rate - np.log(forward_low / spot) / tau,
                pair_strike=float(pair.strike),
                pair_moneyness=float(pair.strike / spot),
                call_symbol=pair.option_symbol_call,
                put_symbol=pair.option_symbol_put,
                call_mid=float(pair.mid_call),
                put_mid=float(pair.mid_put),
            )
        )
    return output, pd.DataFrame(carry_rows)


def screen_static_arbitrage(
    table: pd.DataFrame, config: SurfaceConfig = DEFAULT_CONFIG
) -> pd.DataFrame:
    """Greedy midpoint screen per expiry/kind, rechecking after every removal.

    Calls decrease, puts increase; vertical slopes are bounded by exp(-r*T),
    and successive secant slopes on the actual unequal strike grid increase.
    For the first violation remove the involved quote with widest relative
    spread; equal spreads remove the lexicographically last option symbol.
    This deterministic heuristic is not a maximum
    clean subset or a bid/ask-feasibility optimization. No midpoint is repaired.
    The tolerance is 1e-8 in price differences / dimensionless secant slopes.
    """
    output = table.copy(deep=True)
    for _, group in output[output.status == "candidate"].groupby(["expiry", "kind"], sort=True):
        active = group.sort_values(["strike", "option_symbol"])
        while len(active) >= 2:
            differences = np.diff(active.mid.to_numpy())
            slopes = differences / np.diff(active.strike.to_numpy())
            direction = -1 if active.kind.iloc[0] == "call" else 1
            bad = np.flatnonzero(direction * differences < -config.shape_tolerance)
            reason, width = "strike_monotonicity", 2
            if not len(bad):
                discount = float(np.exp(-active.rate.iloc[0] * active.tau.iloc[0]))
                bad = np.flatnonzero(direction * slopes > discount + config.shape_tolerance)
                reason = "vertical_spread_bound"
            if not len(bad):
                bad = np.flatnonzero(np.diff(slopes) < -config.shape_tolerance)
                reason, width = "strike_convexity", 3
            if not len(bad):
                break
            involved = active.iloc[bad[0] : bad[0] + width]
            removed = involved.sort_values(["relative_spread", "option_symbol"]).index[-1]
            output.loc[removed, ["status", "reason"]] = ["dropped", reason]
            active = active.drop(index=removed)
    return output


@dataclass(frozen=True)
class SurfaceRun:
    """All quotes and decisions, fitted carry, plotted data and sensitivity rows."""

    quotes: pd.DataFrame
    carry: pd.DataFrame
    smile: pd.DataFrame
    term_structure: pd.DataFrame
    sensitivity: pd.DataFrame
    config: SurfaceConfig


def _solve(row: pd.Series, price: float, rate: float, div: float) -> dict:
    try:
        result = implied_vol(
            float(row.spot),
            float(row.strike),
            float(row.tau),
            rate,
            price,
            div,
            cast(OptionKind, row.kind),
        )
    except (ValueError, FloatingPointError) as error:
        return dict(
            vol=None,
            iterations=0,
            method="rejected",
            price_residual=None,
            status=type(error).__name__,
            detail=str(error),
        )
    return asdict(result) | {"detail": "" if result.status == "converged" else result.status}


def analyze_surface(
    chain: pd.DataFrame,
    curve: pd.DataFrame,
    external_div: float,
    config: SurfaceConfig = DEFAULT_CONFIG,
) -> SurfaceRun:
    """Screen and invert offline, retaining all exclusions and failed inversions.

    Baseline carry is estimated only from the recorded ATM pair, never from
    CBOE IV. Sensitivities use +/-25 bp rates, +/-50 bp q, external trailing q,
    and the ATM parity pair's bid/ask carry endpoints. Rate changes are shown
    both at fixed q and fixed F (change q by the same amount as r). They are
    model-input perturbations, not estimates of a probability distribution.
    The smile uses OTM puts K<=F and calls K>F; all retained ITM inversions
    remain in quotes. Moneyness for plots and 90%-ATM comparison is K/S.
    Nearest observations are used, never extrapolated: <=1% from ATM and
    <=3% from 90%; exact selected moneyness and contract IDs are recorded.
    """
    if not np.isfinite(external_div):
        raise ValueError("external_div must be finite")
    table, carry = infer_carry(clean_quotes(chain, config), curve)
    table = screen_static_arbitrage(table, config)
    for column in (
        "iv",
        "iv_bid",
        "iv_ask",
        "price_residual",
        "iterations",
        "vega",
        "cboe_price_residual",
        "source_iv_diff_pp",
        "parity_price_residual",
    ):
        table[column] = np.nan
    for column in (
        "iv_status",
        "iv_method",
        "iv_detail",
        "bid_status",
        "bid_detail",
        "ask_status",
        "ask_detail",
        "comparison_status",
    ):
        table[column] = ""
    sensitivity_rows = []
    for index, row in table[table.status == "candidate"].iterrows():
        result = _solve(row, float(row.mid), float(row.rate), float(row["div"]))
        table.loc[
            index, ["iv", "iterations", "iv_method", "price_residual", "iv_status", "iv_detail"]
        ] = [
            result["vol"],
            result["iterations"],
            result["method"],
            result["price_residual"],
            result["status"],
            result["detail"],
        ]
        converged = result["status"] == "converged"
        table.loc[index, ["status", "reason"]] = [
            "inverted" if converged else "inversion_failed",
            "" if converged else f"iv_{result['status']}",
        ]
        for side in ("bid", "ask"):
            endpoint = _solve(row, float(row[side]), float(row.rate), float(row["div"]))
            table.loc[index, [f"iv_{side}", f"{side}_status", f"{side}_detail"]] = [
                endpoint["vol"],
                endpoint["status"],
                endpoint["detail"],
            ]
        if converged:
            table.loc[index, "vega"] = float(
                bsm_greeks(
                    row.spot,
                    row.strike,
                    row.tau,
                    row.rate,
                    result["vol"],
                    row["div"],
                    cast(OptionKind, row.kind),
                ).vega
            )
            if np.isfinite(row.cboe_iv) and row.cboe_iv > 0:
                table.loc[index, "source_iv_diff_pp"] = 100 * (result["vol"] - row.cboe_iv)
                table.loc[index, "cboe_price_residual"] = (
                    float(
                        bsm_price(
                            row.spot,
                            row.strike,
                            row.tau,
                            row.rate,
                            row.cboe_iv,
                            row["div"],
                            cast(OptionKind, row.kind),
                        )
                    )
                    - row.mid
                )
                table.loc[index, "comparison_status"] = "compared_not_calibrated"
            else:
                table.loc[index, "comparison_status"] = "source_iv_missing_or_nonpositive"
        fitted = carry[carry.expiry == row.expiry].iloc[0]
        scenarios = (
            ("rate_down_fixed_q", row.rate - config.rate_bump, row["div"]),
            ("rate_up_fixed_q", row.rate + config.rate_bump, row["div"]),
            ("rate_down_fixed_forward", row.rate - config.rate_bump, row["div"] - config.rate_bump),
            ("rate_up_fixed_forward", row.rate + config.rate_bump, row["div"] + config.rate_bump),
            ("div_down", row.rate, row["div"] - config.div_bump),
            ("div_up", row.rate, row["div"] + config.div_bump),
            ("external_div", row.rate, external_div),
            ("parity_div_low", row.rate, fitted.div_low),
            ("parity_div_high", row.rate, fitted.div_high),
        )
        for scenario, rate, div in scenarios:
            varied = _solve(row, float(row.mid), float(rate), float(div))
            delta = (
                100 * (varied["vol"] - result["vol"])
                if (converged and varied["status"] == "converged")
                else np.nan
            )
            sensitivity_rows.append(
                dict(
                    quote_id=int(row.quote_id),
                    option_symbol=row.option_symbol,
                    expiry=row.expiry,
                    kind=row.kind,
                    moneyness=float(row.moneyness),
                    scenario=scenario,
                    rate=float(rate),
                    div=float(div),
                    baseline_iv=result["vol"] if converged else None,
                    iv=varied["vol"],
                    delta_iv_pp=delta,
                    status=varied["status"],
                    reason=varied["detail"],
                    price_residual=varied["price_residual"],
                )
            )

    valid = table[table.status == "inverted"]
    pairs = valid[valid.kind == "call"].merge(
        valid[valid.kind == "put"], on=["expiry", "strike"], suffixes=("_call", "_put")
    )
    for _, pair in pairs.iterrows():
        parity_residual = float(
            pair.mid_call
            - pair.mid_put
            - (
                pair.spot_call * np.exp(-pair.div_call * pair.tau_call)
                - pair.strike * np.exp(-pair.rate_call * pair.tau_call)
            )
        )
        table.loc[
            table.quote_id.isin([pair.quote_id_call, pair.quote_id_put]), "parity_price_residual"
        ] = parity_residual
    smile = table[
        (table.status == "inverted")
        & (
            ((table.kind == "put") & (table.strike <= table.forward))
            | ((table.kind == "call") & (table.strike > table.forward))
        )
        & table.moneyness.between(config.plot_min_moneyness, config.plot_max_moneyness)
    ].copy()
    smile = smile.sort_values(["expiry", "strike", "kind"]).reset_index(drop=True)
    term_rows = []
    for _, fitted in carry.iterrows():
        summary = fitted.to_dict()
        group = smile[smile.expiry == fitted.expiry]
        for label, target, tolerance in (
            ("atm", 1.0, config.atm_distance),
            ("wing90", 0.9, config.wing_distance),
        ):
            eligible = group[abs(group.moneyness - target) <= tolerance].copy()
            if eligible.empty:
                summary.update({f"{label}_iv": np.nan, f"{label}_status": "no_nearby_quote"})
                continue
            eligible["distance"] = abs(eligible.moneyness - target)
            closest = eligible.sort_values(["distance", "strike"]).iloc[0]
            summary.update(
                {
                    f"{label}_{key}": closest[key]
                    for key in (
                        "iv",
                        "iv_bid",
                        "iv_ask",
                        "moneyness",
                        "option_symbol",
                        "source_iv_diff_pp",
                    )
                }
            )
            summary[f"{label}_status"] = "observed"
        summary["wing90_minus_atm_pp"] = 100 * (summary["wing90_iv"] - summary["atm_iv"])
        summary["plot_quotes"] = len(group)
        term_rows.append(summary)
    sensitivity = pd.DataFrame(
        sensitivity_rows,
        columns=(
            "quote_id",
            "option_symbol",
            "expiry",
            "kind",
            "moneyness",
            "scenario",
            "rate",
            "div",
            "baseline_iv",
            "iv",
            "delta_iv_pp",
            "status",
            "reason",
            "price_residual",
        ),
    )
    return SurfaceRun(table, carry, smile, pd.DataFrame(term_rows), sensitivity, config)


def _json_records(table: pd.DataFrame) -> list[dict]:
    return table.astype(object).where(pd.notna(table), None).to_dict(orient="records")


def surface_summary(run: SurfaceRun) -> dict:
    """JSON-safe counts and per-expiry observations; no hidden exclusion denominator."""
    comparisons = []
    for expiry, group in run.quotes.groupby("expiry", sort=True):
        delta = group.source_iv_diff_pp.dropna()
        comparisons.append(
            dict(
                expiry=expiry,
                compared=len(delta),
                median_signed_pp=float(delta.median()) if len(delta) else None,
                median_abs_pp=float(delta.abs().median()) if len(delta) else None,
                max_abs_pp=float(delta.abs().max()) if len(delta) else None,
            )
        )
    sensitivities = []
    for (expiry, scenario), group in run.sensitivity.groupby(["expiry", "scenario"], sort=True):
        delta = group.delta_iv_pp.dropna()
        sensitivities.append(
            dict(
                expiry=expiry,
                scenario=scenario,
                attempted=len(group),
                compared=len(delta),
                failed=int((group.status != "converged").sum()),
                median_signed_pp=float(delta.median()) if len(delta) else None,
                max_abs_pp=float(delta.abs().max()) if len(delta) else None,
            )
        )
    for summary in sensitivities:
        term = run.term_structure[run.term_structure.expiry == summary["expiry"]].iloc[0]
        group = run.sensitivity[
            (run.sensitivity.expiry == summary["expiry"])
            & (run.sensitivity.scenario == summary["scenario"])
        ]
        for label in ("atm", "wing90"):
            match = group[group.option_symbol == term.get(f"{label}_option_symbol")]
            summary[f"{label}_status"] = str(match.status.iloc[0]) if len(match) else "unavailable"
            delta = match.delta_iv_pp.iloc[0] if len(match) else np.nan
            summary[f"{label}_delta_iv_pp"] = float(delta) if np.isfinite(delta) else None
        summary["skew_change_pp"] = (
            summary["wing90_delta_iv_pp"] - summary["atm_delta_iv_pp"]
            if summary["wing90_delta_iv_pp"] is not None and summary["atm_delta_iv_pp"] is not None
            else None
        )
    attempted = run.quotes[run.quotes.iv_status != ""]
    return dict(
        rows=len(run.quotes),
        endpoint_failure_counts={
            side: int((~attempted[f"{side}_status"].isin(["converged", "lower_bound"])).sum())
            for side in ("bid", "ask")
        },
        status_counts={
            str(key): int(value) for key, value in run.quotes.status.value_counts().items()
        },
        reason_counts={
            str(key): int(value)
            for key, value in run.quotes.loc[run.quotes.reason != "", "reason"]
            .value_counts()
            .items()
        },
        plotted_quotes=len(run.smile),
        source_comparison=comparisons,
        sensitivities=sensitivities,
        term_structure=_json_records(run.term_structure),
        max_price_residual=float(run.quotes.price_residual.abs().max())
        if run.quotes.price_residual.notna().any()
        else None,
        method_counts={
            str(key): int(value)
            for key, value in run.quotes.loc[run.quotes.status == "inverted", "iv_method"]
            .value_counts()
            .items()
        },
    )


def write_artifacts(run: SurfaceRun, out: Path, provenance: dict) -> dict:
    """Write deterministic CSV/JSON/PNG artifacts, never overwriting input snapshots."""
    from matplotlib.figure import Figure

    out.mkdir(parents=True, exist_ok=True)
    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    frames = {
        "surface_quotes.csv": run.quotes,
        "surface_carry.csv": run.carry,
        "surface_sensitivity.csv": run.sensitivity,
        "figures/iv_smile.csv": run.smile,
        "figures/iv_term_structure.csv": run.term_structure,
    }
    for name, table in frames.items():
        table.to_csv(out / name, index=False, float_format="%.17g", lineterminator="\n")
    expiries = list(run.carry.expiry)
    figure = Figure(figsize=(12, 3.4 * max(1, (len(expiries) + 1) // 2)), layout="constrained")
    axes = figure.subplots(max(1, (len(expiries) + 1) // 2), 2, squeeze=False).ravel()
    for axis, expiry in zip(axes, expiries, strict=False):
        group = run.smile[run.smile.expiry == expiry]
        axis.plot(
            group.moneyness,
            100 * group.iv,
            ".-",
            color="#087f8c",
            markersize=3,
            label="BSM midpoint IV",
        )
        source = group[group.cboe_iv > 0]
        axis.plot(
            source.moneyness,
            100 * source.cboe_iv,
            "--",
            color="#b65321",
            linewidth=1,
            label="CBOE published IV",
        )
        axis.fill_between(
            group.moneyness,
            100 * group.iv_bid,
            100 * group.iv_ask,
            color="#087f8c",
            alpha=0.15,
            label="Bid/ask inversion range",
        )
        axis.set(
            title=f"{expiry} | {len(group)} OTM quotes",
            xlabel="Strike / snapshot spot",
            ylabel="Annual IV (%)",
            xlim=(run.config.plot_min_moneyness, run.config.plot_max_moneyness),
        )
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    for axis in axes[len(expiries) :]:
        axis.set_visible(False)
    capture_date = str(provenance.get("retrieved_at", "synthetic fixture"))[:10]
    valuation_date = str(provenance.get("valuation_timestamp", "synthetic fixture"))[:10]
    dates = f"Capture {capture_date} | spot-clock proxy {valuation_date}"
    figure.suptitle("SPX standard AM expiries: observed IV skew (no surface fit)\n" + dates)
    figure.savefig(figures / "iv_smile.png", dpi=150, metadata={"Software": "crossprice"})
    figure = Figure(figsize=(9, 4.5), layout="constrained")
    axis = figure.subplots()
    term = run.term_structure
    if not term.empty and "tau" in term:
        axis.plot(
            365 * term.tau,
            100 * term.atm_iv,
            "o-",
            color="#087f8c",
            label="Nearest spot-ATM OTM quote",
        )
        axis.plot(
            365 * term.tau,
            100 * term.wing90_iv,
            "s--",
            color="#b65321",
            label="Nearest 90% moneyness put",
        )
        if "atm_iv_bid" in term and "atm_iv_ask" in term:
            axis.fill_between(
                365 * term.tau,
                100 * term.atm_iv_bid,
                100 * term.atm_iv_ask,
                color="#087f8c",
                alpha=0.15,
            )
        axis.legend()
    axis.set(
        xlabel="Calendar days to AM fixing proxy (ACT/365F)",
        ylabel="Annual IV (%)",
        title="SPX observed ATM term structure and 90% downside wing\n" + dates,
    )
    axis.grid(alpha=0.2)
    figure.savefig(figures / "iv_term_structure.png", dpi=150, metadata={"Software": "crossprice"})
    record = dict(
        provenance=provenance,
        config=asdict(run.config),
        summary=surface_summary(run),
        versions={
            package: version(package) for package in ("numpy", "scipy", "pandas", "matplotlib")
        },
        artifacts=[
            *frames,
            "figures/iv_smile.png",
            "figures/iv_term_structure.png",
            "surface_run.json",
        ],
        limitations=[
            "A single delayed snapshot; same-payload spot may be stale and quotes asynchronous.",
            "Valuation clock is the spot timestamp proxy; "
            "actual quote update times are unavailable.",
            "Par yields used as zero-rate proxies, continuous carry from one ATM pair per expiry.",
            "AM opening-fixing time approximated by 09:30 ET; payment lag is omitted.",
            "Greedy midpoint shape screen, not bid/ask-feasibility or a maximal clean subset.",
            "No cross-expiry arbitrage-free surface or density fitted; lines join observed quotes.",
            "CBOE inputs and methodology unknown; published IV is a comparator, not ground truth.",
            "External trailing dividend yield is a sensitivity proxy, "
            "not a forward dividend forecast.",
            "Bid/ask inversion ranges are quote ranges, not statistical confidence intervals.",
        ],
    )
    (out / "surface_run.json").write_text(
        json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return record


def main(argv: list[str] | None = None) -> int:
    """Regenerate the surface artifacts using only the committed CSV and sidecars."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("docs"))
    args = parser.parse_args(argv)
    metadata_path = args.snapshot.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text())
    curve_path = args.snapshot.parent / metadata["treasury_csv"]
    chain = pd.read_csv(args.snapshot, float_precision="round_trip")
    curve = pd.read_csv(curve_path, float_precision="round_trip")
    run = analyze_surface(chain, curve, metadata["external_dividend_yield"])
    provenance = metadata | {
        "input_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (args.snapshot, metadata_path, curve_path)
        }
    }
    record = write_artifacts(run, args.out, provenance)
    print(
        json.dumps(
            {
                key: record["summary"][key]
                for key in (
                    "rows",
                    "status_counts",
                    "reason_counts",
                    "plotted_quotes",
                    "max_price_residual",
                    "method_counts",
                )
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(
        run.term_structure[["expiry", "atm_iv", "wing90_iv", "wing90_minus_atm_pp"]].to_string(
            index=False
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
