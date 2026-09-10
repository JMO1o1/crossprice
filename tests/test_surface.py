"""Offline snapshot parsing and surface checks; no test accesses the network."""

from __future__ import annotations

import csv
import json
import runpy
import socket
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from crossprice.analytic import bsm_price
from crossprice.surface import (
    SurfaceConfig,
    analyze_surface,
    clean_quotes,
    infer_carry,
    main,
    screen_static_arbitrage,
    surface_summary,
    treasury_rate,
    write_artifacts,
)

FETCH = runpy.run_path(str(Path(__file__).parents[1] / "scripts" / "fetch_chain.py"))


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def blocked(*args: object, **kwargs: object) -> None:
        raise AssertionError("surface tests must not access the network")

    monkeypatch.setattr(socket.socket, "connect", blocked)


def test_occ_symbol_and_expiry_selection() -> None:
    assert FETCH["parse_symbol"]("SPX261016P06800000") == ("SPX", date(2026, 10, 16), "put", 6800.0)
    assert FETCH["parse_symbol"]("SPXW261016C07800000")[0] == "SPXW"
    with pytest.raises(ValueError, match="symbol"):
        FETCH["parse_symbol"]("malformed")
    expiries = {
        date.fromisoformat(value)
        for value in (
            "2026-09-11",
            "2026-09-18",
            "2026-10-16",
            "2026-11-20",
            "2026-12-18",
            "2027-01-15",
            "2027-03-19",
            "2027-06-17",
            "2027-09-17",
        )
    }
    selected = FETCH["select_expiries"](expiries, date(2026, 9, 9))
    assert selected == [
        date.fromisoformat(value)
        for value in (
            "2026-10-16",
            "2026-11-20",
            "2026-12-18",
            "2027-03-19",
            "2027-06-17",
            "2027-09-17",
        )
    ]
    with pytest.raises(ValueError, match="six"):
        FETCH["select_expiries"]({date(2026, 9, 11)}, date(2026, 9, 9))


def test_same_payload_spot_timestamps_settlement_and_raw_iv(tmp_path: Path) -> None:
    dates = ("261016", "261120", "261218", "270319", "270617", "270917")
    quotes = [
        dict(
            option=f"{root}{expiry}C07500000",
            bid=100,
            ask=102,
            bid_size=2,
            ask_size=3,
            volume=10,
            open_interest=20,
            iv=0.2,
            last_trade_price=150,
            last_trade_time="2026-09-08T13:00:00",
        )
        for root in ("SPX", "SPXW")
        for expiry in dates
    ]
    payload = dict(
        symbol="_SPX",
        timestamp="2026-09-10 12:00:00",
        data=dict(
            current_price=7636.36,
            last_trade_time="2026-09-09T16:14:59",
            options=quotes,
        ),
    )
    rows, metadata = FETCH["normalize_chain"](payload, datetime(2026, 9, 10, 12, tzinfo=UTC))
    assert len(rows) == 6
    assert {row["spot"] for row in rows} == {7636.36}
    assert {row["cboe_iv"] for row in rows} == {0.2}
    assert rows[0]["valuation_timestamp"] == "2026-09-09T16:14:59-04:00"
    assert rows[0]["snapshot_timestamp_raw"] == "2026-09-10 12:00:00"
    assert rows[0]["expiry_timestamp"] == "2026-10-16T09:30:00-04:00"
    assert rows[1]["expiry_timestamp"] == "2026-11-20T09:30:00-05:00"
    assert metadata["spot_age_at_retrieval_hours"] > 15
    target = tmp_path / "snapshot.csv"
    FETCH["write_csv"](target, rows, FETCH["COLUMNS"])
    with target.open() as source:
        reader = csv.DictReader(source)
        assert tuple(reader.fieldnames) == FETCH["COLUMNS"]
        assert len(list(reader)) == 6
    with pytest.raises(FileExistsError):
        FETCH["write_csv"](target, rows, FETCH["COLUMNS"])
    with pytest.raises(ValueError, match="_SPX"):
        FETCH["normalize_chain"](payload | {"symbol": "SPY"}, datetime.now(UTC))
    with pytest.raises(ValueError, match="spot"):
        FETCH["normalize_chain"](
            payload | {"data": payload["data"] | {"current_price": 0}}, datetime.now(UTC)
        )


def test_treasury_parser_uses_latest_observation_not_future() -> None:
    xml = b"""<feed xmlns:m="urn:metadata" xmlns:d="urn:data">
      <m:properties><d:NEW_DATE>2026-09-08T00:00:00</d:NEW_DATE>
        <d:BC_1MONTH>4.0</d:BC_1MONTH><d:BC_1YEAR>3.5</d:BC_1YEAR></m:properties>
      <m:properties><d:NEW_DATE>2026-09-09T00:00:00</d:NEW_DATE>
        <d:BC_1MONTH>4.1</d:BC_1MONTH><d:BC_1YEAR>3.6</d:BC_1YEAR></m:properties>
      <m:properties><d:NEW_DATE>2026-09-10T00:00:00</d:NEW_DATE>
        <d:BC_1MONTH>9.0</d:BC_1MONTH><d:BC_1YEAR>9.0</d:BC_1YEAR></m:properties>
    </feed>"""
    rows = FETCH["parse_treasury"](xml, date(2026, 9, 9))
    assert rows == [
        dict(date="2026-09-09", tenor_years=1 / 12, par_yield_pct=4.1),
        dict(date="2026-09-09", tenor_years=1.0, par_yield_pct=3.6),
    ]
    with pytest.raises(ValueError, match="observation"):
        FETCH["parse_treasury"](xml, date(2026, 1, 1))


@pytest.fixture
def synthetic_chain() -> pd.DataFrame:
    rows = []
    for kind in ("call", "put"):
        for strike in (80.0, 90.0, 100.0, 115.0, 130.0):
            mid = float(bsm_price(100, strike, 1, 0.04, 0.25, 0.015, kind))
            rows.append(
                dict(
                    option_symbol=f"{kind}-{strike}",
                    underlying="SPX",
                    settlement="AM",
                    spot=100.0,
                    strike=strike,
                    expiry="2027-09-09",
                    kind=kind,
                    bid=mid * 0.99,
                    ask=mid * 1.01,
                    bid_size=10,
                    ask_size=10,
                    volume=20,
                    open_interest=30,
                    cboe_iv=0.25,
                    valuation_timestamp="2026-09-09T09:30:00-04:00",
                    expiry_timestamp="2027-09-09T09:30:00-04:00",
                    last_trade_price=999,
                )
            )
    return pd.DataFrame(rows)


@pytest.fixture
def flat_curve() -> pd.DataFrame:
    return pd.DataFrame(dict(tenor_years=[1 / 12, 2], par_yield_pct=[200 * np.expm1(0.02)] * 2))


@pytest.mark.parametrize(
    "updates,reason",
    [
        ({"bid": np.nan}, "nonfinite_quote"),
        ({"spot": 0}, "invalid_contract"),
        ({"underlying": "SPY"}, "unsupported_contract"),
        ({"settlement": "PM"}, "unsupported_contract"),
        ({"expiry_timestamp": "bad"}, "invalid_timestamp"),
        ({"expiry_timestamp": "2026-09-10T09:30:00-04:00"}, "under_seven_days"),
        ({"bid": 5, "ask": 4}, "crossed_quote"),
        ({"bid": 0}, "nonpositive_bid"),
        ({"bid_size": 0}, "zero_size"),
        ({"ask_size": 0}, "zero_size"),
        ({"volume": 0}, "zero_volume"),
        ({"open_interest": 0}, "zero_open_interest"),
        ({"bid": 1, "ask": 5}, "wide_spread"),
    ],
)
def test_each_basic_cleaning_rule(
    synthetic_chain: pd.DataFrame, updates: dict, reason: str
) -> None:
    chain = synthetic_chain.copy(deep=True)
    for key, value in updates.items():
        chain.loc[0, key] = value
    original = chain.copy(deep=True)
    table = clean_quotes(chain)
    assert table.loc[0, "reason"] == reason
    assert table.loc[0, "status"] == "dropped"
    assert len(table) == len(chain)
    assert (table.loc[1:, "status"] == "candidate").all()
    assert_frame_equal(chain, original)


def test_midpoint_duplicates_and_missing_source_iv(synthetic_chain: pd.DataFrame) -> None:
    synthetic_chain.loc[0, "cboe_iv"] = np.nan
    table = clean_quotes(synthetic_chain)
    assert (table.status == "candidate").all()
    assert np.all(table.mid == (table.bid + table.ask) / 2)
    assert np.all(table.mid != 999)
    duplicate = clean_quotes(
        pd.concat([synthetic_chain, synthetic_chain.iloc[[0]]], ignore_index=True)
    )
    assert (duplicate.reason == "duplicate_contract").sum() == 2
    with pytest.raises(ValueError, match="missing"):
        clean_quotes(synthetic_chain.drop(columns="bid"))


def test_treasury_interpolation_and_no_extrapolation() -> None:
    curve = pd.DataFrame(dict(tenor_years=[0.5, 1], par_yield_pct=[4, 5]))
    assert treasury_rate(0.75, curve) == pytest.approx(2 * np.log1p(0.045 / 2), rel=0, abs=1e-15)
    with pytest.raises(ValueError, match="Treasury"):
        treasury_rate(0.1, curve)
    with pytest.raises(ValueError, match="Treasury"):
        treasury_rate(0.75, pd.concat([curve, curve]))


def test_parity_forward_and_carry_are_recovered(
    synthetic_chain: pd.DataFrame, flat_curve: pd.DataFrame
) -> None:
    table, carry = infer_carry(clean_quotes(synthetic_chain), flat_curve)
    assert carry.iloc[0]["status"] == "ok"
    assert carry.iloc[0]["pair_strike"] == 100
    assert carry.iloc[0]["div"] == pytest.approx(0.015, rel=0, abs=1e-14)
    assert carry.iloc[0]["forward"] == pytest.approx(100 * np.exp(0.025), rel=0, abs=1e-12)
    assert carry.iloc[0].forward_low < carry.iloc[0].forward < carry.iloc[0].forward_high
    assert carry.iloc[0].div_low < 0.015 < carry.iloc[0].div_high
    screened = screen_static_arbitrage(table)
    assert (screened.status == "candidate").all()
    assert_frame_equal(screened, table)


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize(
    "prices,reason",
    [
        ([16.0, 17.0, 6.0], "strike_monotonicity"),
        ([16.0, 13.0, 6.0], "strike_convexity"),
        ([30.0, 13.0, 6.0], "vertical_spread_bound"),
    ],
)
def test_static_screen_rechecks_unequal_strikes(
    kind: str, prices: list[float], reason: str
) -> None:
    table = pd.DataFrame(
        dict(
            expiry=["2027-09-09"] * 3,
            kind=[kind] * 3,
            strike=[90.0, 100.0, 120.0],
            mid=prices,
            relative_spread=[0.01, 0.2, 0.01],
            option_symbol=["a", "b", "c"],
            status=["candidate"] * 3,
            reason=[""] * 3,
            rate=[0.0] * 3,
            tau=[1.0] * 3,
        )
    )
    if kind == "put":
        if reason == "strike_monotonicity":
            table["mid"] = [1.0, 0.5, 9.0]
        elif reason == "strike_convexity":
            table["mid"] = [1.0, 5.0, 9.0]
        else:
            table["mid"] = [1.0, 15.0, 25.0]
    screened = screen_static_arbitrage(table)
    assert screened.loc[1, "reason"] == reason
    assert (screened.status == "candidate").sum() == 2
    assert_frame_equal(screen_static_arbitrage(screened), screened)
    reordered = screen_static_arbitrage(table.iloc[::-1]).sort_index()
    assert_frame_equal(reordered, screened)


def test_no_parity_pair_is_logged(synthetic_chain: pd.DataFrame, flat_curve: pd.DataFrame) -> None:
    calls = synthetic_chain[synthetic_chain.kind == "call"]
    table, carry = infer_carry(clean_quotes(calls), flat_curve)
    assert (table.status == "inversion_failed").all()
    assert (table.reason == "no_atm_parity_pair").all()
    assert carry.iloc[0]["status"] == "no_atm_parity_pair"


def test_synthetic_surface_inversions_and_sensitivity(
    synthetic_chain: pd.DataFrame, flat_curve: pd.DataFrame
) -> None:
    run = analyze_surface(synthetic_chain, flat_curve, 0.015)
    assert (run.quotes.status == "inverted").all()
    assert np.max(abs(run.quotes.iv - 0.25)) < 5e-10
    assert np.max(abs(run.quotes.source_iv_diff_pp)) < 5e-8
    assert np.max(abs(run.quotes.parity_price_residual)) < 1e-12
    assert (run.quotes.iv_bid < run.quotes.iv).all()
    assert (run.quotes.iv_ask > run.quotes.iv).all()
    assert len(run.sensitivity) == 9 * len(run.quotes)
    external = run.sensitivity[run.sensitivity.scenario == "external_div"]
    assert (external.status == "converged").all()
    assert np.max(abs(external.delta_iv_pp)) < 5e-8
    assert run.term_structure.iloc[0].atm_moneyness == 1
    assert run.term_structure.iloc[0].wing90_moneyness == 0.9
    assert abs(run.term_structure.iloc[0].wing90_minus_atm_pp) < 5e-8
    assert (run.smile.loc[run.smile.strike <= run.smile.forward, "kind"] == "put").all()
    fixed_forward = run.sensitivity[run.sensitivity.scenario == "rate_up_fixed_forward"]
    np.testing.assert_allclose(fixed_forward.rate - fixed_forward["div"], 0.025, rtol=0, atol=1e-14)
    for kind, rate_sign, div_sign in (("call", -1, 1), ("put", 1, -1)):
        varied = run.sensitivity[run.sensitivity.kind == kind]
        assert (
            rate_sign * varied.loc[varied.scenario == "rate_up_fixed_q", "delta_iv_pp"] > 0
        ).all()
        assert (div_sign * varied.loc[varied.scenario == "div_up", "delta_iv_pp"] > 0).all()


def test_failed_inversions_and_endpoints_are_not_hidden(
    synthetic_chain: pd.DataFrame, flat_curve: pd.DataFrame
) -> None:
    call = synthetic_chain[(synthetic_chain.kind == "call") & (synthetic_chain.strike == 80)].index[
        0
    ]
    synthetic_chain.loc[call, ["bid", "ask"]] = [15.49, 15.51]
    run = analyze_surface(synthetic_chain, flat_curve, 0.015)
    failures = run.quotes[run.quotes.status == "inversion_failed"]
    assert len(failures) >= 1
    assert failures.iv_detail.str.contains("no-arbitrage bounds").any()
    assert (failures.reason != "").all()
    assert set(run.quotes.status) <= {"dropped", "inverted", "inversion_failed"}
    assert len(run.quotes) == len(synthetic_chain)
    assert len(run.sensitivity) == 9 * (run.quotes.status != "dropped").sum()
    assert run.sensitivity.loc[run.sensitivity.status != "converged", "reason"].ne("").all()


def test_missing_atm_and_source_iv_are_reported(
    synthetic_chain: pd.DataFrame, flat_curve: pd.DataFrame
) -> None:
    synthetic_chain["cboe_iv"] = 0
    run = analyze_surface(synthetic_chain, flat_curve, 0.015)
    assert run.quotes.comparison_status.eq("source_iv_missing_or_nonpositive").all()
    assert run.quotes.source_iv_diff_pp.isna().all()
    sparse = synthetic_chain[synthetic_chain.strike != 100]
    run = analyze_surface(sparse, flat_curve, 0.015)
    assert run.term_structure.iloc[0].atm_status == "no_nearby_quote"
    assert pd.isna(run.term_structure.iloc[0].wing90_minus_atm_pp)


def test_artifacts_reproduce_bytes(
    synthetic_chain: pd.DataFrame, flat_curve: pd.DataFrame, tmp_path: Path
) -> None:
    run = analyze_surface(synthetic_chain, flat_curve, 0.015)
    first, second = tmp_path / "first", tmp_path / "second"
    record = write_artifacts(run, first, {"fixture": "synthetic"})
    write_artifacts(run, second, {"fixture": "synthetic"})
    assert len(record["artifacts"]) == 8
    for name in record["artifacts"]:
        assert (first / name).read_bytes() == (second / name).read_bytes()
    assert json.loads((first / "surface_run.json").read_text())["summary"] == surface_summary(run)
    assert len(pd.read_csv(first / "surface_quotes.csv")) == len(synthetic_chain)


@pytest.mark.parametrize(
    "settings", [{"min_days": 0}, {"rate_bump": np.nan}, {"plot_min_moneyness": 0.95}]
)
def test_invalid_surface_config(settings: dict) -> None:
    with pytest.raises(ValueError):
        SurfaceConfig(**settings)


def test_published_iv_does_not_calibrate_ours(
    synthetic_chain: pd.DataFrame,
    flat_curve: pd.DataFrame,
) -> None:
    baseline = analyze_surface(synthetic_chain, flat_curve, 0.015)
    synthetic_chain["cboe_iv"] += 0.1
    changed = analyze_surface(synthetic_chain, flat_curve, 0.015)
    assert_frame_equal(baseline.carry, changed.carry)
    np.testing.assert_array_equal(baseline.quotes.iv, changed.quotes.iv)
    np.testing.assert_allclose(
        changed.quotes.source_iv_diff_pp - baseline.quotes.source_iv_diff_pp,
        -10,
        rtol=0,
        atol=1e-12,
    )


def test_endpoint_failure_keeps_valid_midpoint(
    synthetic_chain: pd.DataFrame,
    flat_curve: pd.DataFrame,
) -> None:
    midpoint = (synthetic_chain.loc[0, "bid"] + synthetic_chain.loc[0, "ask"]) / 2
    lower = float(bsm_price(100, 80, 1, 0.04, 0, 0.015))
    synthetic_chain.loc[0, ["bid", "ask"]] = [lower - 0.01, 2 * midpoint - lower + 0.01]
    run = analyze_surface(synthetic_chain, flat_curve, 0.015)
    row = run.quotes.iloc[0]
    assert row.status == "inverted"
    assert row.bid_status == "ValueError"
    assert "no-arbitrage bounds" in row.bid_detail
    assert pd.isna(row.iv_bid)
    assert surface_summary(run)["endpoint_failure_counts"]["bid"] >= 1


def test_no_eligible_quotes_still_produce_an_audit(
    synthetic_chain: pd.DataFrame,
    flat_curve: pd.DataFrame,
    tmp_path: Path,
) -> None:
    synthetic_chain["volume"] = 0
    run = analyze_surface(synthetic_chain, flat_curve, 0.015)
    assert run.quotes.status.eq("dropped").all()
    assert run.carry.status.eq("no_eligible_quotes").all()
    assert run.smile.empty and run.sensitivity.empty
    record = write_artifacts(run, tmp_path, {})
    assert record["summary"]["max_price_residual"] is None
    assert record["summary"]["term_structure"][0]["atm_iv"] is None


def test_programming_errors_propagate(
    synthetic_chain: pd.DataFrame,
    flat_curve: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken(*args: object, **kwargs: object) -> None:
        raise RuntimeError("injected programming defect")

    monkeypatch.setattr("crossprice.surface.implied_vol", broken)
    with pytest.raises(RuntimeError, match="programming defect"):
        analyze_surface(synthetic_chain, flat_curve, 0.015)


def test_snapshot_cli_and_regression_evidence(tmp_path: Path) -> None:
    """Observed snapshot regression, not a universal claim about new chains."""
    root = Path(__file__).parents[1]
    snapshot = root / "data" / "SPX_20260910.csv"
    original = snapshot.read_bytes()
    assert main(["--snapshot", str(snapshot), "--out", str(tmp_path)]) == 0
    assert snapshot.read_bytes() == original
    record = json.loads((tmp_path / "surface_run.json").read_text())
    summary = record["summary"]
    assert summary["rows"] == 3946
    assert summary["status_counts"] == {"dropped": 2895, "inverted": 1038, "inversion_failed": 13}
    assert sum(summary["reason_counts"].values()) == 2895 + 13
    assert summary["plotted_quotes"] == 670
    assert summary["max_price_residual"] <= 1e-10
    quotes = pd.read_csv(tmp_path / "surface_quotes.csv", float_precision="round_trip")
    assert quotes.loc[quotes.status != "inverted", "reason"].notna().all()
    assert quotes.loc[quotes.status == "inversion_failed", "iv_detail"].str.contains("bounds").all()
    assert (quotes.loc[quotes.status == "inverted", "source_iv_diff_pp"].abs() > 1).any()
    for _, group in quotes[quotes.status == "inverted"].groupby(["expiry", "kind"]):
        group = group.sort_values("strike")
        slopes = np.diff(group.mid) / np.diff(group.strike)
        direction = -1 if group.kind.iloc[0] == "call" else 1
        assert np.all(direction * slopes >= -1e-8)
        assert np.all(direction * slopes <= np.exp(-group.rate.iloc[0] * group.tau.iloc[0]) + 1e-8)
        assert np.all(np.diff(slopes) >= -1e-8)
    sensitivity = pd.read_csv(tmp_path / "surface_sensitivity.csv")
    assert len(sensitivity) == 9459
    assert sensitivity.loc[sensitivity.status != "converged", "reason"].notna().all()
    assert all(
        item["atm_status"] == item["wing90_status"] == "converged"
        for item in summary["sensitivities"]
    )
    term = pd.read_csv(tmp_path / "figures" / "iv_term_structure.csv", float_precision="round_trip")
    expected = pd.read_csv(
        root / "docs" / "figures" / "iv_term_structure.csv", float_precision="round_trip"
    )
    np.testing.assert_allclose(term.atm_iv, expected.atm_iv, rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        term.wing90_minus_atm_pp, expected.wing90_minus_atm_pp, rtol=0, atol=1e-10
    )
    assert record["provenance"]["treasury_date"] == "2026-09-09"
    assert record["provenance"]["spot_age_at_retrieval_hours"] > 16
