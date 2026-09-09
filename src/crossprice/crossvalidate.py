"""Cross-validation harness: agreement table, stress cases and convergence plots.

Regenerate every reported artifact with one command from the repository root::

    uv run python -m crossprice.crossvalidate --out docs

That writes ``docs/agreement_table.csv`` (every row), ``docs/agreement_table.md``
(the per-method summary and stress rows quoted in the README),
``docs/crossvalidation_run.json`` (configuration, thresholds, environment and
summary numbers) and the two convergence figures with their data under
``docs/figures/``. ``--quick`` runs a coarse smoke configuration with its own
scaled tolerances; it is for checking the machinery, not for reporting.

The harness prices a reproducible European grid, an American grid, a small
discrete Asian set and explicitly declared stress cases by every method the
library implements, then judges each row against a stated reference with
criteria fixed before any number is looked at:

- deterministic methods against an exact or fine-tree reference use
  ``|error| <= atol + rtol * |reference|``, or the much tighter
  ``exact_atol``/``exact_rtol`` when both sides are deterministic
  (``tau == 0`` or ``vol == 0``);
- stochastic methods use ``z = (price - reference) / SE`` against a per-row
  threshold ``ndtri(1 - family_alpha / (2 n))`` where ``n`` is the number of
  stochastic rows declared before the run (a Bonferroni family-wise false-alarm
  budget), and separately record whether each 95% interval covered the
  reference so the family coverage can be compared with Binomial(n, 0.95);
- degenerate samples (zero SE from no variation, or an SE below the
  floating-point resolution floor because a control variate was exact on every
  sampled path), zero references (relative error undefined), pricer rejections
  (``ValueError``/``FloatingPointError``) and unsupported instrument/method
  pairs are recorded as such, never converted into finite z-scores or dropped.
  Any other exception propagates.

References: ``bsm_price`` is the European reference only. American rows are
compared with a labelled fine CRR tree, which is a resolution self-consistency
check rather than an independent benchmark. Arithmetic Asians have no exact
reference here; the discrete geometric closed form is used as an AM-GM bound
(lower for calls, upper for puts), not as ground truth.

Every method row declares the outcome it expects. The ``status`` column then
reads: ``pass`` (agreement where agreement was expected), ``expected`` (a
declared rejection, degeneracy or unsupported pair occurred as declared),
``degenerate`` (an undeclared degenerate sample: logged with its mechanism,
not evidence of agreement, not an implementation failure) and ``FAIL``
(everything else, including a declared failure that did not occur).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import scipy
from scipy.special import ndtri
from scipy.stats import binom

from crossprice import __version__
from crossprice.analytic import OptionKind, bsm_price, geometric_asian_price
from crossprice.binomial import crr_price
from crossprice.montecarlo import MCResult, asian_mc_price, mc_price

type Instrument = Literal["european", "american", "asian"]
type Outcome = Literal["agree", "disagree", "degenerate", "error", "unsupported"]
type Status = Literal["pass", "expected", "degenerate", "FAIL"]
type Criterion = Literal["tolerance", "exact", "z_score", "lower_bound", "upper_bound", "none"]

COLUMNS: tuple[str, ...] = (
    "case_id",
    "group",
    "instrument",
    "kind",
    "spot",
    "strike",
    "tau",
    "rate",
    "vol",
    "div",
    "method",
    "settings",
    "seed",
    "steps",
    "n_paths",
    "n_samples",
    "pilot_paths",
    "total_paths",
    "antithetic",
    "control_variate",
    "control_beta",
    "variance_reduction",
    "n_dates",
    "average",
    "reference",
    "reference_price",
    "price",
    "error",
    "abs_error",
    "rel_error",
    "se",
    "ci_low",
    "ci_high",
    "ci_status",
    "z_score",
    "covered_95",
    "criterion",
    "outcome",
    "expected",
    "status",
    "reason",
)
_INT_COLUMNS = ("seed", "steps", "n_paths", "n_samples", "pilot_paths", "total_paths", "n_dates")
_BOOL_COLUMNS = ("antithetic", "control_variate", "covered_95")

DETERMINISTIC_METHODS = frozenset({"bsm", "crr", "crr_american"})
STOCHASTIC_METHODS = frozenset(
    {"mc_plain", "mc_antithetic_control", "asian_mc_geometric", "asian_mc_arithmetic_control"}
)
METHODS = DETERMINISTIC_METHODS | STOCHASTIC_METHODS
SUPPORTED: dict[str, frozenset[str]] = {
    "european": frozenset({"crr", "mc_plain", "mc_antithetic_control"}),
    "american": frozenset({"crr_american"}),
    "asian": frozenset({"asian_mc_geometric", "asian_mc_arithmetic_control"}),
}
UNSUPPORTED_REASONS: dict[tuple[str, str], str] = {
    ("european", "bsm"): "bsm_price is the European reference itself; see reference_price",
    ("american", "bsm"): "BSM prices European exercise only; it is not an American reference",
    ("american", "mc_plain"): "no early-exercise Monte Carlo (Longstaff-Schwartz) is implemented",
    ("american", "mc_antithetic_control"): "no early-exercise Monte Carlo is implemented",
    ("asian", "bsm"): (
        "no elementary closed form for an arithmetic average; the geometric formula is a "
        "control and bound, not the arithmetic price"
    ),
    ("asian", "crr"): "the recombining CRR tree carries no running-average state",
    ("asian", "crr_american"): "the recombining CRR tree carries no running-average state",
    ("asian", "mc_plain"): "use asian_mc_geometric / asian_mc_arithmetic_control for path averages",
    ("asian", "mc_antithetic_control"): "use the Asian Monte Carlo methods for path averages",
    (
        "european",
        "crr_american",
    ): "American exercise is a different contract; see the American grid",
    (
        "european",
        "asian_mc_geometric",
    ): "Asian averaging is a different contract; see the Asian set",
    ("european", "asian_mc_arithmetic_control"): "Asian averaging is a different contract",
    ("american", "asian_mc_geometric"): "Asian averaging is a different contract",
    ("american", "asian_mc_arithmetic_control"): "Asian averaging is a different contract",
}


@dataclass(frozen=True)
class HarnessConfig:
    """Method settings, acceptance thresholds and experiment definitions.

    ``atol``/``rtol`` are the NumPy-style deterministic tolerances at
    ``crr_steps`` steps; ``exact_atol``/``exact_rtol`` apply when both sides are
    deterministic. ``family_alpha`` is the two-sided family-wise false-alarm
    budget spread over every declared stochastic row. A stochastic SE at or
    below ``se_floor_relative`` times the price scale cannot be a sampling
    standard error in double precision; such rows are numerically degenerate
    (a control variate reproduced every sampled payoff exactly) and get no
    z-score. The convergence fields reproduce the committed Step 2 and Step 4
    experiments by default.
    """

    crr_steps: int = 1000
    american_reference_steps: int = 6400
    mc_paths: int = 65_536
    pilot_paths: int = 4096
    asian_dates: int = 12
    base_seed: int = 20260916
    atol: float = 5e-3
    rtol: float = 2e-3
    exact_atol: float = 1e-10
    exact_rtol: float = 1e-12
    family_alpha: float = 0.01
    se_floor_relative: float = 1e-12
    crr_convergence_steps: tuple[int, ...] = (50, 100, 200, 400, 800)
    mc_convergence_paths: tuple[int, ...] = (1024, 4096, 16384, 65536)
    mc_convergence_seeds: int = 128
    mc_convergence_first_seed: int = 1000


def quick_config() -> HarnessConfig:
    """Coarse smoke configuration for tests and ``--quick`` runs.

    The tree tolerances are scaled to the coarser step count (CRR error is
    O(1/steps)); they are not the reported tolerances and this configuration
    is never used for the committed table.
    """
    return HarnessConfig(
        crr_steps=200,
        american_reference_steps=1600,
        mc_paths=4096,
        pilot_paths=512,
        atol=2.5e-2,
        rtol=1e-2,
        crr_convergence_steps=(20, 40, 80),
        mc_convergence_paths=(256, 1024, 4096),
        mc_convergence_seeds=16,
    )


@dataclass(frozen=True)
class Contract:
    """Scalar contract inputs in the shared library parameter order."""

    spot: float
    strike: float
    tau: float
    rate: float
    vol: float
    div: float = 0.0
    kind: OptionKind = "call"

    def args(self) -> tuple[float, float, float, float, float, float, OptionKind]:
        return (self.spot, self.strike, self.tau, self.rate, self.vol, self.div, self.kind)


@dataclass(frozen=True)
class MethodSpec:
    """One method row: name, per-row setting overrides and the declared outcome."""

    method: str
    settings: Mapping[str, Any] = field(default_factory=dict)
    expected: Outcome = "agree"


@dataclass(frozen=True)
class Case:
    """A contract, its instrument type and the method rows to evaluate for it."""

    case_id: str
    group: str
    instrument: Instrument
    contract: Contract
    methods: tuple[MethodSpec, ...]
    note: str = ""


@dataclass(frozen=True)
class GridSpec:
    """Factor levels of the European grid; spot is fixed and strikes vary."""

    spot: float = 100.0
    strikes: tuple[float, ...] = (80.0, 100.0, 120.0)
    taus: tuple[float, ...] = (0.25, 1.0, 2.0)
    vols: tuple[float, ...] = (0.10, 0.30)
    carries: tuple[tuple[float, float], ...] = ((0.05, 0.0), (0.01, 0.03))
    kinds: tuple[OptionKind, ...] = ("call", "put")


@dataclass(frozen=True)
class AgreementRun:
    """The agreement table with the thresholds that were fixed before pricing."""

    table: pd.DataFrame
    config: HarnessConfig
    z_crit: float
    n_stochastic: int


def _european_methods() -> tuple[MethodSpec, ...]:
    return (MethodSpec("crr"), MethodSpec("mc_plain"), MethodSpec("mc_antithetic_control"))


def _american_methods() -> tuple[MethodSpec, ...]:
    return (
        MethodSpec("bsm", expected="unsupported"),
        MethodSpec("crr_american"),
        MethodSpec("mc_plain", expected="unsupported"),
    )


def _asian_methods(n_dates: int) -> tuple[MethodSpec, ...]:
    return (
        MethodSpec("bsm", expected="unsupported"),
        MethodSpec("crr", expected="unsupported"),
        MethodSpec("asian_mc_geometric", {"n_dates": n_dates}),
        MethodSpec("asian_mc_arithmetic_control", {"n_dates": n_dates}),
    )


def european_grid(spec: GridSpec | None = None) -> list[Case]:
    """Full factorial European grid; every case gets CRR, plain MC and reduced MC."""
    spec = spec or GridSpec()
    cases = []
    for kind in spec.kinds:
        for strike in spec.strikes:
            for tau in spec.taus:
                for vol in spec.vols:
                    for rate, div in spec.carries:
                        case_id = f"eu-{kind}-K{strike:g}-T{tau:g}-v{vol:g}-r{rate:g}-q{div:g}"
                        contract = Contract(spec.spot, strike, tau, rate, vol, div, kind)
                        cases.append(
                            Case(case_id, "european", "european", contract, _european_methods())
                        )
    return cases


def american_grid(
    strikes: tuple[float, ...] = (80.0, 100.0, 120.0),
    taus: tuple[float, ...] = (0.25, 1.0),
    carries: tuple[tuple[float, float], ...] = ((0.05, 0.0), (0.02, 0.05)),
    vol: float = 0.2,
    kinds: tuple[OptionKind, ...] = ("call", "put"),
) -> list[Case]:
    """American grid compared with a fine tree of the same implementation."""
    cases = []
    for kind in kinds:
        for strike in strikes:
            for tau in taus:
                for rate, div in carries:
                    case_id = f"am-{kind}-K{strike:g}-T{tau:g}-v{vol:g}-r{rate:g}-q{div:g}"
                    contract = Contract(100.0, strike, tau, rate, vol, div, kind)
                    cases.append(
                        Case(case_id, "american", "american", contract, _american_methods())
                    )
    return cases


def asian_cases(
    strikes: tuple[float, ...] = (90.0, 100.0, 110.0),
    n_dates: int = 12,
    kinds: tuple[OptionKind, ...] = ("call", "put"),
) -> list[Case]:
    """Discrete Asian set: geometric MC against its closed form, arithmetic against the bound."""
    return [
        Case(
            f"asian-{kind}-K{strike:g}-m{n_dates}",
            "asian",
            "asian",
            Contract(100.0, strike, 1.0, 0.05, 0.2, 0.02, kind),
            _asian_methods(n_dates),
        )
        for kind in kinds
        for strike in strikes
    ]


_STRESS_TREE = {"steps": 1000}
_STRESS_MC = {"n_paths": 65_536, "pilot_paths": 4096}


def _stress_methods(reduced_expected: Outcome = "agree") -> tuple[MethodSpec, ...]:
    return (
        MethodSpec("crr", _STRESS_TREE),
        MethodSpec("mc_plain", _STRESS_MC),
        MethodSpec("mc_antithetic_control", _STRESS_MC, expected=reduced_expected),
    )


def stress_cases() -> list[Case]:
    """Explicitly identified corners, each declaring the outcome it expects.

    Stress rows carry explicit tree/path settings so they are identical under
    every configuration. Where a payoff is in the money on essentially every
    path, the discounted-terminal control variate reproduces it exactly and
    the controlled estimator is declared numerically degenerate in advance:
    its roundoff-level SE says nothing about the unsampled rare region.
    """
    eu = _stress_methods
    day = 1 / 365
    return [
        Case(
            "stress-expiry_otm_call",
            "stress",
            "european",
            Contract(100, 110, 0.0, 0.05, 0.2, 0.0, "call"),
            eu(),
            "tau=0 and out of the money: the reference price is exactly zero, so the relative "
            "error is undefined; every method must return the intrinsic value exactly",
        ),
        Case(
            "stress-expiry_itm_put",
            "stress",
            "european",
            Contract(100, 110, 0.0, 0.05, 0.2, 0.0, "put"),
            eu(),
            "tau=0 in the money: intrinsic value, exact criterion, zero SE without draws",
        ),
        Case(
            "stress-short_expiry_atm_call",
            "stress",
            "european",
            Contract(100, 100, day, 0.05, 0.2, 0.0, "call"),
            eu(),
            "one calendar day to expiry, at the money",
        ),
        Case(
            "stress-short_expiry_otm_call",
            "stress",
            "european",
            Contract(100, 103, day, 0.05, 0.2, 0.0, "call"),
            eu(),
            "one day, 3% out of the money: tiny reference price and a rare Monte Carlo payoff",
        ),
        Case(
            "stress-zero_vol_call",
            "stress",
            "european",
            Contract(100, 95, 1.0, 0.05, 0.0, 0.02, "call"),
            eu(),
            "vol=0: every method is deterministic; Monte Carlo returns zero SE without draws",
        ),
        Case(
            "stress-low_vol_coarse_tree",
            "stress",
            "european",
            Contract(100, 100, 1.0, 0.10, 0.02, 0.0, "call"),
            (MethodSpec("crr", {"steps": 16}, expected="error"), *eu("degenerate")),
            "vol=0.02 with r=0.10: the CRR probability leaves (0,1) unless steps > 25, so the "
            "16-step tree must raise rather than clip; the 1000-step tree and plain MC must "
            "agree; the option finishes in the money on essentially every path, so the "
            "controlled estimator is numerically degenerate",
        ),
        Case(
            "stress-deep_itm_call",
            "stress",
            "european",
            Contract(100, 20, 1.0, 0.05, 0.2, 0.0, "call"),
            eu("degenerate"),
            "deep in the money: the payoff is linear in the control on every sampled path, so "
            "the controlled estimator has roundoff-level SE and is declared degenerate",
        ),
        Case(
            "stress-deep_itm_put",
            "stress",
            "european",
            Contract(100, 300, 1.0, 0.05, 0.2, 0.0, "put"),
            eu("degenerate"),
            "deep in the money put: same perfect-control degeneracy; its tiny bias comes from "
            "the unsampled region where the put expires worthless",
        ),
        Case(
            "stress-deep_otm_call",
            "stress",
            "european",
            Contract(100, 200, 1.0, 0.05, 0.2, 0.02, "call"),
            eu(),
            "reference near 3e-3: relative error is large for every method; rare MC payoff",
        ),
        Case(
            "stress-deep_otm_put",
            "stress",
            "european",
            Contract(100, 50, 1.0, 0.05, 0.2, 0.02, "put"),
            eu(),
            "reference near 5e-4: only a handful of paths pay off, so the normal interval is "
            "poorly resolved",
        ),
        Case(
            "stress-degenerate_mc",
            "stress",
            "european",
            Contract(100, 1000, 1.0, 0.0, 0.2, 0.0, "call"),
            (
                MethodSpec("crr", _STRESS_TREE),
                MethodSpec("mc_plain", {"n_paths": 1000}, expected="degenerate"),
            ),
            "no positive payoff in 1000 paths although the BSM price is strictly positive: the "
            "zero-width interval is recorded as degenerate, not as certainty",
        ),
        Case(
            "stress-negative_rate_call",
            "stress",
            "european",
            Contract(100, 100, 1.0, -0.02, 0.2, 0.0, "call"),
            eu(),
            "negative risk-free rate",
        ),
        Case(
            "stress-negative_rate_put",
            "stress",
            "european",
            Contract(100, 100, 1.0, -0.02, 0.2, 0.0, "put"),
            eu(),
            "negative risk-free rate",
        ),
        Case(
            "stress-negative_carry_call",
            "stress",
            "european",
            Contract(100, 100, 1.0, 0.01, 0.2, 0.06, "call"),
            eu(),
            "dividend yield above the rate: negative carry",
        ),
        Case(
            "stress-negative_dividend_put",
            "stress",
            "european",
            Contract(100, 100, 1.0, 0.03, 0.2, -0.02, "put"),
            eu(),
            "negative dividend yield (a holding cost)",
        ),
        Case(
            "stress-high_vol_long_call",
            "stress",
            "european",
            Contract(100, 100, 5.0, 0.05, 1.0, 0.0, "call"),
            eu(),
            "vol=1 for five years: heavy-tailed payoffs stretch the normal-interval approximation",
        ),
        Case(
            "stress-american_negative_rate_call",
            "stress",
            "american",
            Contract(120, 100, 1.0, -0.05, 0.2, 0.0, "call"),
            (
                MethodSpec("bsm", expected="unsupported"),
                MethodSpec("crr_american", _STRESS_TREE),
                MethodSpec("mc_plain", expected="unsupported"),
            ),
            "r<0 with q=0: the American call exceeds the European one, so BSM is not a reference",
        ),
    ]


def default_cases() -> list[Case]:
    """Every case in the reported table, in table order."""
    return european_grid() + american_grid() + asian_cases() + stress_cases()


def quick_cases() -> list[Case]:
    """A reduced case list for smoke runs and tests; includes every stress case."""
    grid = GridSpec(strikes=(90.0, 110.0), taus=(1.0,), vols=(0.3,), carries=((0.05, 0.02),))
    return (
        european_grid(grid)
        + american_grid(strikes=(100.0,), taus=(1.0,), carries=((0.05, 0.0),), kinds=("put",))
        + asian_cases(strikes=(100.0,))
        + stress_cases()
    )


def seed_for(case_id: str, method: str, base_seed: int) -> int:
    """Deterministic 32-bit seed per (case, method) so rows are independent."""
    digest = hashlib.blake2b(f"{base_seed}|{case_id}|{method}".encode(), digest_size=4).digest()
    return int.from_bytes(digest, "big")


def _reference(case: Case, spec: MethodSpec, config: HarnessConfig) -> tuple[str, float]:
    contract = case.contract
    if case.instrument == "european":
        return "bsm_price", float(bsm_price(*contract.args()))
    if case.instrument == "american":
        steps = config.american_reference_steps
        price = crr_price(*contract.args(), steps=steps, american=True)
        return f"crr_price(american=True, steps={steps}) [same implementation]", price
    n_dates = int(spec.settings.get("n_dates", config.asian_dates))
    price = geometric_asian_price(*contract.args(), n_dates=n_dates)
    if spec.method == "asian_mc_arithmetic_control":
        return f"geometric_asian_price(n_dates={n_dates}) [AM-GM bound, not ground truth]", price
    return f"geometric_asian_price(n_dates={n_dates})", price


def _price(
    case: Case, spec: MethodSpec, config: HarnessConfig, seed: int | None
) -> tuple[float | MCResult, dict[str, Any]]:
    contract, settings = case.contract, spec.settings
    method = spec.method
    if method in ("crr", "crr_american"):
        steps = int(settings.get("steps", config.crr_steps))
        american = method == "crr_american"
        price = crr_price(*contract.args(), steps=steps, american=american)
        label = f"steps={steps}" + ("; american" if american else "")
        return price, {"steps": steps, "settings": label}
    if method == "bsm":
        return float(bsm_price(*contract.args())), {"settings": "closed form"}
    assert seed is not None
    n_paths = int(settings.get("n_paths", config.mc_paths))
    pilot = int(settings.get("pilot_paths", config.pilot_paths))
    if method == "mc_plain":
        result = mc_price(*contract.args(), seed=seed, n_paths=n_paths)
        return result, {"settings": f"n_paths={n_paths}"}
    if method == "mc_antithetic_control":
        result = mc_price(
            *contract.args(),
            seed=seed,
            n_paths=n_paths,
            antithetic=True,
            control_variate=True,
            pilot_paths=pilot,
        )
        return result, {"settings": f"n_paths={n_paths}+{pilot} pilot; antithetic; control"}
    n_dates = int(settings.get("n_dates", config.asian_dates))
    if method == "asian_mc_geometric":
        result = asian_mc_price(
            *contract.args(), seed=seed, n_paths=n_paths, n_dates=n_dates, average="geometric"
        )
        return result, {"settings": f"n_paths={n_paths}; n_dates={n_dates}; geometric"}
    result = asian_mc_price(
        *contract.args(),
        seed=seed,
        n_paths=n_paths,
        n_dates=n_dates,
        average="arithmetic",
        control_variate=True,
        pilot_paths=pilot,
    )
    label = f"n_paths={n_paths}+{pilot} pilot; n_dates={n_dates}; arithmetic; geometric control"
    return result, {"settings": label}


def _blank_row(case: Case, spec: MethodSpec) -> dict[str, Any]:
    contract = case.contract
    row: dict[str, Any] = dict.fromkeys(COLUMNS)
    row.update(
        case_id=case.case_id,
        group=case.group,
        instrument=case.instrument,
        kind=contract.kind,
        spot=contract.spot,
        strike=contract.strike,
        tau=contract.tau,
        rate=contract.rate,
        vol=contract.vol,
        div=contract.div,
        method=spec.method,
        settings="",
        expected=spec.expected,
        criterion="none",
        reason="",
    )
    for name in (
        "reference_price",
        "price",
        "error",
        "abs_error",
        "rel_error",
        "se",
        "ci_low",
        "ci_high",
        "z_score",
        "control_beta",
        "variance_reduction",
    ):
        row[name] = np.nan
    return row


def _finish(row: dict[str, Any], outcome: Outcome, reason: str = "") -> dict[str, Any]:
    row["outcome"] = outcome
    expected = row["expected"]
    if outcome == "agree" and expected == "agree":
        row["status"] = "pass"
    elif outcome == expected:
        row["status"] = "expected"
    elif outcome == "degenerate" and expected == "agree":
        row["status"] = "degenerate"
        reason = f"undeclared degenerate sample, logged: {reason}"
    else:
        row["status"] = "FAIL"
        reason = f"expected {expected}, got {outcome}" + (f": {reason}" if reason else "")
    row["reason"] = reason
    return row


def _evaluate(case: Case, spec: MethodSpec, config: HarnessConfig, z_crit: float) -> dict[str, Any]:
    if spec.method not in METHODS:
        raise ValueError(f"unknown method {spec.method!r}")
    row = _blank_row(case, spec)
    if spec.method not in SUPPORTED[case.instrument]:
        return _finish(row, "unsupported", UNSUPPORTED_REASONS[(case.instrument, spec.method)])
    try:
        row["reference"], reference = _reference(case, spec, config)
    except (ValueError, FloatingPointError) as exc:
        return _finish(row, "error", f"reference failed: {type(exc).__name__}: {exc}")
    row["reference_price"] = reference
    seed = None
    if spec.method in STOCHASTIC_METHODS:
        seed = row["seed"] = seed_for(case.case_id, spec.method, config.base_seed)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            result, extra = _price(case, spec, config, seed)
        except (ValueError, FloatingPointError) as exc:
            row.update(_requested_settings(spec, config))
            return _finish(row, "error", f"{type(exc).__name__}: {exc}")
    row.update(extra)
    notes = [str(warning.message).split(":")[0] for warning in caught]

    contract = case.contract
    deterministic_inputs = contract.tau == 0 or contract.vol == 0
    if isinstance(result, MCResult):
        row.update(
            price=result.price,
            se=result.se,
            ci_low=result.ci_low,
            ci_high=result.ci_high,
            ci_status=result.ci_status,
            n_paths=result.n_paths,
            n_samples=result.n_samples,
            pilot_paths=result.pilot_paths,
            total_paths=result.total_paths,
            antithetic=bool(result.antithetic),
            control_variate=bool(result.control_variate),
            control_beta=result.control_beta,
            variance_reduction=(
                np.nan if result.variance_reduction is None else result.variance_reduction
            ),
            n_dates=result.n_dates,
            average=result.average,
        )
        price = result.price
    else:
        price = float(result)
        row["price"] = price
    error = price - reference
    row.update(error=error, abs_error=abs(error))
    row["rel_error"] = error / reference if reference != 0 else np.nan

    if spec.method == "asian_mc_arithmetic_control":
        criterion: Criterion = "lower_bound" if contract.kind == "call" else "upper_bound"
    elif isinstance(result, MCResult):
        criterion = "exact" if result.ci_status == "deterministic" else "z_score"
    else:
        criterion = "exact" if deterministic_inputs else "tolerance"
    row["criterion"] = criterion

    if criterion in ("exact", "tolerance"):
        atol, rtol = (
            (config.exact_atol, config.exact_rtol)
            if criterion == "exact"
            else (config.atol, config.rtol)
        )
        bound = atol + rtol * abs(reference)
        if abs(error) <= bound:
            note = ""
            if criterion == "tolerance" and reference != 0 and abs(error / reference) > rtol:
                note = (
                    f"passes on atol: |rel_error|={abs(error / reference):.3g} exceeds "
                    f"rtol={rtol:g} because the reference price is small"
                )
            return _finish(row, "agree", "; ".join(filter(None, [note, *notes])))
        return _finish(
            row, "disagree", f"|error|={abs(error):.3g} exceeds atol+rtol*|ref|={bound:.3g}"
        )

    assert isinstance(result, MCResult)
    if result.ci_status == "degenerate":
        reason = (
            "no sample variation: SE=0, z undefined; a zero-width interval is not evidence of "
            f"certainty (reference={reference:.6g})"
        )
        return _finish(row, "degenerate", reason)
    scale = max(abs(price), abs(reference))
    if criterion == "z_score" and result.se <= config.se_floor_relative * scale:
        reason = (
            f"SE={result.se:.2g} is below the floating-point resolution floor "
            f"({config.se_floor_relative:g} x price scale): the control variate reproduced every "
            f"sampled payoff exactly, so z is undefined; the residual error={error:.3g} is the "
            "unsampled region where the payoff is not linear in the control"
        )
        return _finish(row, "degenerate", reason)
    if criterion == "z_score":
        z = error / result.se
        row["z_score"] = z
        row["covered_95"] = bool(result.ci_low <= reference <= result.ci_high)
        if abs(z) <= z_crit:
            return _finish(row, "agree", "; ".join(notes))
        return _finish(row, "disagree", f"|z|={abs(z):.2f} exceeds z_crit={z_crit:.2f}")
    slack = z_crit * result.se + config.exact_atol
    respected = error >= -slack if criterion == "lower_bound" else error <= slack
    side = "lower" if criterion == "lower_bound" else "upper"
    if respected:
        note = f"AM-GM {side} bound respected within z_crit*SE; no exact arithmetic reference"
        return _finish(row, "agree", "; ".join([note, *notes]))
    return _finish(row, "disagree", f"AM-GM {side} bound violated by more than z_crit*SE")


def _requested_settings(spec: MethodSpec, config: HarnessConfig) -> dict[str, Any]:
    """Settings a row asked for, recorded even when the pricer rejected the request."""
    if spec.method in ("crr", "crr_american"):
        steps = int(spec.settings.get("steps", config.crr_steps))
        return {"steps": steps, "settings": f"steps={steps}"}
    if spec.method in STOCHASTIC_METHODS:
        n_paths = int(spec.settings.get("n_paths", config.mc_paths))
        return {"n_paths": n_paths, "settings": f"n_paths={n_paths}"}
    return {"settings": "closed form"}


def run_agreement(cases: Sequence[Case], config: HarnessConfig | None = None) -> AgreementRun:
    """Evaluate every (case, method) row and return the typed agreement table.

    The z threshold is fixed from the number of stochastic rows declared in
    ``cases`` before any price is computed.
    """
    config = config or HarnessConfig()
    n_stochastic = sum(
        spec.method in STOCHASTIC_METHODS and spec.method in SUPPORTED[case.instrument]
        for case in cases
        for spec in case.methods
    )
    z_crit = float(ndtri(1 - config.family_alpha / (2 * max(n_stochastic, 1))))
    rows = [_evaluate(case, spec, config, z_crit) for case in cases for spec in case.methods]
    table = pd.DataFrame(rows, columns=list(COLUMNS))
    for name in _INT_COLUMNS:
        table[name] = table[name].astype("Int64")
    for name in _BOOL_COLUMNS:
        table[name] = table[name].astype("boolean")
    return AgreementRun(table, config, z_crit, n_stochastic)


def summarize_methods(table: pd.DataFrame) -> pd.DataFrame:
    """Per (instrument, method) counts and worst-case diagnostics.

    Error maxima cover only rows judged against a reference (exact, tolerance
    or z-score criteria); bound rows have no error to report.
    """
    records = []
    for (instrument, method), part in table.groupby(["instrument", "method"], sort=False):
        judged = part[part["criterion"].isin(["exact", "tolerance", "z_score"])]
        z_scores = part["z_score"].dropna()
        covered = part["covered_95"].dropna()
        rel = judged["rel_error"].abs().dropna()
        criteria = sorted(set(part["criterion"]) - {"none"})
        records.append(
            {
                "instrument": instrument,
                "method": method,
                "criteria": "/".join(criteria) if criteria else "n/a",
                "rows": len(part),
                "pass": int((part["status"] == "pass").sum()),
                "expected": int((part["status"] == "expected").sum()),
                "degenerate": int((part["status"] == "degenerate").sum()),
                "FAIL": int((part["status"] == "FAIL").sum()),
                "max_abs_error": float(judged["abs_error"].max())
                if judged["abs_error"].notna().any()
                else np.nan,
                "max_abs_rel_error": float(rel.max()) if len(rel) else np.nan,
                "max_abs_z": float(z_scores.abs().max()) if len(z_scores) else np.nan,
                "covered_95": f"{int(covered.sum())}/{len(covered)}" if len(covered) else "n/a",
            }
        )
    return pd.DataFrame.from_records(records)


def family_statistics(run: AgreementRun) -> dict[str, Any]:
    """Family-level statements about the stochastic rows and the status counts."""
    table = run.table
    z_scores = table["z_score"].dropna().to_numpy(dtype=float)
    covered = table["covered_95"].dropna().to_numpy(dtype=bool)
    n_intervals, n_covered = int(covered.size), int(covered.sum())
    if n_intervals:
        lower, upper = (int(value) for value in binom.interval(0.99, n_intervals, 0.95))
    else:
        lower, upper = 0, 0
    return {
        "n_stochastic_declared": run.n_stochastic,
        "family_alpha": run.config.family_alpha,
        "z_crit": run.z_crit,
        "n_z_scores": int(z_scores.size),
        "mean_z": float(np.mean(z_scores)) if z_scores.size else np.nan,
        "sd_z": float(np.std(z_scores, ddof=1)) if z_scores.size > 1 else np.nan,
        "max_abs_z": float(np.max(np.abs(z_scores))) if z_scores.size else np.nan,
        "n_intervals": n_intervals,
        "covered_95": n_covered,
        "binomial_99_band": [lower, upper],
        "coverage_within_band": bool(lower <= n_covered <= upper) if n_intervals else None,
        "status_counts": {k: int(v) for k, v in table["status"].value_counts().items()},
        "outcome_counts": {k: int(v) for k, v in table["outcome"].value_counts().items()},
        "n_rows": len(table),
    }


def _fmt(value: Any, spec: str = "") -> str:
    if value is None or value is pd.NA:
        return "—"
    if isinstance(value, float) and not np.isfinite(value):
        return "—"
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "no"
    if spec and isinstance(value, (int, float, np.integer, np.floating)):
        return format(value, spec)
    return str(value)


def _markdown_table(frame: pd.DataFrame, formats: Mapping[str, str]) -> str:
    header = "| " + " | ".join(frame.columns) + " |"
    rule = "|" + "|".join(["---"] * len(frame.columns)) + "|"
    lines = [header, rule]
    for _, record in frame.iterrows():
        cells = [_fmt(record[name], formats.get(name, "")) for name in frame.columns]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


STRESS_COLUMNS = (
    "case_id",
    "method",
    "settings",
    "reference_price",
    "price",
    "error",
    "rel_error",
    "se",
    "z_score",
    "status",
    "reason",
)


def render_markdown(run: AgreementRun) -> str:
    """The README block: thresholds, per-method summary, family check, stress rows."""
    config, table = run.config, run.table
    summary = summarize_methods(table)
    family = family_statistics(run)
    stress = table[table["group"] == "stress"][list(STRESS_COLUMNS)]
    formats = {
        "max_abs_error": ".3g",
        "max_abs_rel_error": ".3g",
        "max_abs_z": ".2f",
        "reference_price": ".6g",
        "price": ".6g",
        "error": ".3g",
        "rel_error": ".3g",
        "se": ".3g",
        "z_score": ".2f",
    }
    counts = family["status_counts"]
    band = family["binomial_99_band"]
    lines = [
        f"Rows: {family['n_rows']} — pass {counts.get('pass', 0)}, expected "
        f"{counts.get('expected', 0)}, degenerate {counts.get('degenerate', 0)}, "
        f"FAIL {counts.get('FAIL', 0)}.",
        "",
        "`pass`: agreement under the criterion below. `expected`: a declared rejection, "
        "degeneracy or unsupported instrument/method pair occurred as declared. `degenerate`: an "
        "undeclared Monte Carlo sample with no usable variation (no payoff observed, or a control "
        "variate exact on every path), logged with its mechanism and not counted as agreement. "
        "`FAIL`: anything else, including a declared failure that did not occur.",
        "",
        f"Criteria fixed before pricing: deterministic rows `|error| <= {config.atol:g} + "
        f"{config.rtol:g}*|reference|` at {config.crr_steps} tree steps (exact rows "
        f"`{config.exact_atol:g} + {config.exact_rtol:g}*|reference|`); stochastic rows "
        f"`|z| <= {run.z_crit:.2f}` from a two-sided family-wise false-alarm budget of "
        f"{config.family_alpha:g} over {run.n_stochastic} declared stochastic rows; "
        f"Monte Carlo uses {config.mc_paths} paths (+{config.pilot_paths} independent pilot "
        f"paths when a control variate is fitted).",
        "",
        "### Per-method summary",
        "",
        _markdown_table(summary, formats),
        "",
        f"95% intervals covering the reference: {family['covered_95']}/{family['n_intervals']}; "
        f"the Binomial({family['n_intervals']}, 0.95) 99% band is [{band[0]}, {band[1]}]"
        f"{' (inside)' if family['coverage_within_band'] else ' (OUTSIDE)'}. "
        f"z-scores: mean {_fmt(family['mean_z'], '.3f')}, SD {_fmt(family['sd_z'], '.3f')}, "
        f"max |z| {_fmt(family['max_abs_z'], '.2f')}.",
        "",
        "### Stress cases",
        "",
        _markdown_table(stress, formats),
    ]
    return "\n".join(lines)


CRR_CONVERGENCE_CONTRACTS: dict[str, Contract] = {
    "atm": Contract(100.0, 100.0, 1.0, 0.05, 0.2, 0.0),
    "off_strike": Contract(100.0, 110.0, 1.0, 0.05, 0.2, 0.02),
}
MC_CONVERGENCE_CONTRACT = Contract(100.0, 100.0, 1.0, 0.05, 0.2, 0.02)


def crr_convergence(config: HarnessConfig | None = None) -> pd.DataFrame:
    """Signed CRR errors on even/odd step subsequences; orders fitted only at the money.

    Reproduces ``test_atm_convergence_order_and_oscillation``: the strike sits
    on a node for even steps and between nodes for odd steps, so each parity
    is a controlled subsequence. Off-strike rows show that the sign pattern is
    not universal and carry no fitted order.
    """
    config = config or HarnessConfig()
    records = []
    for label, base in CRR_CONVERGENCE_CONTRACTS.items():
        for kind in ("call", "put"):
            contract = replace(base, kind=kind)
            reference = float(bsm_price(*contract.args()))
            for parity, offset in (("even", 0), ("odd", 1)):
                steps = np.asarray(config.crr_convergence_steps, dtype=int) + offset
                prices = np.array([crr_price(*contract.args(), steps=int(n)) for n in steps])
                errors = prices - reference
                order = np.nan
                if label == "atm":
                    order = -float(np.polyfit(np.log(steps), np.log(np.abs(errors)), 1)[0])
                for n, price, error in zip(steps, prices, errors, strict=True):
                    records.append(
                        {
                            "contract": label,
                            "kind": kind,
                            "parity": parity,
                            "steps": int(n),
                            **{k: v for k, v in asdict(contract).items() if k != "kind"},
                            "reference": reference,
                            "price": float(price),
                            "error": float(error),
                            "abs_error": float(abs(error)),
                            "fitted_order": order,
                        }
                    )
    return pd.DataFrame.from_records(records)


def mc_convergence(config: HarnessConfig | None = None) -> pd.DataFrame:
    """RMS Monte Carlo error over independent seeds at each path count.

    Reproduces ``test_rms_error_converges_as_inverse_square_root``: plain
    sampling, seeds ``first_seed .. first_seed + n_seeds - 1`` at every path
    count, slope fitted to log RMS error against log paths. The mean reported
    SE is recorded beside the RMS error as a calibration check.
    """
    config = config or HarnessConfig()
    records = []
    for kind in ("call", "put"):
        contract = replace(MC_CONVERGENCE_CONTRACT, kind=kind)
        reference = float(bsm_price(*contract.args()))
        per_count = []
        for n_paths in config.mc_convergence_paths:
            results = [
                mc_price(
                    *contract.args(), seed=config.mc_convergence_first_seed + s, n_paths=n_paths
                )
                for s in range(config.mc_convergence_seeds)
            ]
            errors = np.array([r.price for r in results]) - reference
            per_count.append(
                {
                    "kind": kind,
                    "n_paths": int(n_paths),
                    "n_seeds": config.mc_convergence_seeds,
                    "first_seed": config.mc_convergence_first_seed,
                    **{k: v for k, v in asdict(contract).items() if k != "kind"},
                    "reference": reference,
                    "rms_error": float(np.sqrt(np.mean(np.square(errors)))),
                    "mean_error": float(np.mean(errors)),
                    "mean_se": float(np.mean([r.se for r in results])),
                }
            )
        slope = float(
            np.polyfit(
                np.log([r["n_paths"] for r in per_count]),
                np.log([r["rms_error"] for r in per_count]),
                1,
            )[0]
        )
        for record in per_count:
            record["fitted_slope"] = slope
        records.extend(per_count)
    return pd.DataFrame.from_records(records)


def _plain_ticks(axis: Any, values: Sequence[int]) -> None:
    from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator

    axis.xaxis.set_major_locator(FixedLocator(list(values)))
    axis.set_xticks(list(values), [str(v) for v in values])
    axis.xaxis.set_minor_locator(NullLocator())
    axis.xaxis.set_minor_formatter(NullFormatter())


def plot_crr_convergence(frame: pd.DataFrame, path: Path) -> None:
    """Two panels: log-log |error| with fitted ATM orders; signed errors incl. off-strike."""
    from matplotlib.figure import Figure

    fig = Figure(figsize=(11.5, 4.8))
    left, right = fig.subplots(1, 2)
    styles = {
        ("call", "even"): "o-",
        ("call", "odd"): "s--",
        ("put", "even"): "^-",
        ("put", "odd"): "v--",
    }
    atm = frame[frame["contract"] == "atm"]
    contract = CRR_CONVERGENCE_CONTRACTS["atm"]
    for (kind, parity), part in atm.groupby(["kind", "parity"], sort=False):
        order = float(part["fitted_order"].iloc[0])
        left.loglog(
            part["steps"],
            part["abs_error"],
            styles[(kind, parity)],
            label=f"{kind}, {parity} N: fitted order {order:.4f}",
            markerfacecolor="none" if (parity == "odd" or kind == "put") else None,
            markersize=9 if kind == "put" else 6,
        )
    anchor = atm[(atm["kind"] == "call") & (atm["parity"] == "even")]
    n0, e0 = float(anchor["steps"].iloc[0]), float(anchor["abs_error"].iloc[0])
    grid = np.array([atm["steps"].min(), atm["steps"].max()], dtype=float)
    left.loglog(grid, e0 * n0 / grid, "k:", linewidth=1, label="reference slope -1 (order 1)")
    left.set_xlabel("CRR steps N")
    left.set_ylabel("|CRR - BSM| (currency units)")
    left.set_title(
        f"ATM S=K={contract.spot:g}, T={contract.tau:g}, r={contract.rate:g}, "
        rf"$\sigma$={contract.vol:g}, q={contract.div:g}"
    )
    left.legend(fontsize=8)
    left.grid(True, which="both", alpha=0.3)
    left.text(
        0.02,
        0.03,
        "put markers overlay call markers: both methods satisfy put-call parity exactly",
        transform=left.transAxes,
        fontsize=7,
    )
    even_steps = sorted(set(atm.loc[atm["parity"] == "even", "steps"]))
    _plain_ticks(left, even_steps)

    off = CRR_CONVERGENCE_CONTRACTS["off_strike"]
    for label, part in frame[frame["kind"] == "call"].groupby("contract", sort=False):
        for parity, sub in part.groupby("parity", sort=False):
            name = (
                f"ATM K={contract.strike:g}, q={contract.div:g}"
                if label == "atm"
                else f"off-strike K={off.strike:g}, q={off.div:g}"
            )
            right.semilogx(
                sub["steps"],
                sub["error"],
                "o-" if parity == "even" else "s--",
                markerfacecolor="none" if parity == "odd" else None,
                label=f"{name}, {parity} N",
            )
    right.axhline(0.0, color="k", linewidth=0.8)
    right.set_xlabel("CRR steps N")
    right.set_ylabel("signed error CRR - BSM (call)")
    right.set_title("Even/odd straddling holds at the money, not off-strike")
    right.legend(fontsize=8)
    right.grid(True, which="both", alpha=0.3)
    _plain_ticks(right, even_steps)
    fig.suptitle("CRR European convergence; orders fitted only on the controlled ATM subsequences")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def plot_mc_convergence(frame: pd.DataFrame, path: Path) -> None:
    """Log-log RMS error over seeds versus paths, with mean reported SE and slope −1/2."""
    from matplotlib.figure import Figure

    fig = Figure(figsize=(7.2, 5.0))
    ax = fig.subplots()
    contract = MC_CONVERGENCE_CONTRACT
    n_seeds = int(frame["n_seeds"].iloc[0])
    first = int(frame["first_seed"].iloc[0])
    for kind, part in frame.groupby("kind", sort=False):
        slope = float(part["fitted_slope"].iloc[0])
        line = ax.loglog(
            part["n_paths"],
            part["rms_error"],
            "o-" if kind == "call" else "s-",
            label=f"{kind}: RMS error, fitted slope {slope:.4f}",
        )[0]
        ax.loglog(
            part["n_paths"],
            part["mean_se"],
            "--",
            color=line.get_color(),
            alpha=0.7,
            label=f"{kind}: mean reported SE",
        )
    anchor = frame[frame["kind"] == "call"].iloc[0]
    grid = np.array([frame["n_paths"].min(), frame["n_paths"].max()], dtype=float)
    ax.loglog(
        grid,
        float(anchor["rms_error"]) * np.sqrt(float(anchor["n_paths"]) / grid),
        "k:",
        linewidth=1,
        label="reference slope -1/2",
    )
    ax.set_xlabel("Monte Carlo paths N (plain sampling)")
    ax.set_ylabel(f"RMS error over {n_seeds} seeds ({first}-{first + n_seeds - 1}), currency units")
    ax.set_title(
        f"Terminal GBM Monte Carlo vs BSM: S=K={contract.spot:g}, T={contract.tau:g}, "
        rf"r={contract.rate:g}, $\sigma$={contract.vol:g}, q={contract.div:g}"
    )
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    _plain_ticks(ax, sorted(set(frame["n_paths"])))
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def _environment() -> dict[str, str]:
    import matplotlib

    return {
        "crossprice": __version__,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pd.__version__,
        "matplotlib": matplotlib.__version__,
    }


def write_artifacts(out_dir: Path, cases: Sequence[Case], config: HarnessConfig) -> dict[str, Any]:
    """Run everything and write table, markdown, JSON record, figures and figure data."""
    out_dir = Path(out_dir)
    figures = out_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    run = run_agreement(cases, config)
    crr_frame = crr_convergence(config)
    mc_frame = mc_convergence(config)

    run.table.to_csv(out_dir / "agreement_table.csv", index=False)
    (out_dir / "agreement_table.md").write_text(render_markdown(run) + "\n", encoding="utf-8")
    crr_frame.to_csv(figures / "crr_convergence.csv", index=False)
    mc_frame.to_csv(figures / "mc_convergence.csv", index=False)
    plot_crr_convergence(crr_frame, figures / "crr_convergence.png")
    plot_mc_convergence(mc_frame, figures / "mc_convergence.png")

    orders = {
        f"{kind}_{parity}": float(part["fitted_order"].iloc[0])
        for (kind, parity), part in crr_frame[crr_frame["contract"] == "atm"].groupby(
            ["kind", "parity"], sort=False
        )
    }
    slopes = {kind: float(part["fitted_slope"].iloc[0]) for kind, part in mc_frame.groupby("kind")}
    record = {
        "command": "uv run python -m crossprice.crossvalidate --out docs",
        "config": asdict(config),
        "environment": _environment(),
        "n_cases": len(cases),
        "family": family_statistics(run),
        "per_method": summarize_methods(run.table).to_dict(orient="records"),
        "crr_convergence": {
            "contracts": {k: asdict(v) for k, v in CRR_CONVERGENCE_CONTRACTS.items()},
            "fitted_orders_atm": orders,
        },
        "mc_convergence": {
            "contract": asdict(MC_CONVERGENCE_CONTRACT),
            "fitted_slopes": slopes,
            "n_seeds": config.mc_convergence_seeds,
            "first_seed": config.mc_convergence_first_seed,
        },
        "artifacts": [
            "agreement_table.csv",
            "agreement_table.md",
            "crossvalidation_run.json",
            "figures/crr_convergence.png",
            "figures/crr_convergence.csv",
            "figures/mc_convergence.png",
            "figures/mc_convergence.csv",
        ],
    }
    (out_dir / "crossvalidation_run.json").write_text(
        json.dumps(record, indent=2, default=_json_default) + "\n", encoding="utf-8"
    )
    return record


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    raise TypeError(f"cannot serialise {type(value).__name__}")


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point; prints the markdown summary and elapsed time."""
    parser = argparse.ArgumentParser(
        prog="python -m crossprice.crossvalidate", description=__doc__.split("\n\n")[0]
    )
    parser.add_argument("--out", type=Path, default=Path("docs"), help="output directory")
    parser.add_argument(
        "--quick", action="store_true", help="coarse smoke configuration (not the reported table)"
    )
    args = parser.parse_args(argv)
    config = quick_config() if args.quick else HarnessConfig()
    cases = quick_cases() if args.quick else default_cases()
    started = time.perf_counter()
    record = write_artifacts(args.out, cases, config)
    elapsed = time.perf_counter() - started
    print((args.out / "agreement_table.md").read_text(encoding="utf-8"))
    print(f"CRR fitted ATM orders: {record['crr_convergence']['fitted_orders_atm']}")
    print(f"MC fitted RMS slopes: {record['mc_convergence']['fitted_slopes']}")
    print(
        f"Wrote {len(record['artifacts'])} artifacts to {args.out} in {elapsed:.1f}s "
        "(wall time is machine-dependent)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
