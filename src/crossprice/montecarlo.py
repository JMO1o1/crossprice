"""Seeded risk-neutral Monte Carlo estimates with uncertainty on every price."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
from scipy.special import ndtri

from crossprice.analytic import FloatArray, OptionKind

type CIStatus = Literal["normal", "deterministic", "degenerate"]

_NORMAL_95 = float(ndtri(0.975))


@dataclass(frozen=True)
class MCResult:
    """A price estimate with sample SE and a two-sided 95% normal interval.

    A normal interval is asymptotic, not an exact finite-sample guarantee.
    'degenerate' means no variation was observed in a stochastic sample; that
    interval is not evidence of zero uncertainty. 'deterministic' is reserved
    for contracts without model randomness. n_paths counts requested main
    payoff evaluations, n_samples independent observations used for the SE.
    pilot_paths records extra coefficient-fitting evaluations. baseline_se is
    the estimated plain-MC SE at total_paths; variance_reduction is its squared
    ratio to se, or None when either empirical variance is unresolved/zero.
    Cost means payoff evaluations, not wall-clock time.
    """

    price: float
    se: float
    ci_low: float
    ci_high: float
    n_paths: int
    n_samples: int
    seed: int
    ci_status: CIStatus
    antithetic: bool = False
    control_variate: bool = False
    pilot_paths: int = 0
    control_beta: float = 0.0
    baseline_se: float | None = None
    variance_reduction: float | None = None

    @property
    def ci(self) -> tuple[float, float]:
        """Return the reported 95% confidence interval endpoints."""
        return self.ci_low, self.ci_high

    @property
    def total_paths(self) -> int:
        """Main plus pilot payoff evaluations for equal-cost comparisons."""
        return self.n_paths + self.pilot_paths


def _validate_count(value: int, name: str, minimum: int) -> None:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < minimum
    ):
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _validate_inputs(
    spot: float,
    strike: float,
    tau: float,
    rate: float,
    vol: float,
    div: float,
    kind: OptionKind,
    n_paths: int,
    seed: int,
) -> None:
    inputs = (spot, strike, tau, rate, vol, div)
    if any(np.ndim(value) != 0 for value in inputs):
        raise ValueError("Monte Carlo requires scalar contract inputs")
    if not np.all(np.isfinite(inputs)):
        raise ValueError("contract inputs must be finite")
    if spot <= 0 or strike <= 0 or tau < 0 or vol < 0:
        raise ValueError("spot/strike must be positive and tau/vol non-negative")
    if kind not in ("call", "put"):
        raise ValueError("kind must be 'call' or 'put'")
    _validate_count(n_paths, "n_paths", 2)
    _validate_count(seed, "seed", 0)


def _validate_reduction(
    n_paths: int, antithetic: bool, control_variate: bool, pilot_paths: int
) -> None:
    for name, flag in (("antithetic", antithetic), ("control_variate", control_variate)):
        if not isinstance(flag, (bool, np.bool_)):
            raise ValueError(f"{name} must be a boolean")
    if antithetic and (n_paths < 4 or n_paths % 2):
        raise ValueError("antithetic requires an even n_paths >= 4")
    if control_variate:
        _validate_count(pilot_paths, "pilot_paths", 2)
        if antithetic and (pilot_paths < 4 or pilot_paths % 2):
            raise ValueError("antithetic requires an even pilot_paths >= 4")


def _summarize(observations: FloatArray, *, n_paths: int, seed: int) -> MCResult:
    if not np.all(np.isfinite(observations)):
        raise FloatingPointError("nonfinite discounted payoff samples")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        scale = float(np.max(np.abs(observations))) or 1.0
        scaled = observations / scale
        price = float(scale * np.mean(scaled))
        se = float(scale * (np.std(scaled, ddof=1) / np.sqrt(observations.size)))
        half_width = float(_NORMAL_95 * np.float64(se))
    status: CIStatus = "normal"
    if se == 0 or np.all(observations == observations[0]):
        se = half_width = 0.0
        status = "degenerate"
        warnings.warn(
            "No sample variation: the normal confidence interval is degenerate "
            "and may miss rare events",
            RuntimeWarning,
            stacklevel=3,
        )
    return MCResult(
        price, se, price - half_width, price + half_width, n_paths, observations.size, seed, status
    )


def _fit_control(payoffs: FloatArray, controls: FloatArray) -> float:
    scale = float(max(np.max(np.abs(payoffs)), np.max(np.abs(controls)))) or 1.0
    scaled_payoffs, scaled_controls = payoffs / scale, controls / scale
    centered_payoffs = scaled_payoffs - np.mean(scaled_payoffs)
    centered_controls = scaled_controls - np.mean(scaled_controls)
    denominator = float(np.dot(centered_controls, centered_controls))
    if denominator == 0:
        return 0.0
    return float(np.dot(centered_controls, centered_payoffs) / denominator)


def _estimate_samples(
    raw_payoffs: FloatArray,
    raw_controls: FloatArray,
    expected_control: float,
    *,
    seed: int,
    pilot: tuple[FloatArray, FloatArray] | None = None,
) -> MCResult:
    observations = np.mean(raw_payoffs, axis=0)
    beta, pilot_paths = 0.0, 0
    if pilot is not None:
        pilot_payoffs, pilot_controls = pilot
        beta = _fit_control(np.mean(pilot_payoffs, axis=0), np.mean(pilot_controls, axis=0))
        pilot_paths = pilot_payoffs.size
        observations = observations - beta * (np.mean(raw_controls, axis=0) - expected_control)
    result = _summarize(observations, n_paths=raw_payoffs.size, seed=seed)
    scale = float(np.max(np.abs(raw_payoffs))) or 1.0
    marginal_variance = np.mean(np.var(raw_payoffs / scale, axis=1, ddof=1))
    baseline_se = float(scale * np.sqrt(marginal_variance / (raw_payoffs.size + pilot_paths)))
    reduction = (baseline_se / result.se) ** 2 if baseline_se > 0 and result.se > 0 else None
    return replace(
        result,
        antithetic=raw_payoffs.shape[0] == 2,
        control_variate=pilot is not None,
        pilot_paths=pilot_paths,
        control_beta=beta,
        baseline_se=baseline_se,
        variance_reduction=reduction,
    )


def _european_samples(
    spot: float,
    strike: float,
    tau: float,
    rate: float,
    vol: float,
    div: float,
    kind: OptionKind,
    *,
    n_paths: int,
    rng: np.random.Generator,
    antithetic: bool,
) -> tuple[FloatArray, FloatArray]:
    normals = rng.standard_normal(n_paths // 2 if antithetic else n_paths)
    normals = np.stack((normals, -normals)) if antithetic else normals[None, :]
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        discounted_terminal = spot * np.exp(
            (-div - 0.5 * vol**2) * tau + vol * np.sqrt(tau) * normals
        )
        direction = 1.0 if kind == "call" else -1.0
        payoffs = np.maximum(direction * (discounted_terminal - strike * np.exp(-rate * tau)), 0.0)
    return payoffs, discounted_terminal


def mc_price(
    spot: float,
    strike: float,
    tau: float,
    rate: float,
    vol: float,
    div: float = 0.0,
    kind: OptionKind = "call",
    *,
    seed: int,
    n_paths: int = 100_000,
    antithetic: bool = False,
    control_variate: bool = False,
    pilot_paths: int = 4096,
) -> MCResult:
    """Estimate a European price by exact terminal GBM sampling.

    Units match bsm_price; numeric contract inputs are scalar. An explicit
    nonnegative integer seed and at least two paths are required. A fresh
    numpy.random.Generator is used on every call, never global random state.
    Return price, SE and 95% CI together in MCResult, never a bare float.

    Sampling the discounted terminal directly is algebraically equivalent to
    simulating S_T then discounting, without an unnecessary large intermediate.
    There is no time discretisation error. Storage and work are O(n_paths);
    this version does not chunk draws. At tau=0 or vol=0 the deterministic
    result is evaluated without draws and retains the requested path count.
    Invalid inputs raise ValueError; nonfinite sample arithmetic raises
    FloatingPointError. Zero sample variation in a stochastic model emits a
    RuntimeWarning and is marked degenerate, not deterministic.

    Antithetic sampling uses N/2 independent pair averages, not N independent
    payoffs. Control variates use discounted S_T, with known mean S*exp(-div*tau),
    and a coefficient fitted on an independent SeedSequence child (spawn_key=1).
    The main stream remains default_rng(seed). Pilot paths are checked and used
    only when control_variate=True; with antithetic sampling both counts must be
    even and at least four. A constant pilot control gives beta=0.

    baseline_se estimates plain-MC uncertainty at the same total main+pilot
    payoff cost, using marginal sample variances (separate strands for pairs).
    variance_reduction is (baseline_se/se)**2, or None if either variance is
    unresolved/zero. It is a measured diagnostic, not a guaranteed improvement.
    """
    _validate_inputs(spot, strike, tau, rate, vol, div, kind, n_paths, seed)
    _validate_reduction(n_paths, antithetic, control_variate, pilot_paths)
    n_paths, seed = int(n_paths), int(seed)
    direction = 1.0 if kind == "call" else -1.0
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        if tau == 0 or vol == 0:
            price = float(
                max(direction * (spot * np.exp(-div * tau) - strike * np.exp(-rate * tau)), 0)
            )
            return MCResult(
                price,
                0.0,
                price,
                price,
                n_paths,
                n_paths // 2 if antithetic else n_paths,
                seed,
                "deterministic",
                antithetic=bool(antithetic),
                control_variate=bool(control_variate),
                baseline_se=0.0,
            )
        samples = _european_samples(
            spot,
            strike,
            tau,
            rate,
            vol,
            div,
            kind,
            n_paths=n_paths,
            rng=np.random.default_rng(seed),
            antithetic=antithetic,
        )
        pilot = None
        if control_variate:
            pilot_rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(1,)))
            pilot = _european_samples(
                spot,
                strike,
                tau,
                rate,
                vol,
                div,
                kind,
                n_paths=int(pilot_paths),
                rng=pilot_rng,
                antithetic=antithetic,
            )
        return _estimate_samples(
            *samples,
            float(spot * np.exp(-div * tau)),
            seed=seed,
            pilot=pilot,
        )
