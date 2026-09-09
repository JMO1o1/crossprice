"""Seeded risk-neutral Monte Carlo estimates with uncertainty on every price."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
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
    """

    price: float
    se: float
    ci_low: float
    ci_high: float
    n_paths: int
    n_samples: int
    seed: int
    ci_status: CIStatus

    @property
    def ci(self) -> tuple[float, float]:
        """Return the reported 95% confidence interval endpoints."""
        return self.ci_low, self.ci_high


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
    """
    _validate_inputs(spot, strike, tau, rate, vol, div, kind, n_paths, seed)
    n_paths, seed = int(n_paths), int(seed)
    direction = 1.0 if kind == "call" else -1.0
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        if tau == 0 or vol == 0:
            price = float(
                max(direction * (spot * np.exp(-div * tau) - strike * np.exp(-rate * tau)), 0)
            )
            return MCResult(price, 0.0, price, price, n_paths, n_paths, seed, "deterministic")
        rng = np.random.default_rng(seed)
        discounted_terminal = spot * np.exp(
            (-div - 0.5 * vol**2) * tau + vol * np.sqrt(tau) * rng.standard_normal(n_paths)
        )
        observations = np.maximum(
            direction * (discounted_terminal - strike * np.exp(-rate * tau)), 0.0
        )
    return _summarize(observations, n_paths=n_paths, seed=seed)
