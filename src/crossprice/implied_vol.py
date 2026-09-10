"""Scalar BSM implied volatility with safeguarded Newton and Brent fallback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.optimize import brentq

from crossprice.analytic import OptionKind, bsm_greeks, bsm_price

type IVStatus = Literal[
    "converged",
    "lower_bound",
    "upper_bound",
    "expired",
    "no_bracket",
    "max_iterations",
    "residual_too_large",
]
type IVMethod = Literal["newton", "brent", "boundary", "bracket"]


@dataclass(frozen=True)
class IVResult:
    """An inversion result, not a claim about the precision of the input quote.

    ``vol`` is ``None`` at expiry, at the upper price bound and when no bracket
    exists; on ``max_iterations`` or ``residual_too_large`` it is the last
    iterate, not a verified root. ``iterations``
    counts Newton evaluations plus Brent iterations, excluding bracket search.
    ``price_residual`` is BSM(vol) - price; boundary residuals use the limiting
    price. For no_bracket it is the residual at max_vol, not at a returned root.
    Always inspect status before consuming vol. No implicit float conversion.
    """

    vol: float | None
    iterations: int
    method: IVMethod
    price_residual: float
    status: IVStatus


def implied_vol(
    spot: float,
    strike: float,
    tau: float,
    rate: float,
    price: float,
    div: float = 0.0,
    kind: OptionKind = "call",
    *,
    initial_vol: float | None = None,
    price_atol: float = 1e-10,
    vol_tol: float = 1e-12,
    max_newton: int = 12,
    max_iterations: int = 128,
    max_vol: float = 16.0,
) -> IVResult:
    """Invert a European price; price replaces vol in the common argument order.

    Inputs are scalar, in the units of bsm_price, including signed rates/yields.
    Nonfinite/invalid inputs and prices outside the exact no-arbitrage bounds
    raise ValueError. At the lower bound return vol=0 with status lower_bound;
    an observed rounded price there cannot distinguish small positive vols.
    At the upper bound no finite vol exists (upper_bound, vol=None). At expiry
    only the payoff is admissible and volatility is unidentified (expired).

    The default guess is the larger of the forward-ATM Brenner-Subrahmanyam
    approximation sqrt(2*pi/tau)*(price-lower)/discounted_spot and the BSM
    inflection volatility sqrt(2*abs(log(S/K)+(r-q)*tau)/tau). The former uses
    time value for either kind; away from ATM it is a guess, not an identity.
    Bracket [0, high] expands geometrically up to the explicit max_vol cap.
    Newton stays inside that bracket and requires both a price residual no
    larger than price_atol (currency units) and abs(residual/vega) <= vol_tol.
    Zero/tiny vega, an escaping step, or max_newton exhaustion switches to
    scipy.optimize.brentq, xtol=vol_tol and rtol=4*float64 epsilon. Set
    max_newton=0 to use Brent alone. The total iteration budget is shared.

    Failure to bracket or converge is a status, never NaN or a clipped quote.
    Even a converged root is poorly identified when quote error divided by
    vega is large. Float64 BSM cancellation/underflow is not repaired here.
    """
    inputs = (spot, strike, tau, rate, price, div)
    if any(np.ndim(value) != 0 for value in inputs):
        raise ValueError("implied volatility requires scalar contract inputs")
    if not np.isfinite(price):
        raise ValueError("price must be finite")
    lower_bound = float(bsm_price(spot, strike, tau, rate, 0.0, div, kind))
    for name, value in (("price_atol", price_atol), ("vol_tol", vol_tol), ("max_vol", max_vol)):
        if np.ndim(value) != 0 or not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    for name, value, minimum in (
        ("max_newton", max_newton, 0),
        ("max_iterations", max_iterations, 1),
    ):
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value < minimum
        ):
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if initial_vol is not None and (
        np.ndim(initial_vol) != 0 or not np.isfinite(initial_vol) or initial_vol <= 0
    ):
        raise ValueError("initial_vol must be finite and positive")
    if tau == 0:
        if price != lower_bound:
            raise ValueError(f"at expiry price must equal payoff {lower_bound}")
        return IVResult(None, 0, "boundary", 0.0, "expired")

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        discounted_spot = float(spot * np.exp(-div * tau))
        discounted_strike = float(strike * np.exp(-rate * tau))
        upper_bound = discounted_spot if kind == "call" else discounted_strike
        if price < lower_bound or price > upper_bound:
            raise ValueError(f"price outside no-arbitrage bounds [{lower_bound}, {upper_bound}]")
        if price == lower_bound:
            return IVResult(0.0, 0, "boundary", 0.0, "lower_bound")
        if price == upper_bound:
            return IVResult(None, 0, "boundary", 0.0, "upper_bound")
        if initial_vol is None:
            log_moneyness = np.log(spot) - np.log(strike) + (rate - div) * tau
            initial_vol = float(
                max(
                    np.sqrt(2 * np.pi / tau) * (price - lower_bound) / discounted_spot,
                    np.sqrt(2 * abs(log_moneyness) / tau),
                )
            )

    def residual(vol: float) -> float:
        return float(bsm_price(spot, strike, tau, rate, vol, div, kind)) - price

    lower, upper = 0.0, min(max_vol, max(0.5, initial_vol))
    upper_residual = residual(upper)
    while upper_residual < 0 and upper < max_vol:
        upper = min(max_vol, 2 * upper)
        upper_residual = residual(upper)
    if upper_residual < 0:
        return IVResult(None, 0, "bracket", upper_residual, "no_bracket")
    if upper_residual == 0:
        return IVResult(float(upper), 0, "bracket", 0.0, "converged")

    vol = initial_vol if 0 < initial_vol < upper else upper / 2
    iterations = 0
    for _ in range(min(max_newton, max_iterations)):
        iterations += 1
        error = residual(vol)
        vega = float(bsm_greeks(spot, strike, tau, rate, vol, div, kind).vega)
        if vega <= np.finfo(float).tiny:
            break
        with np.errstate(over="ignore", divide="ignore"):
            step = float(np.divide(error, vega))
        if abs(error) <= price_atol and abs(step) <= vol_tol:
            return IVResult(float(vol), iterations, "newton", error, "converged")
        if error < 0:
            lower = vol
        else:
            upper = vol
        candidate = vol - step
        if not np.isfinite(candidate) or not lower < candidate < upper:
            break
        if iterations < min(max_newton, max_iterations):
            vol = candidate

    if iterations == max_iterations:
        return IVResult(float(vol), iterations, "newton", residual(vol), "max_iterations")
    root, report = brentq(
        residual,
        lower,
        upper,
        xtol=vol_tol,
        rtol=4 * np.finfo(float).eps,
        maxiter=max_iterations - iterations,
        full_output=True,
        disp=False,
    )
    error = residual(root)
    status: IVStatus = "converged"
    if not report.converged:
        status = "max_iterations"
    elif abs(error) > price_atol:
        status = "residual_too_large"
    return IVResult(float(root), iterations + report.iterations, "brent", error, status)
