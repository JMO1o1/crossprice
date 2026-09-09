"""Black-Scholes-Merton prices and Greeks with continuous dividends."""

from __future__ import annotations

from typing import Literal, NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import ndtr

type OptionKind = Literal["call", "put"]
type FloatArray = NDArray[np.float64]
type FloatResult = float | FloatArray


class Greeks(NamedTuple):
    """Price derivatives; theta is per calendar year, vega/rho per unit change.

    Vega for one volatility percentage point is ``vega / 100``; likewise rho
    for one rate percentage point. Each field follows the input broadcast shape.
    """

    delta: FloatResult
    gamma: FloatResult
    vega: FloatResult
    theta: FloatResult
    rho: FloatResult


def _broadcast_inputs(
    spot: ArrayLike,
    strike: ArrayLike,
    tau: ArrayLike,
    rate: ArrayLike,
    vol: ArrayLike,
    div: ArrayLike,
    kind: OptionKind,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    if kind not in ("call", "put"):
        raise ValueError("kind must be 'call' or 'put'")
    spot, strike, tau, rate, vol, div = np.broadcast_arrays(
        *(np.asarray(value, dtype=np.float64) for value in (spot, strike, tau, rate, vol, div))
    )
    for name, value in zip(
        ("spot", "strike", "tau", "rate", "vol", "div"),
        (spot, strike, tau, rate, vol, div),
        strict=True,
    ):
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{name} must be finite")
    if np.any(spot <= 0) or np.any(strike <= 0):
        raise ValueError("spot and strike must be positive")
    if np.any(tau < 0) or np.any(vol < 0):
        raise ValueError("tau and vol must be non-negative")
    return spot, strike, tau, rate, vol, div


def _scalar_or_array(value: FloatArray) -> FloatResult:
    return float(value) if value.ndim == 0 else value


def _normal_arguments(
    spot: FloatArray,
    strike: FloatArray,
    tau: FloatArray,
    rate: FloatArray,
    div: FloatArray,
    stddev: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    log_forward_moneyness = np.log(spot) - np.log(strike) + (rate - div) * tau
    with np.errstate(over="ignore"):
        d1 = log_forward_moneyness / stddev + 0.5 * stddev
    return d1, d1 - stddev


def bsm_price(
    spot: ArrayLike,
    strike: ArrayLike,
    tau: ArrayLike,
    rate: ArrayLike,
    vol: ArrayLike,
    div: ArrayLike = 0.0,
    kind: OptionKind = "call",
) -> FloatResult:
    """Return a European call or put price in the spot/strike currency units.

    All six numeric inputs broadcast together as float64. Scalar inputs return
    a float; otherwise the result has the broadcast shape. Time is in years;
    rate, volatility and continuous dividend yield are annual decimal inputs.
    Spot and strike must be positive, tau and vol non-negative, and all inputs
    finite. Invalid inputs (including kind or incompatible shapes) raise
    ValueError. Negative rates and dividend yields are supported.

    At tau=0 return intrinsic value; at vol=0 return the discounted deterministic
    payoff. These branches do not evaluate d1 or d2. Extremely large finite
    inputs can exceed float64's range and raise FloatingPointError.
    """
    spot, strike, tau, rate, vol, div = _broadcast_inputs(spot, strike, tau, rate, vol, div, kind)
    direction = 1.0 if kind == "call" else -1.0
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        discounted_spot = spot * np.exp(-div * tau)
        discounted_strike = strike * np.exp(-rate * tau)
        price = np.asarray(np.maximum(direction * (discounted_spot - discounted_strike), 0.0))
        stddev = vol * np.sqrt(tau)
        regular = stddev > 0
        d1, d2 = _normal_arguments(
            spot[regular],
            strike[regular],
            tau[regular],
            rate[regular],
            div[regular],
            stddev[regular],
        )
        price[regular] = direction * (
            discounted_spot[regular] * ndtr(direction * d1)
            - discounted_strike[regular] * ndtr(direction * d2)
        )
    return _scalar_or_array(price)


def bsm_greeks(
    spot: ArrayLike,
    strike: ArrayLike,
    tau: ArrayLike,
    rate: ArrayLike,
    vol: ArrayLike,
    div: ArrayLike = 0.0,
    kind: OptionKind = "call",
) -> Greeks:
    """Return delta, gamma, vega, calendar-time theta and rho.

    Inputs and validation follow bsm_price. Theta is -dV/dtau, not dV/dtau.
    Boundary conventions are explicit: at zero volatility and positive time,
    use positive-volatility limits. At the deterministic forward-strike kink,
    delta/rho/theta use the half-weight limit, gamma is +inf and vega is the
    nonzero right derivative with respect to vol. These are smoothing limits,
    not claims that the kink has ordinary two-sided derivatives.

    At expiry, delta is the payoff slope (midpoint at the strike), gamma is
    +inf at the strike and zero elsewhere, and vega/rho are zero. Theta is the
    limit from positive time: -inf at the strike for positive vol. At tau=vol=0
    and spot=strike, use the deterministic right-time derivative for theta.
    No NaN is returned for these boundaries; the joint limit can be path-dependent.
    """
    spot, strike, tau, rate, vol, div = _broadcast_inputs(spot, strike, tau, rate, vol, div, kind)
    direction = 1.0 if kind == "call" else -1.0
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        root_tau = np.sqrt(tau)
        stddev = vol * root_tau
        dividend_discount = np.exp(-div * tau)
        discounted_spot = spot * dividend_discount
        discounted_strike = strike * np.exp(-rate * tau)
        exercise_weight = np.heaviside(direction * (discounted_spot - discounted_strike), 0.5)
        delta = np.asarray(direction * dividend_discount * exercise_weight)
        gamma = np.zeros_like(spot)
        vega = np.zeros_like(spot)
        theta = np.asarray(
            direction * (div * discounted_spot - rate * discounted_strike) * exercise_weight
        )
        rho = np.asarray(direction * tau * discounted_strike * exercise_weight)

        kink = (stddev == 0) & (discounted_spot == discounted_strike)
        gamma[kink] = np.inf
        vega[kink] = discounted_spot[kink] * root_tau[kink] / np.sqrt(2 * np.pi)
        expired_at_strike = (tau == 0) & (spot == strike)
        theta[expired_at_strike & (vol > 0)] = -np.inf
        deterministic_corner = expired_at_strike & (vol == 0)
        theta[deterministic_corner] = -np.maximum(
            direction
            * (rate[deterministic_corner] - div[deterministic_corner])
            * spot[deterministic_corner],
            0.0,
        )

        regular = stddev > 0
        d1, d2 = _normal_arguments(
            spot[regular],
            strike[regular],
            tau[regular],
            rate[regular],
            div[regular],
            stddev[regular],
        )
        with np.errstate(over="ignore", under="ignore"):
            density = np.exp(-0.5 * np.square(d1)) / np.sqrt(2 * np.pi)
        cdf1, cdf2 = ndtr(direction * d1), ndtr(direction * d2)
        delta[regular] = direction * dividend_discount[regular] * cdf1
        gamma[regular] = dividend_discount[regular] * density / spot[regular] / stddev[regular]
        vega[regular] = discounted_spot[regular] * density * root_tau[regular]
        theta[regular] = -discounted_spot[regular] * density * vol[regular] / (
            2 * root_tau[regular]
        ) + direction * (
            div[regular] * discounted_spot[regular] * cdf1
            - rate[regular] * discounted_strike[regular] * cdf2
        )
        rho[regular] = direction * tau[regular] * discounted_strike[regular] * cdf2
    return Greeks(*(_scalar_or_array(value) for value in (delta, gamma, vega, theta, rho)))
