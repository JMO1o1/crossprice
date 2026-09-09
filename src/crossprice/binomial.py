"""Cox-Ross-Rubinstein pricing with O(steps) storage and vectorised layers."""

from __future__ import annotations

import numpy as np

from crossprice.analytic import OptionKind


def crr_price(
    spot: float,
    strike: float,
    tau: float,
    rate: float,
    vol: float,
    div: float = 0.0,
    kind: OptionKind = "call",
    *,
    steps: int = 200,
) -> float:
    """Return a European CRR price for scalar contract inputs.

    Parameter units and kind follow bsm_price. This method independently prices
    the lattice; it never calls the analytic price. Each layer is vectorised,
    requiring O(steps) memory and O(steps**2) work. Inputs must be finite, spot
    and strike positive, tau and vol non-negative, and steps a positive integer
    (not a bool). Array contracts are not supported.

    At expiry return the payoff; at zero volatility use deterministic discounted
    value. Otherwise the CRR probability must lie strictly between zero and one.
    Invalid inputs or probabilities raise ValueError; increase steps when a
    coarse grid cannot represent the drift. Probabilities are never clipped.
    Intermediates outside float64's range raise FloatingPointError.
    """
    inputs = (spot, strike, tau, rate, vol, div)
    if any(np.ndim(value) != 0 for value in inputs):
        raise ValueError("CRR requires scalar contract inputs")
    if not np.all(np.isfinite(inputs)):
        raise ValueError("contract inputs must be finite")
    if spot <= 0 or strike <= 0 or tau < 0 or vol < 0:
        raise ValueError("spot/strike must be positive and tau/vol non-negative")
    if kind not in ("call", "put"):
        raise ValueError("kind must be 'call' or 'put'")
    if isinstance(steps, (bool, np.bool_)) or not isinstance(steps, (int, np.integer)) or steps < 1:
        raise ValueError("steps must be a positive integer")
    direction = 1.0 if kind == "call" else -1.0
    if tau == 0:
        return max(direction * (spot - strike), 0.0)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        if vol == 0:
            return float(
                max(direction * (spot * np.exp(-div * tau) - strike * np.exp(-rate * tau)), 0)
            )
        dt = tau / steps
        jump = vol * np.sqrt(dt)
        carry = (rate - div) * dt
        if abs(carry) >= jump:
            raise ValueError(
                "CRR probability requires abs(rate-div)*dt < vol*sqrt(dt); increase steps"
            )
        probability = np.expm1(carry + jump) / np.expm1(2 * jump)
        if not 0 < probability < 1:
            raise ValueError("CRR probability is not strictly between 0 and 1; increase steps")
        discount = np.exp(-rate * dt)
        terminal = spot * np.exp(jump * (2 * np.arange(steps + 1) - steps))
        values = np.maximum(direction * (terminal - strike), 0.0)
        for _remaining in range(steps, 0, -1):
            values = discount * ((1 - probability) * values[:-1] + probability * values[1:])
    return float(values[0])
