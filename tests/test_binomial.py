"""Independent recursion, invariants and measured CRR convergence checks."""

from __future__ import annotations

from math import comb, exp, sqrt
from timeit import repeat

import numpy as np
import pytest
from numpy.testing import assert_allclose

from crossprice.analytic import OptionKind, bsm_price
from crossprice.binomial import crr_price


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("steps", [1, 2, 3])
def test_small_tree_matches_terminal_distribution(kind: OptionKind, steps: int) -> None:
    spot, strike, tau, rate, vol, div = 100, 105, 1, 0.05, 0.3, 0.02
    up = exp(vol * sqrt(tau / steps))
    down = 1 / up
    probability = (exp((rate - div) * tau / steps) - down) / (up - down)
    direction = 1 if kind == "call" else -1
    expected = exp(-rate * tau) * sum(
        comb(steps, ups)
        * probability**ups
        * (1 - probability) ** (steps - ups)
        * max(direction * (spot * up**ups * down ** (steps - ups) - strike), 0)
        for ups in range(steps + 1)
    )
    assert crr_price(spot, strike, tau, rate, vol, div, kind, steps=steps) == pytest.approx(
        expected, rel=0, abs=2e-13
    )


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("offset", [0, 1], ids=["even", "odd"])
def test_atm_convergence_order_and_oscillation(kind: OptionKind, offset: int) -> None:
    counts = np.array([50, 100, 200, 400, 800]) + offset
    reference = bsm_price(100, 100, 1, 0.05, 0.2, kind=kind)
    prices = np.array(
        [crr_price(100, 100, 1, 0.05, 0.2, kind=kind, steps=count) for count in counts]
    )
    errors = prices - reference
    order = -np.polyfit(np.log(counts), np.log(np.abs(errors)), 1)[0]
    assert 0.9 < order < 1.1
    assert np.all(errors < 0) if offset == 0 else np.all(errors > 0)
    assert abs(errors[-1]) < 0.005
    print(
        f"{kind} offset={offset}: order={order:.6f}; N={counts.tolist()}; errors={errors.tolist()}"
    )


def test_off_strike_does_not_have_fixed_even_odd_bias() -> None:
    counts = np.array([50, 51, 100, 101, 200, 201])
    reference = bsm_price(100, 110, 1, 0.05, 0.2, 0.02)
    errors = np.array(
        [crr_price(100, 110, 1, 0.05, 0.2, 0.02, steps=count) - reference for count in counts]
    )
    assert np.any(errors[counts % 2 == 0] > 0)
    print(f"off-strike: N={counts.tolist()}; errors={errors.tolist()}")


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize(
    "inputs",
    [
        (100, 100, 1, 0.05, 0.2, 0),
        (100, 90, 0.5, 0.04, 0.3, 0.02),
        (100, 110, 2, -0.01, 0.5, 0.04),
        (42, 40, 0.5, 0.1, 0.2, 0),
        (100, 100, 0.01, 0.03, 0.05, 0.01),
        (100, 150, 1, 0.01, 0.8, 0.03),
    ],
)
def test_european_converges_on_parameter_grid(inputs: tuple[float, ...], kind: OptionKind) -> None:
    assert_allclose(
        crr_price(*inputs, kind=kind, steps=1000),
        bsm_price(*inputs, kind=kind),
        rtol=2e-3,
        atol=5e-3,
    )


def test_seeded_tree_parity_and_bounds() -> None:
    rng = np.random.default_rng(20260910)
    cases = np.column_stack(
        [
            rng.uniform(lower, upper, 64)
            for lower, upper in (
                (50, 150),
                (50, 150),
                (0.02, 2),
                (-0.02, 0.10),
                (0.05, 0.5),
                (0, 0.08),
            )
        ]
    )
    residuals = []
    for spot, strike, tau, rate, vol, div in cases:
        call = crr_price(spot, strike, tau, rate, vol, div, "call")
        put = crr_price(spot, strike, tau, rate, vol, div, "put")
        discounted_spot, discounted_strike = spot * exp(-div * tau), strike * exp(-rate * tau)
        expected = discounted_spot - discounted_strike
        assert_allclose(call - put, expected, rtol=1e-11, atol=1e-10)
        assert max(expected, 0) - 1e-10 <= call <= discounted_spot + 1e-10
        assert max(-expected, 0) - 1e-10 <= put <= discounted_strike + 1e-10
        residuals.append(abs(call - put - expected))
    print(f"tree parity: 64 sets, max residual={max(residuals):.3e}")


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("tau,vol", [(0, 0), (0, 0.2), (2, 0)])
def test_tree_deterministic_limits(tau: float, vol: float, kind: OptionKind) -> None:
    direction = 1 if kind == "call" else -1
    for spot in (80, 100, 120):
        expected = max(direction * (spot * exp(-0.02 * tau) - 100 * exp(-0.05 * tau)), 0)
        assert_allclose(
            crr_price(spot, 100, tau, 0.05, vol, 0.02, kind), expected, rtol=0, atol=1e-13
        )


@pytest.mark.parametrize("rate,vol", [(0.1, 0.01), (0.2, 0.2), (-0.2, 0.2)])
def test_invalid_probability_is_not_clipped(rate: float, vol: float) -> None:
    with pytest.raises(ValueError, match="probability"):
        crr_price(100, 100, 1, rate, vol, steps=1)
    assert np.isfinite(crr_price(100, 100, 1, rate, vol, steps=256))


@pytest.mark.parametrize("steps", [0, -1, 1.5, True])
def test_invalid_step_count(steps: int) -> None:
    with pytest.raises(ValueError, match="steps"):
        crr_price(100, 100, 1, 0.05, 0.2, steps=steps)


@pytest.mark.parametrize(
    "index,value", [(0, 0), (1, 0), (2, -1), (4, -1)] + [(index, np.nan) for index in range(6)]
)
def test_invalid_contract_inputs(index: int, value: float) -> None:
    inputs = [100, 100, 1, 0.05, 0.2, 0.02]
    inputs[index] = value
    with pytest.raises(ValueError):
        crr_price(*inputs)


def _scalar_crr(steps: int) -> float:
    dt = 1 / steps
    up = exp(0.2 * sqrt(dt))
    probability = (exp(0.05 * dt) - 1 / up) / (up - 1 / up)
    discount = exp(-0.05 * dt)
    values = [max(100 * up ** (2 * ups - steps) - 100, 0) for ups in range(steps + 1)]
    for remaining in range(steps, 0, -1):
        values = [
            discount * ((1 - probability) * values[node] + probability * values[node + 1])
            for node in range(remaining)
        ]
    return values[0]


def test_vectorized_recursion_matches_scalar_reference() -> None:
    steps = 400
    assert_allclose(
        crr_price(100, 100, 1, 0.05, 0.2, steps=steps), _scalar_crr(steps), rtol=0, atol=5e-11
    )
    scalar_seconds = float(np.median(repeat(lambda: _scalar_crr(steps), number=1, repeat=3)))
    vector_seconds = float(
        np.median(
            repeat(lambda: crr_price(100, 100, 1, 0.05, 0.2, steps=steps), number=1, repeat=3)
        )
    )
    print(
        f"N=400 timing (median of 3): scalar={scalar_seconds:.6f}s, "
        f"vector={vector_seconds:.6f}s, ratio={scalar_seconds / vector_seconds:.2f}"
    )
