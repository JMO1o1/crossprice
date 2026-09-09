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


def _scalar_crr(steps: int, kind: OptionKind = "call", american: bool = False) -> float:
    dt = 1 / steps
    up = exp(0.2 * sqrt(dt))
    probability = (exp(0.05 * dt) - 1 / up) / (up - 1 / up)
    discount = exp(-0.05 * dt)
    direction = 1 if kind == "call" else -1
    values = [max(direction * (100 * up ** (2 * ups - steps) - 100), 0) for ups in range(steps + 1)]
    for remaining in range(steps, 0, -1):
        values = [
            discount * ((1 - probability) * values[node] + probability * values[node + 1])
            for node in range(remaining)
        ]
        if american:
            values = [
                max(value, direction * (100 * up ** (2 * node - remaining + 1) - 100))
                for node, value in enumerate(values)
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


@pytest.mark.parametrize("steps", [1, 2, 5])
@pytest.mark.parametrize("kind", ["call", "put"])
def test_american_matches_scalar_recursion(steps: int, kind: OptionKind) -> None:
    actual = crr_price(100, 100, 1, 0.05, 0.2, kind=kind, steps=steps, american=True)
    assert_allclose(actual, _scalar_crr(steps, kind, True), rtol=0, atol=2e-13)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_american_dominates_european_and_intrinsic(kind: OptionKind) -> None:
    rng = np.random.default_rng(20260911)
    cases = np.column_stack(
        [
            rng.uniform(lower, upper, 64)
            for lower, upper in (
                (50, 150),
                (50, 150),
                (0.02, 3),
                (-0.05, 0.10),
                (0.1, 0.5),
                (-0.03, 0.10),
            )
        ]
    )
    direction = 1 if kind == "call" else -1
    for spot, strike, tau, rate, vol, div in cases:
        american = crr_price(spot, strike, tau, rate, vol, div, kind, steps=128, american=True)
        european = crr_price(spot, strike, tau, rate, vol, div, kind, steps=128)
        assert american >= european - 1e-12
        assert american >= max(direction * (spot - strike), 0) - 1e-12
        upper = (
            spot * exp(max(-div, 0) * tau) if kind == "call" else strike * exp(max(-rate, 0) * tau)
        )
        assert american <= upper + 1e-10


@pytest.mark.parametrize("spot", [70, 100, 130])
@pytest.mark.parametrize("tau", [0.1, 1, 3])
@pytest.mark.parametrize("rate", [0, 0.05, 0.15])
def test_nondividend_american_call_equals_european(spot: float, tau: float, rate: float) -> None:
    american = crr_price(spot, 100, tau, rate, 0.2, american=True, steps=300)
    european = crr_price(spot, 100, tau, rate, 0.2, steps=300)
    assert_allclose(american, european, rtol=1e-12, atol=2e-11)


def test_nondividend_call_equality_requires_nonnegative_rates() -> None:
    american = crr_price(120, 100, 1, -0.05, 0.2, steps=800, american=True)
    european = crr_price(120, 100, 1, -0.05, 0.2, steps=800)
    assert american - european > 0.1
    print(f"negative-rate call: American={american:.12f}, European={european:.12f}")


def test_dividend_call_can_have_early_exercise_premium() -> None:
    american = crr_price(120, 100, 1, 0.03, 0.2, 0.1, steps=800, american=True)
    european = crr_price(120, 100, 1, 0.03, 0.2, 0.1, steps=800)
    assert american - european > 0.1


def test_put_premium_trends_on_benchmark_grid() -> None:
    strikes = [90, 100, 110, 120]
    maturities = [0.25, 0.5, 1, 2]
    premiums = np.array(
        [
            [
                crr_price(100, strike, tau, 0.05, 0.2, kind="put", steps=800, american=True)
                - crr_price(100, strike, tau, 0.05, 0.2, kind="put", steps=800)
                for strike in strikes
            ]
            for tau in maturities
        ]
    )
    assert np.all(premiums >= -1e-12)
    assert np.all(np.diff(premiums, axis=0) >= -1e-10)
    assert np.all(np.diff(premiums, axis=1) >= -1e-10)
    print(f"put premiums, rows T={maturities}, columns K={strikes}:\n{premiums}")


def test_american_put_convergence_to_fine_tree() -> None:
    counts = np.array([100, 200, 400, 800])
    reference = crr_price(100, 100, 1, 0.05, 0.2, kind="put", steps=6400, american=True)
    prices = np.array(
        [
            crr_price(100, 100, 1, 0.05, 0.2, kind="put", steps=count, american=True)
            for count in counts
        ]
    )
    errors = np.abs(prices - reference)
    order = -np.polyfit(np.log(counts), np.log(errors), 1)[0]
    assert 0.7 < order < 1.5
    assert errors[-1] < 0.002
    assert np.all(np.diff(errors) < 0)
    print(
        f"American put: fine(6400)={reference:.12f}; N={counts.tolist()}; "
        f"prices={prices.tolist()}; observed order={order:.6f}"
    )


@pytest.mark.parametrize("kind", ["call", "put"])
def test_american_deterministic_exercise_dates(kind: OptionKind) -> None:
    spot, strike, tau, rate, div, steps = 100, 100, 30, 0.1, 0.05, 300
    direction = 1 if kind == "call" else -1
    expected = max(
        max(
            direction
            * (spot * exp(-div * tau * date / steps) - strike * exp(-rate * tau * date / steps)),
            0,
        )
        for date in range(steps + 1)
    )
    actual = crr_price(spot, strike, tau, rate, 0, div, kind, steps=steps, american=True)
    assert_allclose(actual, expected, rtol=0, atol=2e-13)
    if kind == "call":
        assert actual > max(spot - strike, crr_price(spot, strike, tau, rate, 0, div)) + 1
        print(f"deterministic interior optimum: price={actual:.12f}")


@pytest.mark.parametrize("kind", ["call", "put"])
def test_american_expiry(kind: OptionKind) -> None:
    direction = 1 if kind == "call" else -1
    for spot in (80, 100, 120):
        assert crr_price(spot, 100, 0, 0.05, 0.2, kind=kind, american=True) == max(
            direction * (spot - 100), 0
        )


def test_american_flag_is_boolean() -> None:
    with pytest.raises(ValueError, match="american"):
        crr_price(100, 100, 1, 0.05, 0.2, american="false")
