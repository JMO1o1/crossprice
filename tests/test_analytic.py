"""Analytic benchmarks and invariants, independent of later numerical methods.

Reference data: J. C. Hull, Options, Futures, and Other Derivatives, the worked
S=42, K=40 example; E. G. Haug, Option Pricing Formulas (McGraw-Hill, 1998),
pp. 2-8, as transcribed in QuantLib's EuropeanOptionTests/testValues:
https://github.com/lballabio/QuantLib/blob/master/test-suite/europeanoption.cpp
Only rounded published values are used, with half-last-digit absolute tolerances.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import numpy as np
import pytest
from numpy.testing import assert_allclose

from crossprice.analytic import FloatArray, FloatResult, Greeks, OptionKind, bsm_greeks, bsm_price


@pytest.mark.parametrize(
    "spot,strike,tau,rate,vol,div,kind,expected,tolerance",
    [
        pytest.param(42, 40, 0.5, 0.10, 0.20, 0, "call", 4.76, 0.005, id="hull-call"),
        pytest.param(42, 40, 0.5, 0.10, 0.20, 0, "put", 0.81, 0.005, id="hull-put"),
        pytest.param(60, 65, 0.25, 0.08, 0.30, 0, "call", 2.1334, 0.00005, id="haug-call"),
        pytest.param(100, 95, 0.5, 0.10, 0.20, 0.05, "put", 2.4648, 0.00005, id="haug-div-put"),
    ],
)
def test_published_prices(
    spot: float,
    strike: float,
    tau: float,
    rate: float,
    vol: float,
    div: float,
    kind: OptionKind,
    expected: float,
    tolerance: float,
) -> None:
    price = bsm_price(spot, strike, tau, rate, vol, div, kind)
    assert isinstance(price, float)
    assert price == pytest.approx(expected, rel=0, abs=tolerance)
    print(f"{kind}: price={price:.12f}, published={expected}, atol={tolerance}")


@pytest.fixture
def random_inputs() -> tuple[FloatArray, ...]:
    rng = np.random.default_rng(20260909)
    return tuple(
        rng.uniform(lower, upper, 256)
        for lower, upper in (
            (20, 200),
            (20, 200),
            (0.001, 5),
            (-0.05, 0.15),
            (0.02, 1),
            (-0.02, 0.12),
        )
    )


def test_seeded_put_call_parity(random_inputs: tuple[FloatArray, ...]) -> None:
    spot, strike, tau, rate, _vol, div = random_inputs
    call = bsm_price(*random_inputs, kind="call")
    put = bsm_price(*random_inputs, kind="put")
    expected = spot * np.exp(-div * tau) - strike * np.exp(-rate * tau)
    assert_allclose(call - put, expected, rtol=5e-13, atol=5e-12)
    print(f"parity: 256 sets, max residual={np.max(np.abs(call - put - expected)):.3e}")


@pytest.mark.parametrize("kind", ["call", "put"])
def test_price_bounds(random_inputs: tuple[FloatArray, ...], kind: OptionKind) -> None:
    spot, strike, tau, rate, _vol, div = random_inputs
    discounted_spot = spot * np.exp(-div * tau)
    discounted_strike = strike * np.exp(-rate * tau)
    direction = 1 if kind == "call" else -1
    lower = np.maximum(direction * (discounted_spot - discounted_strike), 0)
    upper = discounted_spot if kind == "call" else discounted_strike
    price = bsm_price(*random_inputs, kind=kind)
    assert np.all(price >= lower - 5e-12)
    assert np.all(price <= upper + 5e-12)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_broadcast_mixed_limits_without_invalid_arithmetic(kind: OptionKind) -> None:
    spot = np.array([80.0, 100.0, 120.0])[:, None]
    tau = np.array([0.0, 0.5, 1.0, 2.0])
    vol = np.array([0.2, 0.2, 0.0, 0.3])
    with np.errstate(divide="raise", invalid="raise", over="raise"):
        prices = bsm_price(spot, 100, tau, 0.05, vol, 0.02, kind)
    assert isinstance(prices, np.ndarray)
    assert prices.shape == (3, 4)
    assert np.all(np.isfinite(prices))
    for row, current_spot in enumerate(spot[:, 0]):
        for column in range(4):
            expected = bsm_price(current_spot, 100, tau[column], 0.05, vol[column], 0.02, kind)
            assert prices[row, column] == expected


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("tau,vol", [(0, 0), (0, 0.2), (1.5, 0), (1.5, 1e-10)])
def test_deterministic_limits(tau: float, vol: float, kind: OptionKind) -> None:
    spot = np.array([80.0, 100.0, 120.0])
    direction = 1 if kind == "call" else -1
    expected = np.maximum(direction * (spot * np.exp(-0.02 * tau) - 100 * np.exp(-0.05 * tau)), 0)
    assert_allclose(bsm_price(spot, 100, tau, 0.05, vol, 0.02, kind), expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_small_time_at_the_money(kind: OptionKind) -> None:
    tau = 1e-12
    expected = 100 * 0.2 * np.sqrt(tau / (2 * np.pi))
    assert_allclose(bsm_price(100, 100, tau, 0, 0.2, kind=kind), expected, rtol=1e-8, atol=1e-14)


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("greek", ["delta", "gamma", "vega", "theta", "rho"])
def test_greeks_against_central_price_differences(kind: OptionKind, greek: str) -> None:
    inputs = np.array(
        [
            [100, 100, 1, 0.05, 0.2, 0.02],
            [75, 100, 0.25, -0.02, 0.35, 0.01],
            [125, 90, 2, 0.10, 0.55, 0.07],
            [100, 105, 0.05, 0.03, 0.2, 0.08],
        ]
    ).T
    parameter = {"delta": 0, "gamma": 0, "theta": 2, "rho": 3, "vega": 4}[greek]
    step = (1e-4 if greek == "gamma" else 1e-5) * np.maximum(1, np.abs(inputs[parameter]))
    plus, minus = inputs.copy(), inputs.copy()
    plus[parameter] += step
    minus[parameter] -= step
    value_plus = bsm_price(*plus, kind=kind)
    value_minus = bsm_price(*minus, kind=kind)
    if greek == "gamma":
        expected = (value_plus - 2 * bsm_price(*inputs, kind=kind) + value_minus) / step**2
    else:
        expected = (value_plus - value_minus) / (2 * step)
        if greek == "theta":
            expected = -expected
    actual = getattr(bsm_greeks(*inputs, kind=kind), greek)
    assert_allclose(actual, expected, rtol=2e-5, atol=2e-8)
    print(f"{kind} {greek}: max FD residual={np.max(np.abs(actual - expected)):.3e}")


def test_greek_parity(random_inputs: tuple[FloatArray, ...]) -> None:
    spot, strike, tau, rate, _vol, div = random_inputs
    call = bsm_greeks(*random_inputs, kind="call")
    put = bsm_greeks(*random_inputs, kind="put")
    expected = (
        np.exp(-div * tau),
        np.zeros_like(spot),
        np.zeros_like(spot),
        div * spot * np.exp(-div * tau) - rate * strike * np.exp(-rate * tau),
        tau * strike * np.exp(-rate * tau),
    )
    for call_greek, put_greek, difference in zip(call, put, expected, strict=True):
        assert_allclose(call_greek - put_greek, difference, rtol=5e-13, atol=5e-12)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_greeks_broadcast_with_mixed_boundaries(kind: OptionKind) -> None:
    spot = np.array([80.0, 100.0, 120.0])[:, None]
    tau = np.array([0, 1, 1, 0])
    vol = np.array([0.2, 0, 0.2, 0])
    with np.errstate(divide="raise", invalid="raise", over="raise"):
        greeks = bsm_greeks(spot, 100, tau, 0, vol, 0, kind)
    for field in greeks:
        assert isinstance(field, np.ndarray)
        assert field.shape == (3, 4)
        assert not np.any(np.isnan(field))
    for row, current_spot in enumerate(spot[:, 0]):
        for column in range(4):
            scalar = bsm_greeks(current_spot, 100, tau[column], 0, vol[column], 0, kind)
            assert all(isinstance(value, float) for value in scalar)
            assert_allclose([field[row, column] for field in greeks], scalar, rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_zero_volatility_greeks_at_forward_strike(kind: OptionKind) -> None:
    direction = 1 if kind == "call" else -1
    discount = np.exp(-0.05)
    greeks = bsm_greeks(100, 100, 1, 0.05, 0, 0.05, kind)
    assert greeks.delta == pytest.approx(direction * 0.5 * discount)
    assert greeks.gamma == np.inf
    assert greeks.vega == pytest.approx(100 * discount / np.sqrt(2 * np.pi))
    assert greeks.theta == 0
    assert greeks.rho == pytest.approx(direction * 50 * discount)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_greek_limits_away_from_kinks(kind: OptionKind) -> None:
    for spot in (80, 120):
        deterministic = bsm_greeks(spot, 100, 1, 0.05, 0, 0.02, kind)
        nearly_deterministic = bsm_greeks(spot, 100, 1, 0.05, 1e-10, 0.02, kind)
        assert_allclose(nearly_deterministic, deterministic, rtol=0, atol=1e-12)
        expiry = bsm_greeks(spot, 100, 0, 0.05, 0.2, 0.02, kind)
        nearly_expired = bsm_greeks(spot, 100, 1e-12, 0.05, 0.2, 0.02, kind)
        assert_allclose(nearly_expired, expiry, rtol=0, atol=2e-10)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_expiry_kink_and_near_expiry_divergence(kind: OptionKind) -> None:
    direction = 1 if kind == "call" else -1
    greeks = bsm_greeks(100, 100, 0, 0, 0.2, kind=kind)
    assert greeks == Greeks(direction * 0.5, np.inf, 0, -np.inf, 0)
    tau = 1e-12
    near = bsm_greeks(100, 100, tau, 0, 0.2, kind=kind)
    assert near.gamma == pytest.approx(1 / (100 * 0.2 * np.sqrt(2 * np.pi * tau)), rel=1e-12)
    assert near.theta == pytest.approx(-100 * 0.2 / (2 * np.sqrt(2 * np.pi * tau)), rel=1e-12)


@pytest.mark.parametrize("rate,div", [(0.05, 0.02), (-0.02, 0.05)])
@pytest.mark.parametrize("kind", ["call", "put"])
def test_joint_zero_corner_uses_right_time_derivative(
    rate: float, div: float, kind: OptionKind
) -> None:
    direction = 1 if kind == "call" else -1
    expected_theta = -max(direction * (rate - div) * 100, 0)
    greeks = bsm_greeks(100, 100, 0, rate, 0, div, kind)
    assert greeks.theta == pytest.approx(expected_theta)
    assert greeks.vega == greeks.rho == 0


@pytest.mark.parametrize("spot,strike,kind", [(100, 600, "call"), (600, 100, "put")])
def test_deep_otm_prices_are_not_lost_to_parity(
    spot: float, strike: float, kind: OptionKind
) -> None:
    price = bsm_price(spot, strike, 1, 0, 0.2, kind=kind)
    assert 0 < price < 1e-12
    assert np.all(np.isfinite(bsm_greeks(spot, strike, 1, 0, 0.2, kind=kind)))


@pytest.mark.parametrize("kind", ["call", "put"])
def test_extremely_small_positive_volatility(kind: OptionKind) -> None:
    with np.errstate(divide="raise", invalid="raise", over="raise"):
        price = bsm_price([80, 100, 120], 100, 1, 0, 1e-300, kind=kind)
        greeks = bsm_greeks([80, 100, 120], 100, 1, 0, 1e-300, kind=kind)
    assert np.all(np.isfinite(price))
    assert np.all(np.isfinite(greeks))


@pytest.mark.parametrize("function", [bsm_price, bsm_greeks])
@pytest.mark.parametrize(
    "index,value",
    [(0, 0), (1, 0), (0, -1), (1, -1), (2, -1), (4, -1)]
    + [(index, value) for index in range(6) for value in (np.nan, np.inf, -np.inf)],
)
def test_invalid_numeric_inputs(
    function: Callable[..., FloatResult | Greeks], index: int, value: float
) -> None:
    inputs = [100, 100, 1, 0.05, 0.2, 0.02]
    inputs[index] = value
    with pytest.raises(ValueError, match=r"finite|positive|non-negative"):
        function(*inputs)


@pytest.mark.parametrize("function", [bsm_price, bsm_greeks])
def test_invalid_kind_and_shape(function: Callable[..., FloatResult | Greeks]) -> None:
    with pytest.raises(ValueError, match="kind"):
        function(100, 100, 1, 0.05, 0.2, kind=cast(OptionKind, "invalid"))
    with pytest.raises(ValueError, match="shape"):
        function([90, 100], [80, 90, 100], 1, 0.05, 0.2)
