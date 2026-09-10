"""IV acceptance criteria fixed before running, with conditioning reported separately."""

from __future__ import annotations

from collections import Counter
from dataclasses import FrozenInstanceError
from typing import Any, cast

import numpy as np
import pytest

from crossprice.analytic import OptionKind, bsm_greeks, bsm_price
from crossprice.implied_vol import implied_vol

ROUNDTRIP_ATOL = 5e-10
ILL_CONDITIONED_ATOL = 5e-7


@pytest.mark.parametrize("kind", ["call", "put"])
def test_seeded_roundtrip(kind: OptionKind) -> None:
    rng = np.random.default_rng(20260910)
    methods: Counter[str] = Counter()
    errors, iterations, contracts = [], [], []
    for _ in range(256):
        spot = rng.uniform(20, 200)
        tau = rng.uniform(0.01, 3)
        rate, div = rng.uniform(-0.05, 0.10), rng.uniform(-0.03, 0.10)
        vol = rng.uniform(0.05, 1.0)
        log_moneyness = rng.uniform(-2, 2) * vol * np.sqrt(tau)
        strike = spot * np.exp((rate - div) * tau - log_moneyness)
        price = float(bsm_price(spot, strike, tau, rate, vol, div, kind))
        result = implied_vol(spot, strike, tau, rate, price, div, kind)
        assert result.status == "converged", (spot, strike, tau, rate, vol, div, result)
        assert result.vol is not None
        assert abs(result.price_residual) <= 1e-10
        errors.append(abs(result.vol - vol))
        iterations.append(result.iterations)
        contracts.append((spot, strike, tau, rate, vol, div))
        methods[result.method] += 1
    worst = int(np.argmax(errors))
    print(
        f"IV {kind}: n=256; max_error={errors[worst]:.12g}; "
        f"median={np.median(errors):.12g}; p95={np.quantile(errors, 0.95):.12g}; "
        f"max_iterations={max(iterations)}; methods={dict(methods)}; "
        f"worst (S,K,T,r,sigma,q)={contracts[worst]}"
    )
    assert max(errors) <= ROUNDTRIP_ATOL


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("settings", [{"initial_vol": 1e-12}, {"max_newton": 0}])
def test_brent_rescues_zero_vega_or_disabled_newton(kind: OptionKind, settings: dict) -> None:
    price = float(bsm_price(100, 120, 0.25, -0.02, 0.3, -0.01, kind))
    result = implied_vol(100, 120, 0.25, -0.02, price, -0.01, kind, **settings)
    assert result.status == "converged"
    assert result.method == "brent"
    assert result.vol == pytest.approx(0.3, rel=0, abs=ROUNDTRIP_ATOL)
    assert result.iterations > 0
    print(f"forced fallback {kind}: {result}")


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("spot", [80.0, 100.0, 120.0])
def test_bounds_are_explicit(spot: float, kind: OptionKind) -> None:
    lower = float(bsm_price(spot, 100, 1, -0.02, 0, 0.03, kind))
    upper = float(spot * np.exp(-0.03) if kind == "call" else 100 * np.exp(0.02))
    at_lower = implied_vol(spot, 100, 1, -0.02, lower, 0.03, kind)
    at_upper = implied_vol(spot, 100, 1, -0.02, upper, 0.03, kind)
    assert (at_lower.status, at_lower.vol) == ("lower_bound", 0.0)
    assert (at_upper.status, at_upper.vol) == ("upper_bound", None)
    for price in (np.nextafter(lower, -np.inf), np.nextafter(upper, np.inf)):
        with pytest.raises(ValueError, match="no-arbitrage bounds"):
            implied_vol(spot, 100, 1, -0.02, price, 0.03, kind)


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("spot", [80.0, 100.0, 120.0])
def test_expiry_does_not_identify_volatility(spot: float, kind: OptionKind) -> None:
    payoff = float(bsm_price(spot, 100, 0, 0.05, 0.2, kind=kind))
    result = implied_vol(spot, 100, 0, 0.05, payoff, kind=kind)
    assert (result.status, result.vol, result.iterations) == ("expired", None, 0)
    with pytest.raises(ValueError, match="expiry"):
        implied_vol(spot, 100, 0, 0.05, payoff + 0.01, kind=kind)


@pytest.mark.parametrize("rate,div", [(0.05, 0.0), (0.01, 0.03)])
@pytest.mark.parametrize("kind", ["call", "put"])
def test_harness_tiny_time_value_conditioning(rate: float, div: float, kind: OptionKind) -> None:
    price = float(bsm_price(100, 80, 0.25, rate, 0.1, div, kind))
    vega = float(bsm_greeks(100, 80, 0.25, rate, 0.1, div, kind).vega)
    result = implied_vol(100, 80, 0.25, rate, price, div, kind)
    perturbed = implied_vol(100, 80, 0.25, rate, price + 1e-6, div, kind)
    assert result.status == perturbed.status == "converged"
    assert result.vol is not None and perturbed.vol is not None
    error = result.vol - 0.1
    assert abs(error) <= ILL_CONDITIONED_ATOL
    assert perturbed.vol - result.vol > 1e-4
    print(
        f"tiny {kind} r={rate} q={div}: price={price:.12g}; vega={vega:.12g}; "
        f"recovered_sigma_error={error:.12g}; method={result.method}; "
        f"iterations={result.iterations}; +1e-6_price_sigma_shift="
        f"{perturbed.vol - result.vol:.12g}; linear_prediction={1e-6 / vega:.12g}"
    )


def test_one_day_otm_and_rounded_intrinsic_limit() -> None:
    price = float(bsm_price(100, 103, 1 / 365, 0.05, 0.2, 0.02))
    result = implied_vol(100, 103, 1 / 365, 0.05, price, 0.02)
    assert result.status == "converged"
    assert result.vol == pytest.approx(0.2, rel=0, abs=ILL_CONDITIONED_ATOL)
    print(f"one-day OTM: price={price:.12g}; result={result}")
    rounded = float(bsm_price(100, 20, 1 / 365, 0, 0.1))
    assert rounded == 80
    assert implied_vol(100, 20, 1 / 365, 0, rounded).status == "lower_bound"


def test_bracket_expansion_cap_and_iteration_failure() -> None:
    price = float(bsm_price(100, 100, 1, 0.03, 3.0))
    result = implied_vol(100, 100, 1, 0.03, price, initial_vol=0.2)
    assert result.status == "converged"
    assert result.vol == pytest.approx(3, rel=0, abs=ROUNDTRIP_ATOL)
    capped = implied_vol(100, 100, 1, 0.03, price, max_vol=0.5)
    assert (capped.status, capped.vol) == ("no_bracket", None)
    assert capped.price_residual < 0
    for max_newton in (0, 1):
        failed = implied_vol(100, 100, 1, 0.03, price, max_newton=max_newton, max_iterations=1)
        assert failed.status == "max_iterations"
        assert failed.iterations == 1
        assert failed.vol is not None
        assert failed.price_residual == bsm_price(100, 100, 1, 0.03, failed.vol) - price


def test_exact_bracket_endpoint_and_frozen_result() -> None:
    price = float(bsm_price(100, 100, 1, 0, 0.5))
    result = implied_vol(100, 100, 1, 0, price, initial_vol=0.2)
    assert (result.vol, result.method, result.iterations) == (0.5, "bracket", 0)
    with pytest.raises(FrozenInstanceError):
        result.__setattr__("vol", 0.1)
    with pytest.raises(TypeError):
        float(cast(Any, result))


def test_initial_guess_changes_work_not_root() -> None:
    price = float(bsm_price(100, 120, 0.25, -0.02, 0.3, -0.01))
    for guess in (0.02, 0.2, 2.0, 20.0):
        result = implied_vol(100, 120, 0.25, -0.02, price, -0.01, initial_vol=guess)
        assert result.status == "converged"
        assert result.vol == pytest.approx(0.3, rel=0, abs=ROUNDTRIP_ATOL)
        print(f"initial_vol={guess}: {result}")


def test_brent_root_alone_does_not_override_price_tolerance() -> None:
    price = float(bsm_price(100, 120, 0.25, -0.02, 0.3, -0.01))
    result = implied_vol(100, 120, 0.25, -0.02, price, -0.01, max_newton=0, price_atol=1e-30)
    assert result.status == "residual_too_large"
    assert abs(result.price_residual) > 1e-30


@pytest.mark.parametrize(
    "settings",
    [
        {"spot": 0},
        {"strike": -1},
        {"tau": -1},
        {"rate": np.inf},
        {"div": np.nan},
        {"price": np.nan},
        {"price": np.inf},
        {"kind": "invalid"},
        {"spot": [100]},
        {"price_atol": 0},
        {"vol_tol": -1},
        {"max_vol": np.inf},
        {"initial_vol": 0},
        {"initial_vol": np.nan},
        {"initial_vol": [0.2]},
        {"max_newton": -1},
        {"max_newton": True},
        {"max_iterations": 0},
        {"max_iterations": 1.5},
    ],
)
def test_invalid_inputs(settings: dict) -> None:
    inputs = dict(spot=100, strike=100, tau=1, rate=0.05, price=10)
    with pytest.raises(ValueError):
        implied_vol(**(inputs | settings))
