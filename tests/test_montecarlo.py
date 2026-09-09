"""Reproducibility, estimator arithmetic and statistical Monte Carlo checks."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.stats import binom, norm

from crossprice.analytic import OptionKind, bsm_price
from crossprice.montecarlo import MCResult, mc_price


@pytest.mark.parametrize("kind", ["call", "put"])
def test_mc_matches_replayed_discounted_payoffs(kind: OptionKind) -> None:
    spot, strike, tau, rate, vol, div = 100, 105, 0.75, 0.04, 0.3, 0.02
    seed, n_paths = 20260912, 4096
    result = mc_price(spot, strike, tau, rate, vol, div, kind, seed=seed, n_paths=n_paths)
    terminal = np.random.default_rng(seed).lognormal(
        np.log(spot) + (rate - div - 0.5 * vol**2) * tau, vol * np.sqrt(tau), n_paths
    )
    direction = 1 if kind == "call" else -1
    payoffs = np.exp(-rate * tau) * np.maximum(direction * (terminal - strike), 0)
    assert isinstance(result, MCResult)
    assert_allclose(result.price, np.mean(payoffs), rtol=0, atol=1e-12)
    assert_allclose(result.se, np.std(payoffs, ddof=1) / np.sqrt(n_paths), rtol=0, atol=1e-13)
    assert_allclose(
        result.ci,
        result.price + np.array([-1, 1]) * norm.ppf(0.975) * result.se,
        rtol=0,
        atol=1e-14,
    )
    assert result.n_paths == result.n_samples == n_paths
    assert result.seed == seed
    assert result.ci_status == "normal"
    print(f"{kind} seed replay: {result}")


def test_explicit_seed_reproducibility() -> None:
    first = mc_price(100, 100, 1, 0.05, 0.2, seed=42, n_paths=1000)
    assert first == mc_price(100, 100, 1, 0.05, 0.2, seed=42, n_paths=1000)
    assert first.price != mc_price(100, 100, 1, 0.05, 0.2, seed=43, n_paths=1000).price
    with pytest.raises(TypeError):
        float(first)


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("tau,vol", [(0, 0), (0, 0.2), (1, 0)])
def test_mc_deterministic_limits(tau: float, vol: float, kind: OptionKind) -> None:
    result = mc_price(90, 100, tau, 0.05, vol, 0.02, kind, seed=17, n_paths=2)
    expected = bsm_price(90, 100, tau, 0.05, vol, 0.02, kind)
    assert result.price == expected
    assert result.se == 0
    assert result.ci == (expected, expected)
    assert result.ci_status == "deterministic"


def test_no_hits_is_flagged_not_claimed_exact() -> None:
    with pytest.warns(RuntimeWarning, match="degenerate"):
        result = mc_price(100, 1000, 1, 0, 0.2, seed=17, n_paths=1000)
    assert result.price == result.se == 0
    assert result.ci_status == "degenerate"
    assert bsm_price(100, 1000, 1, 0, 0.2) > 0


@pytest.mark.parametrize("scale", [1e-200, 1e200])
def test_price_and_uncertainty_scale_with_currency_units(scale: float) -> None:
    baseline = mc_price(100, 100, 1, 0.05, 0.2, seed=43, n_paths=2000)
    result = mc_price(100 * scale, 100 * scale, 1, 0.05, 0.2, seed=43, n_paths=2000)
    assert_allclose(
        [result.price, result.se, *result.ci],
        np.array([baseline.price, baseline.se, *baseline.ci]) * scale,
        rtol=5e-14,
        atol=0,
    )
    assert result.ci_status == "normal"


@pytest.mark.parametrize("n_paths", [0, 1, -2, 1.5, True])
def test_invalid_path_count(n_paths: int) -> None:
    with pytest.raises(ValueError, match="n_paths"):
        mc_price(100, 100, 1, 0.05, 0.2, seed=42, n_paths=n_paths)


@pytest.mark.parametrize("seed", [-1, 1.5, True, None])
def test_invalid_seed(seed: int) -> None:
    with pytest.raises(ValueError, match="seed"):
        mc_price(100, 100, 1, 0.05, 0.2, seed=seed)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_nominal_95_percent_coverage_across_seeds(kind: OptionKind) -> None:
    reference = bsm_price(100, 100, 1, 0.05, 0.2, 0.02, kind)
    repetitions = 200
    results = [
        mc_price(100, 100, 1, 0.05, 0.2, 0.02, kind, seed=seed, n_paths=8192)
        for seed in range(repetitions)
    ]
    covered = sum(result.ci_low <= reference <= result.ci_high for result in results)
    lower, upper = binom.interval(0.99, repetitions, 0.95)
    assert lower <= covered <= upper
    empirical_sd = np.std([result.price for result in results], ddof=1)
    mean_se = np.mean([result.se for result in results])
    assert 0.8 < empirical_sd / mean_se < 1.2
    print(
        f"{kind} coverage: {covered}/{repetitions}, binomial 99% band=[{lower:.0f},{upper:.0f}], "
        f"empirical SD / mean SE={empirical_sd / mean_se:.6f}"
    )


@pytest.mark.parametrize("kind", ["call", "put"])
def test_rms_error_converges_as_inverse_square_root(kind: OptionKind) -> None:
    counts = np.array([1024, 4096, 16384, 65536])
    reference = bsm_price(100, 100, 1, 0.05, 0.2, 0.02, kind)
    rms_errors = []
    for n_paths in counts:
        errors = [
            mc_price(100, 100, 1, 0.05, 0.2, 0.02, kind, seed=1000 + seed, n_paths=n_paths).price
            - reference
            for seed in range(128)
        ]
        rms_errors.append(float(np.sqrt(np.mean(np.square(errors)))))
    slope = np.polyfit(np.log(counts), np.log(rms_errors), 1)[0]
    assert -0.6 < slope < -0.4
    print(f"{kind} RMS convergence: N={counts.tolist()}, RMSE={rms_errors}, slope={slope:.6f}")


def test_parity_with_independent_samples_uses_combined_se() -> None:
    call = mc_price(100, 100, 1, 0.05, 0.2, 0.02, "call", seed=21, n_paths=100_000)
    put = mc_price(100, 100, 1, 0.05, 0.2, 0.02, "put", seed=22, n_paths=100_000)
    expected = 100 * np.exp(-0.02) - 100 * np.exp(-0.05)
    z_score = (call.price - put.price - expected) / np.hypot(call.se, put.se)
    assert abs(z_score) < norm.ppf(0.9995)
    print(f"independent-sample parity z={z_score:.6f}; two-sided 99.9% acceptance")


def test_sample_mean_can_cross_an_exact_price_bound() -> None:
    result = mc_price(150, 100, 1, 0.05, 0.05, seed=42, n_paths=2000)
    lower_bound = 150 - 100 * np.exp(-0.05)
    assert result.price < lower_bound
    assert lower_bound < result.price + 4 * result.se
    print(f"sampling bound counterexample: {result}; exact lower bound={lower_bound:.12f}")


@pytest.mark.parametrize("kind", ["call", "put"])
def test_bounds_with_sampling_uncertainty(kind: OptionKind) -> None:
    direction = 1 if kind == "call" else -1
    for spot, strike in ((100, 100), (150, 100), (100, 130)):
        result = mc_price(spot, strike, 1, 0.05, 0.2, 0.02, kind, seed=37, n_paths=100_000)
        discounted_spot, discounted_strike = spot * np.exp(-0.02), strike * np.exp(-0.05)
        lower = max(direction * (discounted_spot - discounted_strike), 0)
        upper = discounted_spot if kind == "call" else discounted_strike
        margin = norm.ppf(0.9995) * result.se
        assert result.price + margin >= lower
        assert result.price - margin <= upper


@pytest.mark.parametrize("strike", [100, 160], ids=["atm", "otm"])
def test_precision_budget_reports_absolute_and_relative_error(strike: float) -> None:
    n_paths = 100_000
    result = mc_price(100, strike, 1, 0.05, 0.2, 0.02, seed=20260912, n_paths=n_paths)
    reference = bsm_price(100, strike, 1, 0.05, 0.2, 0.02)
    assert abs(result.price - reference) < 4 * result.se
    assert result.ci_status == "normal"
    half_width = norm.ppf(0.975) * result.se
    cent_budget = int(np.ceil(n_paths * (half_width / 0.01) ** 2))
    relative_budget = int(np.ceil(n_paths * (half_width / (0.01 * reference)) ** 2))
    print(
        f"precision K={strike}: {result}; BSM={reference:.12f}; "
        f"relative 95% half-width={half_width / reference:.6f}; "
        f"estimated N for 0.01 half-width={cent_budget}, for 1% half-width={relative_budget}"
    )


@pytest.mark.parametrize(
    "index,value", [(0, 0), (1, 0), (2, -1), (4, -1)] + [(index, np.inf) for index in range(6)]
)
def test_invalid_mc_contract_inputs(index: int, value: float) -> None:
    inputs = [100, 100, 1, 0.05, 0.2, 0.02]
    inputs[index] = value
    with pytest.raises(ValueError):
        mc_price(*inputs, seed=17)
