"""Reproducibility, estimator arithmetic and statistical Monte Carlo checks."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.stats import binom, norm

from crossprice.analytic import FloatArray, OptionKind, bsm_price
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


def _replay_reduction_samples(
    seed: int | np.random.SeedSequence, n_paths: int, antithetic: bool, kind: OptionKind = "call"
) -> tuple[FloatArray, FloatArray]:
    normals = np.random.default_rng(seed).standard_normal(n_paths // 2 if antithetic else n_paths)
    normals = np.stack([normals, -normals]) if antithetic else normals[None, :]
    discounted_terminal = 100 * np.exp(-0.02 - 0.5 * 0.2**2 + 0.2 * normals)
    direction = 1 if kind == "call" else -1
    return np.maximum(
        direction * (discounted_terminal - 100 * np.exp(-0.05)), 0
    ), discounted_terminal


@pytest.mark.parametrize("kind", ["call", "put"])
def test_antithetic_se_uses_independent_pairs(kind: OptionKind) -> None:
    seed, n_paths = 67, 10_000
    result = mc_price(
        100, 100, 1, 0.05, 0.2, 0.02, kind, seed=seed, n_paths=n_paths, antithetic=True
    )
    raw_payoffs, _controls = _replay_reduction_samples(seed, n_paths, True, kind)
    pairs = np.mean(raw_payoffs, axis=0)
    assert_allclose(result.price, np.mean(pairs), rtol=0, atol=1e-13)
    assert_allclose(result.se, np.std(pairs, ddof=1) / np.sqrt(n_paths // 2), rtol=0, atol=1e-13)
    assert result.n_samples == n_paths // 2
    assert result.n_paths == result.total_paths == n_paths
    assert result.antithetic
    assert not result.control_variate


@pytest.mark.parametrize("antithetic", [False, True])
def test_control_coefficient_and_se_use_independent_pilot(antithetic: bool) -> None:
    seed, n_paths, pilot_paths = 71, 4096, 1024
    result = mc_price(
        100,
        100,
        1,
        0.05,
        0.2,
        0.02,
        seed=seed,
        n_paths=n_paths,
        antithetic=antithetic,
        control_variate=True,
        pilot_paths=pilot_paths,
    )
    raw_payoffs, raw_controls = _replay_reduction_samples(seed, n_paths, antithetic)
    pilot_payoffs, pilot_controls = _replay_reduction_samples(
        np.random.SeedSequence(seed, spawn_key=(1,)), pilot_paths, antithetic
    )
    pilot_payoff_mean, pilot_control_mean = (
        np.mean(pilot_payoffs, axis=0),
        np.mean(pilot_controls, axis=0),
    )
    beta = np.cov(pilot_control_mean, pilot_payoff_mean, ddof=1)[0, 1] / np.var(
        pilot_control_mean, ddof=1
    )
    expected = np.mean(raw_payoffs, axis=0) - beta * (
        np.mean(raw_controls, axis=0) - 100 * np.exp(-0.02)
    )
    baseline_se = np.sqrt(np.mean(np.var(raw_payoffs, axis=1, ddof=1)) / (n_paths + pilot_paths))
    assert_allclose(result.control_beta, beta, rtol=0, atol=1e-12)
    assert_allclose(result.price, np.mean(expected), rtol=0, atol=1e-12)
    assert_allclose(
        result.se, np.std(expected, ddof=1) / np.sqrt(expected.size), rtol=0, atol=1e-13
    )
    assert_allclose(result.baseline_se, baseline_se, rtol=0, atol=1e-13)
    assert result.variance_reduction == pytest.approx((baseline_se / result.se) ** 2, rel=1e-11)
    assert result.total_paths == n_paths + pilot_paths
    assert result.pilot_paths == pilot_paths
    assert result.control_variate


@pytest.mark.parametrize(
    "antithetic,control_variate",
    [(True, False), (False, True), (True, True)],
    ids=["antithetic", "control", "both"],
)
def test_variance_reduction_at_equal_total_cost(antithetic: bool, control_variate: bool) -> None:
    result = mc_price(
        100,
        100,
        1,
        0.05,
        0.2,
        0.02,
        seed=20260913,
        n_paths=65536,
        antithetic=antithetic,
        control_variate=control_variate,
        pilot_paths=4096,
    )
    baseline = mc_price(100, 100, 1, 0.05, 0.2, 0.02, seed=20260913, n_paths=result.total_paths)
    assert result.se < baseline.se
    assert result.variance_reduction > 1
    assert result == mc_price(
        100,
        100,
        1,
        0.05,
        0.2,
        0.02,
        seed=20260913,
        n_paths=65536,
        antithetic=antithetic,
        control_variate=control_variate,
        pilot_paths=4096,
    )
    print(
        f"reduction anti={antithetic}, control={control_variate}: {result}; "
        f"plain-MC SE at total cost={baseline.se:.12f}"
    )


@pytest.mark.parametrize(
    "antithetic,control_variate",
    [(True, False), (False, True), (True, True)],
    ids=["antithetic", "control", "both"],
)
@pytest.mark.parametrize("kind", ["call", "put"])
def test_reduction_coverage_and_empirical_variance(
    antithetic: bool, control_variate: bool, kind: OptionKind
) -> None:
    repetitions, n_paths, pilot_paths = 200, 8192, 2048
    reference = bsm_price(100, 100, 1, 0.05, 0.2, 0.02, kind)
    results, baselines = [], []
    for seed in range(repetitions):
        result = mc_price(
            100,
            100,
            1,
            0.05,
            0.2,
            0.02,
            kind,
            seed=seed,
            n_paths=n_paths,
            antithetic=antithetic,
            control_variate=control_variate,
            pilot_paths=pilot_paths,
        )
        results.append(result)
        baselines.append(
            mc_price(
                100, 100, 1, 0.05, 0.2, 0.02, kind, seed=seed, n_paths=result.total_paths
            ).price
        )
    estimates = np.array([result.price for result in results])
    covered = sum(result.ci_low <= reference <= result.ci_high for result in results)
    lower, upper = binom.interval(1 - 0.01 / 6, repetitions, 0.95)
    factor = np.var(baselines, ddof=1) / np.var(estimates, ddof=1)
    calibration = np.std(estimates, ddof=1) / np.mean([result.se for result in results])
    assert lower <= covered <= upper
    assert factor > 1.2
    assert 0.75 < calibration < 1.25
    print(
        f"{kind} anti={antithetic}, control={control_variate}: coverage={covered}/200 "
        f"in [{lower:.0f},{upper:.0f}], empirical equal-cost VR={factor:.6f}, "
        f"SD/SE={calibration:.6f}"
    )


@pytest.mark.parametrize("antithetic", [False, True])
def test_common_sample_control_parity(antithetic: bool) -> None:
    call = mc_price(
        100,
        100,
        1,
        0.05,
        0.2,
        0.02,
        "call",
        seed=31,
        n_paths=8192,
        antithetic=antithetic,
        control_variate=True,
        pilot_paths=2048,
    )
    put = mc_price(
        100,
        100,
        1,
        0.05,
        0.2,
        0.02,
        "put",
        seed=31,
        n_paths=8192,
        antithetic=antithetic,
        control_variate=True,
        pilot_paths=2048,
    )
    assert_allclose(call.control_beta - put.control_beta, 1, rtol=0, atol=1e-12)
    assert_allclose(
        call.price - put.price, 100 * np.exp(-0.02) - 100 * np.exp(-0.05), rtol=0, atol=3e-12
    )
    assert_allclose(call.se, put.se, rtol=0, atol=1e-12)


def test_antithetic_helps_little_for_rare_payoffs() -> None:
    result = mc_price(100, 160, 1, 0.05, 0.2, 0.02, seed=20260913, n_paths=100_000, antithetic=True)
    assert 1 <= result.variance_reduction < 1.05
    print(f"rare payoff antithetic: {result}")


def test_uninformative_pilot_can_cost_paths_without_helping() -> None:
    result = mc_price(
        100, 160, 1, 0.05, 0.2, 0.02, seed=91, n_paths=16384, control_variate=True, pilot_paths=2
    )
    assert result.control_beta == 0
    assert result.variance_reduction < 1
    print(f"uninformative pilot: {result}")


def test_constant_pilot_control_disables_correction() -> None:
    with pytest.warns(RuntimeWarning, match="degenerate"):
        result = mc_price(
            100, 90, 1, 0.05, 1e-300, seed=1, n_paths=100, control_variate=True, pilot_paths=100
        )
    assert result.control_beta == 0
    assert result.variance_reduction is None
    assert result.ci_status == "degenerate"


@pytest.mark.parametrize("scale", [1e-200, 1e200])
def test_control_statistics_are_currency_scale_invariant(scale: float) -> None:
    baseline = mc_price(
        100,
        100,
        1,
        0.05,
        0.2,
        seed=43,
        n_paths=2000,
        antithetic=True,
        control_variate=True,
        pilot_paths=512,
    )
    result = mc_price(
        100 * scale,
        100 * scale,
        1,
        0.05,
        0.2,
        seed=43,
        n_paths=2000,
        antithetic=True,
        control_variate=True,
        pilot_paths=512,
    )
    assert_allclose(
        [result.price, result.se, *result.ci],
        np.array([baseline.price, baseline.se, *baseline.ci]) * scale,
        rtol=5e-12,
        atol=0,
    )
    assert_allclose(
        [result.control_beta, result.variance_reduction],
        [baseline.control_beta, baseline.variance_reduction],
        rtol=5e-12,
        atol=0,
    )


@pytest.mark.parametrize("antithetic,control_variate", [(True, False), (False, True), (True, True)])
def test_reduction_preserves_deterministic_contracts(
    antithetic: bool, control_variate: bool
) -> None:
    result = mc_price(
        90,
        100,
        1,
        0.05,
        0,
        kind="put",
        seed=17,
        n_paths=100,
        antithetic=antithetic,
        control_variate=control_variate,
    )
    assert result.price == bsm_price(90, 100, 1, 0.05, 0, kind="put")
    assert result.se == 0
    assert result.ci_status == "deterministic"
    assert result.pilot_paths == 0
    assert result.variance_reduction is None


@pytest.mark.parametrize(
    "antithetic,control_variate,n_paths,pilot_paths",
    [
        ("yes", False, 100, 10),
        (False, 1, 100, 10),
        (True, False, 2, 10),
        (True, False, 101, 10),
        (False, True, 100, 1),
        (True, True, 100, 2),
        (True, True, 100, 11),
    ],
)
def test_invalid_reduction_configuration(
    antithetic: bool, control_variate: bool, n_paths: int, pilot_paths: int
) -> None:
    with pytest.raises(ValueError):
        mc_price(
            100,
            100,
            1,
            0.05,
            0.2,
            seed=17,
            n_paths=n_paths,
            antithetic=antithetic,
            control_variate=control_variate,
            pilot_paths=pilot_paths,
        )
