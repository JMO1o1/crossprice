"""Reproducibility, estimator arithmetic and statistical Monte Carlo checks."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.stats import binom, norm

from crossprice.analytic import FloatArray, OptionKind, bsm_price, geometric_asian_price
from crossprice.montecarlo import AverageKind, MCResult, asian_mc_price, mc_price


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("n_dates", [1, 2, 12, 52])
def test_geometric_asian_formula_against_covariance_sum(kind: OptionKind, n_dates: int) -> None:
    spot, strike, tau, rate, vol, div = 100, 105, 1.5, 0.04, 0.3, 0.02
    dates = np.linspace(tau / n_dates, tau, n_dates)
    log_mean = np.log(spot) + (rate - div - 0.5 * vol**2) * dates.mean()
    variance = vol**2 * np.minimum.outer(dates, dates).mean()
    direction = 1 if kind == "call" else -1
    standardised = (log_mean - np.log(strike)) / np.sqrt(variance)
    expected = (
        np.exp(-rate * tau)
        * direction
        * (
            np.exp(log_mean + 0.5 * variance)
            * norm.cdf(direction * (standardised + np.sqrt(variance)))
            - strike * norm.cdf(direction * standardised)
        )
    )
    actual = geometric_asian_price(spot, strike, tau, rate, vol, div, kind, n_dates=n_dates)
    assert_allclose(actual, expected, rtol=0, atol=2e-12)


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("tau,vol", [(1, 0.2), (1, 0), (0, 0.2), (0.1, 0.8)])
def test_single_date_geometric_asian_is_vanilla(kind: OptionKind, tau: float, vol: float) -> None:
    actual = geometric_asian_price(100, 105, tau, -0.02, vol, 0.03, kind, n_dates=1)
    assert_allclose(actual, bsm_price(100, 105, tau, -0.02, vol, 0.03, kind), rtol=0, atol=2e-12)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_geometric_asian_price_parity_and_deterministic_limit(kind: OptionKind) -> None:
    n_dates, tau, rate, div = 12, 2, 0.05, 0.02
    dates = np.linspace(tau / n_dates, tau, n_dates)
    deterministic_geometric = np.exp(np.mean(np.log(100) + (rate - div) * dates))
    direction = 1 if kind == "call" else -1
    expected = np.exp(-rate * tau) * max(direction * (deterministic_geometric - 105), 0)
    assert_allclose(
        geometric_asian_price(100, 105, tau, rate, 0, div, kind, n_dates=n_dates),
        expected,
        rtol=0,
        atol=2e-12,
    )
    log_mean = np.log(100) + (rate - div - 0.5 * 0.3**2) * dates.mean()
    variance = 0.3**2 * np.minimum.outer(dates, dates).mean()
    call = geometric_asian_price(100, 105, tau, rate, 0.3, div, "call", n_dates=n_dates)
    put = geometric_asian_price(100, 105, tau, rate, 0.3, div, "put", n_dates=n_dates)
    assert_allclose(
        call - put,
        np.exp(-rate * tau) * (np.exp(log_mean + 0.5 * variance) - 105),
        rtol=0,
        atol=2e-12,
    )


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


def _replay_asian_samples(
    seed: int | np.random.SeedSequence,
    n_paths: int,
    n_dates: int,
    antithetic: bool,
    kind: OptionKind = "call",
) -> tuple[FloatArray, FloatArray]:
    n_samples = n_paths // 2 if antithetic else n_paths
    normals = np.random.default_rng(seed).standard_normal((n_dates, n_samples))
    normals = np.stack((normals, -normals)) if antithetic else normals[None, :, :]
    increments = (0.05 - 0.02 - 0.5 * 0.2**2) / n_dates + 0.2 / np.sqrt(n_dates) * normals
    paths = 100 * np.exp(np.cumsum(increments, axis=1))
    direction = 1 if kind == "call" else -1
    arithmetic = np.exp(-0.05) * np.maximum(direction * (np.mean(paths, axis=1) - 100), 0)
    geometric = np.exp(-0.05) * np.maximum(
        direction * (np.exp(np.mean(np.log(paths), axis=1)) - 100), 0
    )
    return arithmetic, geometric


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("average", ["arithmetic", "geometric"])
@pytest.mark.parametrize("antithetic", [False, True])
def test_asian_streaming_matches_dense_path_replay(
    kind: OptionKind, average: AverageKind, antithetic: bool
) -> None:
    result = asian_mc_price(
        100,
        100,
        1,
        0.05,
        0.2,
        0.02,
        kind,
        seed=29,
        n_paths=4096,
        n_dates=12,
        average=average,
        antithetic=antithetic,
    )
    arithmetic, geometric = _replay_asian_samples(29, 4096, 12, antithetic, kind)
    observations = np.mean(arithmetic if average == "arithmetic" else geometric, axis=0)
    assert_allclose(result.price, np.mean(observations), rtol=0, atol=1e-12)
    assert_allclose(
        result.se, np.std(observations, ddof=1) / np.sqrt(observations.size), rtol=0, atol=1e-13
    )
    assert result.n_dates == 12
    assert result.average == average
    assert result.n_samples == observations.size


@pytest.mark.parametrize("antithetic", [False, True])
def test_asian_geometric_control_uses_matching_discrete_expectation(antithetic: bool) -> None:
    seed, n_paths, n_dates, pilot_paths = 33, 4096, 12, 1024
    result = asian_mc_price(
        100,
        100,
        1,
        0.05,
        0.2,
        0.02,
        seed=seed,
        n_paths=n_paths,
        n_dates=n_dates,
        antithetic=antithetic,
        control_variate=True,
        pilot_paths=pilot_paths,
    )
    raw_payoffs, raw_controls = _replay_asian_samples(seed, n_paths, n_dates, antithetic)
    pilot_payoffs, pilot_controls = _replay_asian_samples(
        np.random.SeedSequence(seed, spawn_key=(1,)), pilot_paths, n_dates, antithetic
    )
    pilot_payoffs, pilot_controls = np.mean(pilot_payoffs, axis=0), np.mean(pilot_controls, axis=0)
    beta = np.cov(pilot_controls, pilot_payoffs, ddof=1)[0, 1] / np.var(pilot_controls, ddof=1)
    expected_control = geometric_asian_price(100, 100, 1, 0.05, 0.2, 0.02, n_dates=n_dates)
    corrected = np.mean(raw_payoffs, axis=0) - beta * (
        np.mean(raw_controls, axis=0) - expected_control
    )
    assert_allclose(result.control_beta, beta, rtol=0, atol=1e-12)
    assert_allclose(result.price, np.mean(corrected), rtol=0, atol=1e-12)
    assert_allclose(
        result.se, np.std(corrected, ddof=1) / np.sqrt(corrected.size), rtol=0, atol=1e-13
    )
    assert result.total_paths == n_paths + pilot_paths


@pytest.mark.parametrize("kind", ["call", "put"])
def test_geometric_asian_mc_coverage(kind: OptionKind) -> None:
    reference = geometric_asian_price(100, 100, 1, 0.05, 0.2, 0.02, kind, n_dates=12)
    results = [
        asian_mc_price(
            100,
            100,
            1,
            0.05,
            0.2,
            0.02,
            kind,
            seed=seed,
            n_paths=4096,
            n_dates=12,
            average="geometric",
        )
        for seed in range(200)
    ]
    covered = sum(result.ci_low <= reference <= result.ci_high for result in results)
    lower, upper = binom.interval(1 - 0.01 / 2, 200, 0.95)
    assert lower <= covered <= upper
    print(
        f"geometric Asian {kind}: exact={reference:.12f}, coverage={covered}/200 "
        f"in [{lower:.0f},{upper:.0f}]"
    )


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize(
    "inputs",
    [
        (100, 100, 1, 0.05, 0.2, 0.02),
        (120, 100, 1, 0.01, 0.2, 0.1),
        (100, 120, 2, -0.02, 0.3, 0.03),
    ],
)
def test_asian_am_gm_and_jensen_bounds(kind: OptionKind, inputs: tuple[float, ...]) -> None:
    spot, strike, tau, rate, vol, div = inputs
    n_dates = 12
    arithmetic = asian_mc_price(
        *inputs, kind=kind, seed=121, n_paths=32768, n_dates=n_dates, antithetic=True
    )
    geometric = asian_mc_price(
        *inputs,
        kind=kind,
        seed=121,
        n_paths=32768,
        n_dates=n_dates,
        average="geometric",
        antithetic=True,
    )
    exact_geometric = geometric_asian_price(*inputs, kind=kind, n_dates=n_dates)
    if kind == "call":
        assert arithmetic.price >= geometric.price - 1e-12
        assert arithmetic.price + 4 * arithmetic.se >= exact_geometric
    else:
        assert arithmetic.price <= geometric.price + 1e-12
        assert arithmetic.price - 4 * arithmetic.se <= exact_geometric
    dates = np.linspace(tau / n_dates, tau, n_dates)
    convexity_upper = np.mean(
        np.exp(-rate * (tau - dates)) * bsm_price(spot, strike, dates, rate, vol, div, kind)
    )
    discounted_mean = np.mean(spot * np.exp((rate - div) * dates - rate * tau))
    direction = 1 if kind == "call" else -1
    lower_bound = max(direction * (discounted_mean - strike * np.exp(-rate * tau)), 0)
    assert arithmetic.price - 4 * arithmetic.se <= convexity_upper
    assert arithmetic.price + 4 * arithmetic.se >= lower_bound


@pytest.mark.parametrize("kind", ["call", "put"])
def test_asian_terminal_bound_when_spot_is_a_martingale(kind: OptionKind) -> None:
    result = asian_mc_price(
        100,
        100,
        1,
        0.03,
        0.2,
        0.03,
        kind,
        seed=92,
        n_paths=32768,
        n_dates=12,
        antithetic=True,
        control_variate=True,
    )
    vanilla = bsm_price(100, 100, 1, 0.03, 0.2, 0.03, kind)
    assert result.price + 4 * result.se < vanilla


def test_asian_not_unconditionally_bounded_by_terminal_vanilla() -> None:
    result = asian_mc_price(120, 100, 1, 0, 0, 0.1, seed=1, n_paths=4, n_dates=12)
    vanilla = bsm_price(120, 100, 1, 0, 0, 0.1)
    assert result.price > vanilla + 1
    assert result.se == 0
    assert result.ci_status == "deterministic"
    print(f"terminal-bound counterexample: Asian={result}; vanilla={vanilla:.12f}")


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("antithetic", [False, True])
def test_asian_control_empirical_variance_and_mean(kind: OptionKind, antithetic: bool) -> None:
    controlled, plain, reported_se = [], [], []
    for seed in range(128):
        result = asian_mc_price(
            100,
            100,
            1,
            0.05,
            0.2,
            0.02,
            kind,
            seed=seed,
            n_paths=4096,
            n_dates=12,
            antithetic=antithetic,
            control_variate=True,
            pilot_paths=1024,
        )
        baseline = asian_mc_price(
            100, 100, 1, 0.05, 0.2, 0.02, kind, seed=seed, n_paths=result.total_paths, n_dates=12
        )
        controlled.append(result.price)
        plain.append(baseline.price)
        reported_se.append(result.se)
    paired_differences = np.array(controlled) - np.array(plain)
    difference_se = np.std(paired_differences, ddof=1) / np.sqrt(len(controlled))
    factor = np.var(plain, ddof=1) / np.var(controlled, ddof=1)
    calibration = np.std(controlled, ddof=1) / np.mean(reported_se)
    assert abs(np.mean(paired_differences)) < 4 * difference_se
    assert factor > 10
    assert 0.7 < calibration < 1.3
    print(
        f"Asian {kind} anti={antithetic}: equal-cost empirical VR={factor:.6f}, "
        f"SD/SE={calibration:.6f}, "
        f"mean-difference z={np.mean(paired_differences) / difference_se:.6f}"
    )


@pytest.mark.parametrize("kind", ["call", "put"])
def test_asian_showcase_reports_uncertainty_and_reduction(kind: OptionKind) -> None:
    result = asian_mc_price(
        100,
        100,
        1,
        0.05,
        0.2,
        0.02,
        kind,
        seed=20260914,
        n_paths=65536,
        n_dates=12,
        antithetic=True,
        control_variate=True,
        pilot_paths=4096,
    )
    baseline = asian_mc_price(
        100, 100, 1, 0.05, 0.2, 0.02, kind, seed=20260915, n_paths=result.total_paths, n_dates=12
    )
    geometric = geometric_asian_price(100, 100, 1, 0.05, 0.2, 0.02, kind, n_dates=12)
    assert abs(result.price - baseline.price) < 4 * np.hypot(result.se, baseline.se)
    assert result.se < baseline.se
    assert result.variance_reduction > 10
    assert result == asian_mc_price(
        100,
        100,
        1,
        0.05,
        0.2,
        0.02,
        kind,
        seed=20260914,
        n_paths=65536,
        n_dates=12,
        antithetic=True,
        control_variate=True,
        pilot_paths=4096,
    )
    print(f"Asian {kind}: {result}; exact geometric={geometric:.12f}; plain={baseline}")


def test_monitoring_frequency_is_a_contract_parameter() -> None:
    counts = np.array([4, 12, 24, 48, 96])
    geometric_prices, arithmetic_results = [], []
    for n_dates in counts:
        result = asian_mc_price(
            100,
            100,
            1,
            0.05,
            0.2,
            0.02,
            seed=20260914,
            n_paths=32768,
            n_dates=n_dates,
            antithetic=True,
            control_variate=True,
            pilot_paths=2048,
        )
        geometric = geometric_asian_price(100, 100, 1, 0.05, 0.2, 0.02, n_dates=n_dates)
        assert result.price + 4 * result.se >= geometric
        geometric_prices.append(geometric)
        arithmetic_results.append(result)
        print(f"monitoring m={n_dates}: arithmetic={result}; exact geometric={geometric:.12f}")
    log_shift, variance = (0.05 - 0.02 - 0.5 * 0.2**2) / 2, 0.2**2 / 3
    argument = log_shift / np.sqrt(variance)
    continuous_geometric = (
        100
        * np.exp(-0.05)
        * (
            np.exp(log_shift + 0.5 * variance) * norm.cdf(argument + np.sqrt(variance))
            - norm.cdf(argument)
        )
    )
    errors = np.array(geometric_prices) - continuous_geometric
    order = -np.polyfit(np.log(counts), np.log(errors), 1)[0]
    assert 0.85 < order < 1.15
    assert abs(arithmetic_results[-1].price - arithmetic_results[-2].price) < 0.08
    print(
        f"continuous geometric={continuous_geometric:.12f}, monitoring order={order:.6f}; "
        "arithmetic intervals exclude monitoring error"
    )


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("average", ["arithmetic", "geometric"])
@pytest.mark.parametrize("antithetic", [False, True])
def test_one_monitoring_date_reproduces_vanilla_mc(
    kind: OptionKind, average: AverageKind, antithetic: bool
) -> None:
    asian = asian_mc_price(
        100,
        100,
        1,
        0.05,
        0.2,
        0.02,
        kind,
        seed=52,
        n_paths=4096,
        n_dates=1,
        average=average,
        antithetic=antithetic,
    )
    vanilla = mc_price(
        100, 100, 1, 0.05, 0.2, 0.02, kind, seed=52, n_paths=4096, antithetic=antithetic
    )
    assert_allclose(
        [asian.price, asian.se, *asian.ci],
        [vanilla.price, vanilla.se, *vanilla.ci],
        rtol=0,
        atol=1e-12,
    )


@pytest.mark.parametrize("kind", ["call", "put"])
def test_one_date_perfect_geometric_control_is_explicitly_exact(kind: OptionKind) -> None:
    result = asian_mc_price(
        100, 100, 1, 0.05, 0.2, 0.02, kind, seed=52, n_paths=4096, n_dates=1, control_variate=True
    )
    assert_allclose(result.price, bsm_price(100, 100, 1, 0.05, 0.2, 0.02, kind), rtol=0, atol=1e-12)
    assert result.se == 0
    assert result.ci == (result.price, result.price)
    assert result.ci_status == "deterministic"
    assert result.pilot_paths == 0
    assert result.control_beta == 1
    assert result.baseline_se is None


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("average", ["arithmetic", "geometric"])
@pytest.mark.parametrize("tau,vol", [(0, 0), (0, 0.2), (1, 0)])
def test_asian_deterministic_limits(
    kind: OptionKind, average: AverageKind, tau: float, vol: float
) -> None:
    dates = np.linspace(tau / 12, tau, 12)
    path = 105 * np.exp((0.03 - 0.07) * dates)
    average_spot = np.mean(path) if average == "arithmetic" else np.exp(np.mean(np.log(path)))
    direction = 1 if kind == "call" else -1
    expected = np.exp(-0.03 * tau) * max(direction * (average_spot - 100), 0)
    result = asian_mc_price(
        105, 100, tau, 0.03, vol, 0.07, kind, seed=52, n_paths=4, n_dates=12, average=average
    )
    assert_allclose(result.price, expected, rtol=0, atol=1e-12)
    assert result.se == 0
    assert result.ci_status == "deterministic"


@pytest.mark.parametrize("n_dates", [0, -1, 1.5, True])
def test_invalid_asian_monitoring_count(n_dates: int) -> None:
    with pytest.raises(ValueError, match="n_dates"):
        asian_mc_price(100, 100, 1, 0.05, 0.2, seed=1, n_dates=n_dates)
    with pytest.raises(ValueError, match="n_dates"):
        geometric_asian_price(100, 100, 1, 0.05, 0.2, n_dates=n_dates)


def test_invalid_asian_average_or_tautological_control() -> None:
    with pytest.raises(ValueError, match="average"):
        asian_mc_price(100, 100, 1, 0.05, 0.2, seed=1, average="invalid")
    with pytest.raises(ValueError, match="geometric"):
        asian_mc_price(100, 100, 1, 0.05, 0.2, seed=1, average="geometric", control_variate=True)


def test_asian_rare_payoff_is_flagged() -> None:
    with pytest.warns(RuntimeWarning, match="degenerate"):
        result = asian_mc_price(100, 1000, 1, 0, 0.2, seed=17, n_paths=1000)
    assert result.ci_status == "degenerate"
    assert result.variance_reduction is None
