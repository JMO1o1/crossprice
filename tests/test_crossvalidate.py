"""End-to-end, schema, determinism, diagnostic and mutation checks for the harness."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from crossprice import crossvalidate as cv
from crossprice.binomial import crr_price
from crossprice.montecarlo import MCResult, mc_price

ALLOWED_STATUS = {"pass", "expected", "degenerate", "FAIL"}
ALLOWED_OUTCOME = {"agree", "disagree", "degenerate", "error", "unsupported"}
ALLOWED_CRITERION = {"tolerance", "exact", "z_score", "lower_bound", "upper_bound", "none"}


@pytest.fixture(scope="module")
def quick_run() -> cv.AgreementRun:
    return cv.run_agreement(cv.quick_cases(), cv.quick_config())


def _rows(run: cv.AgreementRun, case_id: str, method: str) -> pd.DataFrame:
    table = run.table
    return table[(table["case_id"] == case_id) & (table["method"] == method)]


def test_quick_run_covers_every_group_without_failures(quick_run: cv.AgreementRun) -> None:
    table = quick_run.table
    assert set(table["group"]) == {"european", "american", "asian", "stress"}
    assert set(table["instrument"]) == {"european", "american", "asian"}
    assert (table["status"] != "FAIL").all()
    assert (table["status"] == "pass").sum() > 40
    assert set(table.loc[table["status"] == "expected", "outcome"]) == {
        "unsupported",
        "error",
        "degenerate",
    }
    assert set(table["criterion"]) == ALLOWED_CRITERION
    unsupported = table[table["outcome"] == "unsupported"]
    assert unsupported["reason"].str.len().gt(20).all()
    assert unsupported["price"].isna().all()
    print(table["status"].value_counts().to_dict(), table["outcome"].value_counts().to_dict())


def test_schema_is_stable(quick_run: cv.AgreementRun) -> None:
    table = quick_run.table
    assert tuple(table.columns) == cv.COLUMNS
    for name in ("seed", "steps", "n_paths", "n_samples", "pilot_paths", "total_paths", "n_dates"):
        assert str(table[name].dtype) == "Int64"
    for name in ("antithetic", "control_variate", "covered_95"):
        assert str(table[name].dtype) == "boolean"
    for name in ("price", "reference_price", "error", "abs_error", "rel_error", "se", "z_score"):
        assert table[name].dtype == np.float64
    assert set(table["status"]) <= ALLOWED_STATUS
    assert set(table["outcome"]) <= ALLOWED_OUTCOME
    assert set(table["expected"]) <= ALLOWED_OUTCOME
    assert (table.loc[table["status"] != "pass", "reason"].str.len() > 0).all()
    stochastic = table[
        table["method"].isin(cv.STOCHASTIC_METHODS) & (table["outcome"] != "unsupported")
    ]
    assert stochastic["seed"].notna().all()
    assert stochastic["n_paths"].notna().all()
    assert stochastic["ci_status"].isin(["normal", "deterministic", "degenerate"]).all()
    reduced = table[table["method"] == "mc_antithetic_control"]
    assert (reduced["total_paths"] == reduced["n_paths"] + reduced["pilot_paths"]).all()
    assert reduced["antithetic"].all() and reduced["control_variate"].all()
    asian = table[(table["method"] == "asian_mc_arithmetic_control")]
    assert (asian["average"] == "arithmetic").all() and (asian["n_dates"] == 12).all()


def test_regeneration_is_deterministic(quick_run: cv.AgreementRun) -> None:
    again = cv.run_agreement(cv.quick_cases(), cv.quick_config())
    pd.testing.assert_frame_equal(quick_run.table, again.table)
    assert again.z_crit == quick_run.z_crit
    config = cv.quick_config()
    pd.testing.assert_frame_equal(cv.crr_convergence(config), cv.crr_convergence(config))
    pd.testing.assert_frame_equal(cv.mc_convergence(config), cv.mc_convergence(config))


def test_seeds_are_distinct_and_stable() -> None:
    seeds = {
        cv.seed_for(f"case-{i}", method, 1) for i in range(50) for method in cv.STOCHASTIC_METHODS
    }
    assert len(seeds) == 200
    assert cv.seed_for("stress-deep_otm_call", "mc_plain", 20260916) == cv.seed_for(
        "stress-deep_otm_call", "mc_plain", 20260916
    )
    assert cv.seed_for("a", "mc_plain", 1) != cv.seed_for("a", "mc_plain", 2)


def test_z_threshold_is_fixed_from_declared_stochastic_rows(quick_run: cv.AgreementRun) -> None:
    table = quick_run.table
    declared = table[
        table["method"].isin(cv.STOCHASTIC_METHODS) & (table["outcome"] != "unsupported")
    ]
    assert quick_run.n_stochastic == len(declared)
    from scipy.special import ndtri

    assert quick_run.z_crit == pytest.approx(ndtri(1 - 0.01 / (2 * len(declared))))
    scored = table["z_score"].dropna()
    assert (scored.abs() <= quick_run.z_crit).all()


def test_zero_reference_has_undefined_relative_error(quick_run: cv.AgreementRun) -> None:
    for method in ("crr", "mc_plain", "mc_antithetic_control"):
        row = _rows(quick_run, "stress-expiry_otm_call", method).iloc[0]
        assert row["reference_price"] == 0 and row["price"] == 0
        assert row["abs_error"] == 0 and np.isnan(row["rel_error"])
        assert row["criterion"] == "exact" and row["status"] == "pass"
        assert np.isnan(row["z_score"])


def test_zero_se_rows_use_exact_criterion_not_z(quick_run: cv.AgreementRun) -> None:
    for method in ("mc_plain", "mc_antithetic_control"):
        row = _rows(quick_run, "stress-zero_vol_call", method).iloc[0]
        assert row["ci_status"] == "deterministic" and row["se"] == 0
        assert row["criterion"] == "exact" and row["status"] == "pass"
        assert np.isnan(row["z_score"]) and pd.isna(row["covered_95"])
    tree = _rows(quick_run, "stress-zero_vol_call", "crr").iloc[0]
    assert tree["criterion"] == "exact" and tree["abs_error"] <= 1e-10


def test_degenerate_sample_is_recorded_not_scored(quick_run: cv.AgreementRun) -> None:
    row = _rows(quick_run, "stress-degenerate_mc", "mc_plain").iloc[0]
    assert row["outcome"] == "degenerate" and row["status"] == "expected"
    assert row["ci_status"] == "degenerate" and row["se"] == 0 and row["price"] == 0
    assert row["reference_price"] > 0 and np.isnan(row["z_score"])
    assert "no sample variation" in row["reason"]
    perfect = _rows(quick_run, "stress-deep_itm_call", "mc_antithetic_control").iloc[0]
    assert perfect["ci_status"] == "normal" and 0 <= perfect["se"] < 1e-12
    assert perfect["outcome"] == "degenerate" and perfect["status"] == "expected"
    assert "resolution floor" in perfect["reason"] and np.isnan(perfect["z_score"])


def test_crr_probability_rejection_is_recorded(quick_run: cv.AgreementRun) -> None:
    rows = _rows(quick_run, "stress-low_vol_coarse_tree", "crr")
    coarse = rows[rows["steps"] == 16].iloc[0]
    assert coarse["outcome"] == "error" and coarse["status"] == "expected"
    assert "probability" in coarse["reason"] and np.isnan(coarse["price"])
    assert coarse["reference_price"] > 0
    fine = rows[rows["steps"] == 1000].iloc[0]
    assert fine["status"] == "pass" and fine["criterion"] == "tolerance"


def test_small_reference_relative_error_is_logged(quick_run: cv.AgreementRun) -> None:
    row = _rows(quick_run, "stress-deep_otm_call", "crr").iloc[0]
    assert row["status"] == "pass"
    assert abs(row["rel_error"]) > cv.quick_config().rtol
    assert "passes on atol" in row["reason"]


def test_unsupported_pairs_are_kept_with_reasons(quick_run: cv.AgreementRun) -> None:
    table = quick_run.table
    american_bsm = table[(table["instrument"] == "american") & (table["method"] == "bsm")]
    assert len(american_bsm) >= 1
    assert (american_bsm["outcome"] == "unsupported").all()
    assert american_bsm["reason"].str.contains("not an American reference").all()
    asian_tree = table[(table["instrument"] == "asian") & (table["method"] == "crr")]
    assert asian_tree["reason"].str.contains("running-average").all()


def test_asian_rows_use_bounds_not_ground_truth(quick_run: cv.AgreementRun) -> None:
    table = quick_run.table
    geometric = table[table["method"] == "asian_mc_geometric"].iloc[0]
    assert geometric["criterion"] == "z_score" and geometric["reference"].startswith("geometric")
    arithmetic = table[table["method"] == "asian_mc_arithmetic_control"].iloc[0]
    assert arithmetic["criterion"] == "lower_bound"
    assert "not ground truth" in arithmetic["reference"]
    assert arithmetic["error"] > 0
    assert np.isnan(arithmetic["z_score"])


def test_american_reference_is_labelled_same_implementation(quick_run: cv.AgreementRun) -> None:
    rows = quick_run.table[quick_run.table["method"] == "crr_american"]
    assert rows["reference"].str.contains("same implementation").all()
    assert rows["reference"].str.contains("steps=1600").any()
    assert (rows["status"] == "pass").all()


def test_corrupted_tree_is_detected(monkeypatch: pytest.MonkeyPatch) -> None:
    def scaled(*args: object, **kwargs: object) -> float:
        return 1.01 * crr_price(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(cv, "crr_price", scaled)
    cases = cv.european_grid(cv.GridSpec(strikes=(100.0,), taus=(1.0,), vols=(0.2,)))
    run = cv.run_agreement(cases, cv.HarnessConfig())
    tree = run.table[run.table["method"] == "crr"]
    assert (tree["status"] == "FAIL").all() and (tree["outcome"] == "disagree").all()
    assert tree["reason"].str.contains("exceeds atol").all()
    assert (run.table[run.table["method"] != "crr"]["status"] != "FAIL").all()


def test_biased_monte_carlo_is_detected(monkeypatch: pytest.MonkeyPatch) -> None:
    def biased(*args: object, **kwargs: object) -> MCResult:
        result = mc_price(*args, **kwargs)  # type: ignore[arg-type]
        return replace(
            result,
            price=result.price + 0.5,
            ci_low=result.ci_low + 0.5,
            ci_high=result.ci_high + 0.5,
        )

    monkeypatch.setattr(cv, "mc_price", biased)
    cases = cv.european_grid(cv.GridSpec(strikes=(100.0,), taus=(1.0,), vols=(0.2,)))
    run = cv.run_agreement(cases, cv.HarnessConfig())
    stochastic = run.table[run.table["method"].isin(["mc_plain", "mc_antithetic_control"])]
    assert (stochastic["status"] == "FAIL").all()
    assert stochastic["reason"].str.contains(r"\|z\|=").all()
    assert (stochastic["covered_95"] == False).all()  # noqa: E712 - nullable boolean column
    assert (run.table[run.table["method"] == "crr"]["status"] == "pass").all()


def test_clipping_instead_of_raising_breaks_the_declared_expectation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def clipping(*args: object, **kwargs: object) -> float:
        try:
            return crr_price(*args, **kwargs)  # type: ignore[arg-type]
        except ValueError:
            return float(cv.bsm_price(*args))  # type: ignore[arg-type]

    monkeypatch.setattr(cv, "crr_price", clipping)
    run = cv.run_agreement([case for case in cv.stress_cases() if "coarse" in case.case_id])
    coarse = run.table[(run.table["method"] == "crr") & (run.table["steps"] == 16)].iloc[0]
    assert coarse["outcome"] == "agree" and coarse["status"] == "FAIL"
    assert coarse["reason"].startswith("expected error, got agree")


def test_unexpected_exceptions_propagate(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(*args: object, **kwargs: object) -> float:
        raise RuntimeError("simulated implementation bug")

    monkeypatch.setattr(cv, "crr_price", broken)
    with pytest.raises(RuntimeError, match="simulated"):
        cv.run_agreement(cv.european_grid(cv.GridSpec(strikes=(100.0,), taus=(1.0,), vols=(0.2,))))


def test_violated_asian_bound_is_detected(monkeypatch: pytest.MonkeyPatch) -> None:
    from crossprice.montecarlo import asian_mc_price

    def shifted(*args: object, **kwargs: object) -> MCResult:
        result = asian_mc_price(*args, **kwargs)  # type: ignore[arg-type]
        if kwargs.get("average") == "geometric":
            return result
        return replace(result, price=result.price - 1.0)

    monkeypatch.setattr(cv, "asian_mc_price", shifted)
    run = cv.run_agreement(cv.asian_cases(strikes=(100.0,)), cv.HarnessConfig())
    bounds = run.table[run.table["method"] == "asian_mc_arithmetic_control"]
    call = bounds[bounds["kind"] == "call"].iloc[0]
    assert call["status"] == "FAIL" and "lower bound violated" in call["reason"]
    put = bounds[bounds["kind"] == "put"].iloc[0]
    assert put["status"] == "pass"
    assert (run.table[run.table["method"] == "asian_mc_geometric"]["status"] == "pass").all()


def test_reference_failure_is_recorded_not_raised() -> None:
    cases = cv.american_grid(strikes=(100.0,), taus=(1.0,), carries=((0.10, 0.0),), vol=0.02)
    run = cv.run_agreement(cases, replace(cv.quick_config(), american_reference_steps=8))
    row = run.table[run.table["method"] == "crr_american"].iloc[0]
    assert row["outcome"] == "error" and row["status"] == "FAIL"
    assert row["reason"].startswith("expected agree, got error: reference failed: ValueError")
    assert np.isnan(row["reference_price"]) and np.isnan(row["price"])


def test_undeclared_degenerate_grid_row_is_logged_not_scored() -> None:
    contract = cv.Contract(100.0, 80.0, 0.25, 0.05, 0.10, 0.0, "put")
    case = cv.Case("eu-tiny-put", "european", "european", contract, (cv.MethodSpec("mc_plain"),))
    run = cv.run_agreement([case], cv.HarnessConfig())
    row = run.table.iloc[0]
    assert 0 < row["reference_price"] < 1e-5
    assert row["ci_status"] == "degenerate" and row["price"] == 0 and row["se"] == 0
    assert row["outcome"] == "degenerate" and row["status"] == "degenerate"
    assert row["reason"].startswith("undeclared degenerate sample, logged")
    assert np.isnan(row["z_score"]) and pd.isna(row["covered_95"])


def test_default_case_list_is_well_formed() -> None:
    cases = cv.default_cases()
    assert len(cases) == 72 + 24 + 6 + 17
    ids = [case.case_id for case in cases]
    assert len(set(ids)) == len(ids)
    groups = {case.group for case in cases}
    assert groups == {"european", "american", "asian", "stress"}
    assert all(case.note for case in cases if case.group == "stress")
    assert all(spec.method in cv.METHODS for case in cases for spec in case.methods)


def test_unknown_method_is_rejected() -> None:
    case = cv.Case(
        "bad",
        "european",
        "european",
        cv.Contract(100, 100, 1, 0.05, 0.2),
        (cv.MethodSpec("magic"),),
    )
    with pytest.raises(ValueError, match="unknown method"):
        cv.run_agreement([case])


def test_summary_counts_match_table(quick_run: cv.AgreementRun) -> None:
    summary = cv.summarize_methods(quick_run.table)
    assert summary["rows"].sum() == len(quick_run.table)
    assert (
        summary[["pass", "expected", "degenerate", "FAIL"]].sum(axis=1) == summary["rows"]
    ).all()
    family = cv.family_statistics(quick_run)
    scored = quick_run.table["z_score"].dropna()
    assert family["n_z_scores"] == len(scored) == family["n_intervals"]
    assert family["covered_95"] == int(quick_run.table["covered_95"].dropna().sum())
    assert family["status_counts"] == quick_run.table["status"].value_counts().to_dict()
    assert family["binomial_99_band"][0] <= family["binomial_99_band"][1] <= family["n_intervals"]
    markdown = cv.render_markdown(quick_run)
    assert "### Stress cases" in markdown and "stress-degenerate_mc" in markdown
    assert "Binomial(" in markdown and f"{quick_run.z_crit:.2f}" in markdown
    assert "asian_mc_arithmetic_control" in markdown


def test_crr_convergence_reproduces_committed_experiment() -> None:
    frame = cv.crr_convergence(cv.HarnessConfig())
    atm = frame[frame["contract"] == "atm"]
    orders = {
        (kind, parity): part["fitted_order"].iloc[0]
        for (kind, parity), part in atm.groupby(["kind", "parity"])
    }
    assert_allclose(orders[("call", "even")], 0.999169, atol=5e-7)
    assert_allclose(orders[("call", "odd")], 1.000651, atol=5e-7)
    assert_allclose(orders[("put", "even")], 0.999169, atol=5e-7)
    assert_allclose(orders[("put", "odd")], 1.000651, atol=5e-7)
    even_call = atm[(atm["kind"] == "call") & (atm["parity"] == "even")]
    assert_allclose(
        even_call["error"].to_numpy(),
        [-0.03989203145, -0.01997190994, -0.00999231233, -0.00499773090, -0.00249925730],
        rtol=0,
        atol=5e-12,
    )
    assert (atm[atm["parity"] == "even"]["error"] < 0).all()
    assert (atm[atm["parity"] == "odd"]["error"] > 0).all()
    off = frame[frame["contract"] == "off_strike"]
    assert off["fitted_order"].isna().all()
    off_call_even = off[(off["kind"] == "call") & (off["parity"] == "even")]
    assert (off_call_even["error"] > 0).any()
    print(f"CRR orders: {orders}")


def test_mc_convergence_reproduces_committed_experiment() -> None:
    frame = cv.mc_convergence(cv.HarnessConfig())
    call = frame[frame["kind"] == "call"]
    put = frame[frame["kind"] == "put"]
    assert_allclose(call["fitted_slope"].iloc[0], -0.490007, atol=5e-7)
    assert_allclose(put["fitted_slope"].iloc[0], -0.474954, atol=5e-7)
    assert_allclose(
        call["rms_error"].to_numpy(), [0.41147842, 0.22268654, 0.11295649, 0.05360798], atol=5e-9
    )
    assert (call["n_seeds"] == 128).all() and (call["first_seed"] == 1000).all()
    assert (np.diff(call["rms_error"]) < 0).all() and (np.diff(put["rms_error"]) < 0).all()
    ratio = frame["mean_se"] / frame["rms_error"]
    assert ((ratio > 0.8) & (ratio < 1.25)).all()
    print(frame[["kind", "n_paths", "rms_error", "mean_se", "mean_error", "fitted_slope"]])


def test_main_writes_reproducible_artifacts(tmp_path: Path) -> None:
    assert cv.main(["--quick", "--out", str(tmp_path)]) == 0
    names = {
        "agreement_table.csv",
        "agreement_table.md",
        "crossvalidation_run.json",
        "figures/crr_convergence.png",
        "figures/crr_convergence.csv",
        "figures/mc_convergence.png",
        "figures/mc_convergence.csv",
    }
    for name in names:
        assert (tmp_path / name).stat().st_size > 0
    table = pd.read_csv(tmp_path / "agreement_table.csv")
    assert tuple(table.columns) == cv.COLUMNS
    assert (table["status"] != "FAIL").all()
    text = (tmp_path / "crossvalidation_run.json").read_text()
    assert "NaN" not in text and "Infinity" not in text  # strict JSON: non-finite -> null
    record = json.loads(text)
    assert record["config"]["crr_steps"] == 200 and record["family"]["n_rows"] == len(table)
    assert set(record["crr_convergence"]["fitted_orders_atm"]) == {
        "call_even",
        "call_odd",
        "put_even",
        "put_odd",
    }
    assert set(record["mc_convergence"]["fitted_slopes"]) == {"call", "put"}
    assert set(record["artifacts"]) == names
    markdown = (tmp_path / "agreement_table.md").read_text()
    assert markdown.startswith("Rows:") and "### Per-method summary" in markdown
