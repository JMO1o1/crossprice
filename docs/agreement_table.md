Rows: 363 — pass 291, expected 67, degenerate 5, FAIL 0.

`pass`: agreement under the criterion below. `expected`: a declared rejection, degeneracy or unsupported instrument/method pair occurred as declared. `degenerate`: an undeclared Monte Carlo sample with no usable variation (no payoff observed, or a control variate exact on every path), logged with its mechanism and not counted as agreement. `FAIL`: anything else, including a declared failure that did not occur.

Criteria fixed before pricing: deterministic rows `|error| <= 0.005 + 0.002*|reference|` at 1000 tree steps (exact rows `1e-10 + 1e-12*|reference|`); stochastic rows `|z| <= 4.04` from a two-sided family-wise false-alarm budget of 0.01 over 187 declared stochastic rows; Monte Carlo uses 65536 paths (+4096 independent pilot paths when a control variate is fitted).

### Per-method summary

| instrument | method | criteria | rows | pass | expected | degenerate | FAIL | max_abs_error | max_abs_rel_error | max_abs_z | covered_95 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| european | crr | exact/tolerance | 89 | 88 | 1 | 0 | 0 | 0.015 | 0.8 | — | n/a |
| european | mc_plain | exact/z_score | 88 | 86 | 1 | 1 | 0 | 0.22 | 5.67 | 2.39 | 80/83 |
| european | mc_antithetic_control | exact/z_score | 87 | 80 | 3 | 4 | 0 | 0.0246 | 1 | 3.07 | 72/77 |
| american | bsm | n/a | 25 | 0 | 25 | 0 | 0 | — | — | — | n/a |
| american | crr_american | tolerance | 25 | 25 | 0 | 0 | 0 | 0.00169 | 0.00175 | — | n/a |
| american | mc_plain | n/a | 25 | 0 | 25 | 0 | 0 | — | — | — | n/a |
| asian | bsm | n/a | 6 | 0 | 6 | 0 | 0 | — | — | — | n/a |
| asian | crr | n/a | 6 | 0 | 6 | 0 | 0 | — | — | — | n/a |
| asian | asian_mc_geometric | z_score | 6 | 6 | 0 | 0 | 0 | 0.0337 | 0.0087 | 1.45 | 6/6 |
| asian | asian_mc_arithmetic_control | lower_bound/upper_bound | 6 | 6 | 0 | 0 | 0 | — | — | — | n/a |

95% intervals covering the reference: 158/166; the Binomial(166, 0.95) 99% band is [150, 164] (inside). z-scores: mean -0.087, SD 0.961, max |z| 3.07.

### Stress cases

| case_id | method | settings | reference_price | price | error | rel_error | se | z_score | status | reason |
|---|---|---|---|---|---|---|---|---|---|---|
| stress-expiry_otm_call | crr | steps=1000 | 0 | 0 | 0 | — | — | — | pass |  |
| stress-expiry_otm_call | mc_plain | n_paths=65536 | 0 | 0 | 0 | — | 0 | — | pass |  |
| stress-expiry_otm_call | mc_antithetic_control | n_paths=65536+4096 pilot; antithetic; control | 0 | 0 | 0 | — | 0 | — | pass |  |
| stress-expiry_itm_put | crr | steps=1000 | 10 | 10 | 0 | 0 | — | — | pass |  |
| stress-expiry_itm_put | mc_plain | n_paths=65536 | 10 | 10 | 0 | 0 | 0 | — | pass |  |
| stress-expiry_itm_put | mc_antithetic_control | n_paths=65536+4096 pilot; antithetic; control | 10 | 10 | 0 | 0 | 0 | — | pass |  |
| stress-short_expiry_atm_call | crr | steps=1000 | 0.424486 | 0.424382 | -0.000104 | -0.000246 | — | — | pass |  |
| stress-short_expiry_atm_call | mc_plain | n_paths=65536 | 0.424486 | 0.427023 | 0.00254 | 0.00598 | 0.00243 | 1.04 | pass |  |
| stress-short_expiry_atm_call | mc_antithetic_control | n_paths=65536+4096 pilot; antithetic; control | 0.424486 | 0.423785 | -0.000701 | -0.00165 | 0.000626 | -1.12 | pass |  |
| stress-short_expiry_otm_call | crr | steps=1000 | 0.000780458 | 0.000776741 | -3.72e-06 | -0.00476 | — | — | pass | passes on atol: |rel_error|=0.00476 exceeds rtol=0.002 because the reference price is small |
| stress-short_expiry_otm_call | mc_plain | n_paths=65536 | 0.000780458 | 0.000796108 | 1.57e-05 | 0.0201 | 9.09e-05 | 0.17 | pass |  |
| stress-short_expiry_otm_call | mc_antithetic_control | n_paths=65536+4096 pilot; antithetic; control | 0.000780458 | 0.000730682 | -4.98e-05 | -0.0638 | 7.39e-05 | -0.67 | pass |  |
| stress-zero_vol_call | crr | steps=1000 | 7.65307 | 7.65307 | 0 | 0 | — | — | pass |  |
| stress-zero_vol_call | mc_plain | n_paths=65536 | 7.65307 | 7.65307 | 0 | 0 | 0 | — | pass |  |
| stress-zero_vol_call | mc_antithetic_control | n_paths=65536+4096 pilot; antithetic; control | 7.65307 | 7.65307 | 0 | 0 | 0 | — | pass |  |
| stress-low_vol_coarse_tree | crr | steps=16 | 9.51626 | — | — | — | — | — | expected | ValueError: CRR probability requires abs(rate-div)*dt < vol*sqrt(dt); increase steps |
| stress-low_vol_coarse_tree | crr | steps=1000 | 9.51626 | 9.51626 | -1.71e-08 | -1.79e-09 | — | — | pass |  |
| stress-low_vol_coarse_tree | mc_plain | n_paths=65536 | 9.51626 | 9.50617 | -0.0101 | -0.00106 | 0.00781 | -1.29 | pass |  |
| stress-low_vol_coarse_tree | mc_antithetic_control | n_paths=65536+4096 pilot; antithetic; control | 9.51626 | 9.51626 | -1.02e-07 | -1.07e-08 | 2.7e-17 | — | expected | SE=2.7e-17 is below the floating-point resolution floor (1e-12 x price scale): the control variate reproduced every sampled payoff exactly, so z is undefined; the residual error=-1.02e-07 is the unsampled region where the payoff is not linear in the control |
| stress-deep_itm_call | crr | steps=1000 | 80.9754 | 80.9754 | -3.34e-12 | -4.12e-14 | — | — | pass |  |
| stress-deep_itm_call | mc_plain | n_paths=65536 | 80.9754 | 81.0277 | 0.0523 | 0.000646 | 0.079 | 0.66 | pass |  |
| stress-deep_itm_call | mc_antithetic_control | n_paths=65536+4096 pilot; antithetic; control | 80.9754 | 80.9754 | 0 | 0 | 1.16e-18 | — | expected | SE=1.2e-18 is below the floating-point resolution floor (1e-12 x price scale): the control variate reproduced every sampled payoff exactly, so z is undefined; the residual error=0 is the unsampled region where the payoff is not linear in the control |
| stress-deep_itm_put | crr | steps=1000 | 185.369 | 185.369 | -3.83e-08 | -2.06e-10 | — | — | pass |  |
| stress-deep_itm_put | mc_plain | n_paths=65536 | 185.369 | 185.312 | -0.057 | -0.000307 | 0.0789 | -0.72 | pass |  |
| stress-deep_itm_put | mc_antithetic_control | n_paths=65536+4096 pilot; antithetic; control | 185.369 | 185.369 | -4.75e-07 | -2.56e-09 | 1.48e-16 | — | expected | SE=1.5e-16 is below the floating-point resolution floor (1e-12 x price scale): the control variate reproduced every sampled payoff exactly, so z is undefined; the residual error=-4.75e-07 is the unsampled region where the payoff is not linear in the control |
| stress-deep_otm_call | crr | steps=1000 | 0.00325946 | 0.00321477 | -4.47e-05 | -0.0137 | — | — | pass | passes on atol: |rel_error|=0.0137 exceeds rtol=0.002 because the reference price is small |
| stress-deep_otm_call | mc_plain | n_paths=65536 | 0.00325946 | 0.00223365 | -0.00103 | -0.315 | 0.00066 | -1.55 | pass |  |
| stress-deep_otm_call | mc_antithetic_control | n_paths=65536+4096 pilot; antithetic; control | 0.00325946 | 0.00412556 | 0.000866 | 0.266 | 0.00108 | 0.80 | pass |  |
| stress-deep_otm_put | crr | steps=1000 | 0.000498676 | 0.00049214 | -6.54e-06 | -0.0131 | — | — | pass | passes on atol: |rel_error|=0.0131 exceeds rtol=0.002 because the reference price is small |
| stress-deep_otm_put | mc_plain | n_paths=65536 | 0.000498676 | 0.000660861 | 0.000162 | 0.325 | 0.000203 | 0.80 | pass |  |
| stress-deep_otm_put | mc_antithetic_control | n_paths=65536+4096 pilot; antithetic; control | 0.000498676 | 0.000497344 | -1.33e-06 | -0.00267 | 0.000159 | -0.01 | pass |  |
| stress-degenerate_mc | crr | steps=1000 | 3.05867e-30 | 6.10768e-31 | -2.45e-30 | -0.8 | — | — | pass | passes on atol: |rel_error|=0.8 exceeds rtol=0.002 because the reference price is small |
| stress-degenerate_mc | mc_plain | n_paths=1000 | 3.05867e-30 | 0 | -3.06e-30 | -1 | 0 | — | expected | no sample variation: SE=0, z undefined; a zero-width interval is not evidence of certainty (reference=3.05867e-30) |
| stress-negative_rate_call | crr | steps=1000 | 7.07602 | 7.074 | -0.00202 | -0.000286 | — | — | pass |  |
| stress-negative_rate_call | mc_plain | n_paths=65536 | 7.07602 | 7.12126 | 0.0452 | 0.00639 | 0.0492 | 0.92 | pass |  |
| stress-negative_rate_call | mc_antithetic_control | n_paths=65536+4096 pilot; antithetic; control | 7.07602 | 7.06105 | -0.015 | -0.00212 | 0.0109 | -1.37 | pass |  |
| stress-negative_rate_put | crr | steps=1000 | 9.09615 | 9.09413 | -0.00202 | -0.000222 | — | — | pass |  |
| stress-negative_rate_put | mc_plain | n_paths=65536 | 9.09615 | 9.14752 | 0.0514 | 0.00565 | 0.0435 | 1.18 | pass |  |
| stress-negative_rate_put | mc_antithetic_control | n_paths=65536+4096 pilot; antithetic; control | 9.09615 | 9.10042 | 0.00427 | 0.00047 | 0.0101 | 0.42 | pass |  |
| stress-negative_carry_call | crr | steps=1000 | 5.51807 | 5.51609 | -0.00198 | -0.000359 | — | — | pass |  |
| stress-negative_carry_call | mc_plain | n_paths=65536 | 5.51807 | 5.48345 | -0.0346 | -0.00627 | 0.042 | -0.83 | pass |  |
| stress-negative_carry_call | mc_antithetic_control | n_paths=65536+4096 pilot; antithetic; control | 5.51807 | 5.52458 | 0.00652 | 0.00118 | 0.00785 | 0.83 | pass |  |
| stress-negative_dividend_put | crr | steps=1000 | 5.68612 | 5.68408 | -0.00204 | -0.000359 | — | — | pass |  |
| stress-negative_dividend_put | mc_plain | n_paths=65536 | 5.68612 | 5.70566 | 0.0195 | 0.00344 | 0.0346 | 0.56 | pass |  |
| stress-negative_dividend_put | mc_antithetic_control | n_paths=65536+4096 pilot; antithetic; control | 5.68612 | 5.69623 | 0.0101 | 0.00178 | 0.0109 | 0.93 | pass |  |
| stress-high_vol_long_call | crr | steps=1000 | 76.8231 | 76.8081 | -0.015 | -0.000195 | — | — | pass |  |
| stress-high_vol_long_call | mc_plain | n_paths=65536 | 76.8231 | 76.8335 | 0.0105 | 0.000136 | 3.1 | 0.00 | pass |  |
| stress-high_vol_long_call | mc_antithetic_control | n_paths=65536+4096 pilot; antithetic; control | 76.8231 | 76.8297 | 0.00664 | 8.65e-05 | 0.0698 | 0.10 | pass |  |
| stress-american_negative_rate_call | bsm |  | — | — | — | — | — | — | expected | BSM prices European exercise only; it is not an American reference |
| stress-american_negative_rate_call | crr_american | steps=1000; american | 20.2708 | 20.271 | 0.000218 | 1.07e-05 | — | — | pass |  |
| stress-american_negative_rate_call | mc_plain |  | — | — | — | — | — | — | expected | no early-exercise Monte Carlo (Longstaff-Schwartz) is implemented |
