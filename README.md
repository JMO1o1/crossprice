# crossprice

Cross-validated option pricing: three independent methods that must agree, and a harness that records
exactly where and why they stop agreeing.

> **Status: under construction.** The cross-method agreement table that this README will lead with does
> not exist yet. Nothing here should be cited until `docs/RESULTS.md` is written.

Build Steps 1-6 are implemented and tested. Step 7 is next; the harness, plots, implied-volatility
solver, data snapshot, and formal results remain unfinished.

## What this is

- **Analytic** — Black–Scholes–Merton closed form with continuous dividend yield, plus analytic Greeks.
- **Binomial** — Cox–Ross–Rubinstein tree, European and American (early exercise).
- **Monte Carlo** — risk-neutral GBM simulation; every price ships with a standard error and a 95% CI;
  antithetic and control-variate variance reduction with the *measured* reduction factor.
- **Asian options** — discrete arithmetic-average Monte Carlo, an exact geometric-average reference,
  and an independent-pilot geometric control; monitoring excludes time zero and includes expiry.
- **Implied volatility (planned)** — Newton–Raphson on analytic vega with a bisection fallback.
- **Cross-validation harness (planned)** — prices a parameter grid by every applicable method, asserts agreement to
  a stated tolerance, and logs the regions where agreement degrades instead of hiding them.

This is a pricing and numerical-methods project. Monte Carlo intervals quantify sampling error only;
they do not include model error or the difference between discrete and continuous Asian monitoring.

## Core API

Import deterministic benchmarks from `crossprice.analytic` (`bsm_price`, `bsm_greeks`,
`geometric_asian_price`), trees from `crossprice.binomial` (`crr_price`, `american=True` for early
exercise), and simulation from `crossprice.montecarlo` (`mc_price`, `asian_mc_price`).
All use `(spot, strike, tau, rate, vol, div=0.0, kind="call")` before method-specific keywords.
Analytic BSM inputs broadcast; tree and simulation contracts are scalar.

Monte Carlo requires `seed=` and returns `MCResult`, never a float. Inspect `price`, `se`, `ci`,
and `ci_status` together. `antithetic=True` uses pair-based standard errors;
`control_variate=True` uses an independent pilot. `variance_reduction` accounts for main and pilot
path cost and is a measured diagnostic, not a guaranteed improvement. Read function docstrings for
limit conventions, invalid-grid behavior, count restrictions and the degenerate-sample warning.

## Development

```bash
uv sync --group dev
uv run pytest
uv run ruff check . && uv run ruff format --check .
```

## License

MIT
