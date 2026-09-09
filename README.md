# crossprice

Cross-validated option pricing: three independent methods that must agree, and a harness that records
exactly where and why they stop agreeing.

> **Status: under construction.** The cross-method agreement table that this README will lead with does
> not exist yet. Nothing here should be cited until `docs/RESULTS.md` is written.

## What this is

- **Analytic** — Black–Scholes–Merton closed form with continuous dividend yield, plus analytic Greeks.
- **Binomial** — Cox–Ross–Rubinstein tree, European and American (early exercise).
- **Monte Carlo** — risk-neutral GBM simulation; every price ships with a standard error and a 95% CI;
  antithetic and control-variate variance reduction with the *measured* reduction factor.
- **Implied volatility** — Newton–Raphson on analytic vega with a bisection fallback.
- **Cross-validation harness** — prices a parameter grid by every applicable method, asserts agreement to
  a stated tolerance, and logs the regions where agreement degrades instead of hiding them.

This is a pricing and numerical-methods project. It contains no trading logic, signals, or strategies.

## Development

```bash
uv sync --group dev
uv run pytest
uv run ruff check . && uv run ruff format --check .
```

## License

MIT
