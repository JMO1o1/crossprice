"""crossprice — cross-validated option pricing.

Three independent pricing methods for vanilla options that must agree, and a
harness that records exactly where and why they stop agreeing:

- ``analytic``     Black–Scholes–Merton closed form and analytic Greeks.
- ``binomial``     Cox–Ross–Rubinstein tree, European and American.
- ``montecarlo``   Risk-neutral GBM simulation with standard errors and CIs.
- ``implied_vol``  Newton–Raphson with bisection fallback.
- ``crossvalidate`` The harness that makes the methods argue.

This is a pricing and numerical-methods library. It contains no trading logic.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
