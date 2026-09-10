"""crossprice — cross-validated option pricing.

Three independent pricing methods and the harness that makes them argue:

- ``analytic``      Black–Scholes–Merton, Greeks and geometric Asian prices.
- ``binomial``      Cox–Ross–Rubinstein tree, European and American.
- ``montecarlo``    European and Asian GBM simulation with standard errors and CIs.
- ``crossvalidate`` Agreement table, stress cases and convergence plots
                    (``python -m crossprice.crossvalidate``; needs pandas/matplotlib).
- ``implied_vol``   BSM inversion, safeguarded Newton and bracketed Brent fallback.
- ``surface``       Offline SPX snapshot cleaning, observed IV smile and sensitivities.

This is a pricing and numerical-methods library.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
