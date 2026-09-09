"""crossprice — cross-validated option pricing.

Three independent pricing methods, with a comparison harness planned next:

- ``analytic``     Black–Scholes–Merton, Greeks and geometric Asian prices.
- ``binomial``     Cox–Ross–Rubinstein tree, European and American.
- ``montecarlo``   European and Asian GBM simulation with standard errors and CIs.
- Planned: ``implied_vol`` and ``crossvalidate``.

This is a pricing and numerical-methods library.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
