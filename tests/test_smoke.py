"""Smoke test: the package imports and reports a version."""

from __future__ import annotations

import crossprice


def test_package_imports_and_has_version() -> None:
    assert isinstance(crossprice.__version__, str)
    assert crossprice.__version__.count(".") == 2
