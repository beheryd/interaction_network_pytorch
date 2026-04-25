"""Smoke test: every dmfc.* subpackage imports cleanly.

If this fails, nothing else in the test suite is meaningful.
"""

import importlib

import pytest

SUBPACKAGES = [
    "dmfc",
    "dmfc.envs",
    "dmfc.models",
    "dmfc.training",
    "dmfc.analysis",
    "dmfc.rajalingham",
]


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackage_imports(name: str) -> None:
    importlib.import_module(name)


def test_core_deps_importable() -> None:
    for name in ("torch", "numpy", "scipy", "sklearn", "pandas", "matplotlib", "seaborn", "yaml"):
        importlib.import_module(name)
