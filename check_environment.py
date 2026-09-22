"""Check that the web application packages are installed."""

from __future__ import annotations

import importlib.util
import sys


REQUIRED_PACKAGES = (
    "pandas",
    "numpy",
    "sklearn",
    "scipy",
    "gensim",
    "joblib",
    "matplotlib",
    "torch",
    "jupyter",
    "streamlit",
)


def main() -> None:
    """Exit with an error when one or more packages are unavailable."""

    missing = [
        package
        for package in REQUIRED_PACKAGES
        if importlib.util.find_spec(package) is None
    ]
    if missing:
        print("Missing packages: " + ", ".join(missing))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
