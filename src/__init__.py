"""Reusable code for the code-mixed sentiment analysis project."""

import os
from pathlib import Path


# Some managed Windows environments do not allow writing to the normal user
# cache. Keeping Matplotlib's cache inside the project prevents noisy warnings.
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".matplotlib")
)
