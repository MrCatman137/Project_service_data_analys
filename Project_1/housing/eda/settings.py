"""EDA report output configuration."""
from __future__ import annotations

import os
from pathlib import Path

from housing.config import BASE_DIR

OUTPUT_DIR = Path(os.getenv("EDA_OUTPUT", BASE_DIR / "reports" / "eda"))
