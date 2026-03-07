from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


COLOR_TO_CUT = {
    "green": "sirloin tender steak",
    "red": "new york strip steak",
}

FRESHNESS_TO_HOURS = {
    1: 0,
    2: 24,
    3: 48,
}


@dataclass(frozen=True)
class ProjectPaths:
    root: Path

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def output_dir(self) -> Path:
        return self.root / "output"

    @property
    def models_dir(self) -> Path:
        return self.output_dir / "models"

    @property
    def tables_dir(self) -> Path:
        return self.output_dir / "tables"

    @property
    def figures_dir(self) -> Path:
        return self.output_dir / "figures"

    @property
    def logs_dir(self) -> Path:
        return self.output_dir / "logs"

    @property
    def reports_dir(self) -> Path:
        return self.output_dir / "reports"
