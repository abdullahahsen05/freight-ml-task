from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


SEED = 42


@dataclass(frozen=True)
class ProjectPaths:
    root: Path

    @property
    def train(self) -> Path:
        return self.root / "train-test.csv"

    @property
    def validation(self) -> Path:
        return self.root / "validation.csv"

    @property
    def template(self) -> Path:
        return self.root / "validation-predictions-template.csv"

    @property
    def december_template(self) -> Path:
        return self.root / "december-chart-inputs.csv"

    @property
    def raw_dir(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def artifacts(self) -> Path:
        return self.root / "artifacts"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def validation_predictions(self) -> Path:
        return self.root / "validation_predictions.csv"

    @property
    def december_predictions(self) -> Path:
        return self.root / "december_chart_inputs.csv"


def paths(project_root: str | Path = ".") -> ProjectPaths:
    return ProjectPaths(Path(project_root).resolve())
