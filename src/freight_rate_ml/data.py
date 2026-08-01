from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

TRAIN_COLUMNS = [
    "load_id",
    "pickup",
    "delivery",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "distance",
    "equipment",
    "weight",
    "date",
    "market_index",
    "quote_signal",
    "posted_rate",
]
VALIDATION_COLUMNS = TRAIN_COLUMNS[:-1]
TEMPLATE_COLUMNS = ["load_id", "predicted_rate"]
DECEMBER_COLUMNS = ["pickup", "delivery", "distance", "equipment", "weight", "date", "predicted_rate"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory_files(root: Path, output: Path) -> dict[str, dict[str, object]]:
    names = [
        "CONTEXT.md",
        "PLAN.md",
        "PHASES.md",
        "freight-rate-ml-assessment.pdf",
        "README.md",
        "score.py",
        "requirements.txt",
        "train-test.csv",
        "validation.csv",
        "validation-predictions-template.csv",
        "december-chart-inputs.csv",
    ]
    manifest: dict[str, dict[str, object]] = {}
    for name in names:
        path = root / name
        if not path.is_file() and name == "README.md":
            path = root / "readme.md"
        if not path.is_file():
            raise FileNotFoundError(path)
        manifest[path.name] = {"size": path.stat().st_size, "sha256": sha256_file(path)}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def preserve_raw_inputs(root: Path, raw_dir: Path) -> dict[str, str]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    copied: dict[str, str] = {}
    for name in [
        "train-test.csv",
        "validation.csv",
        "validation-predictions-template.csv",
        "december-chart-inputs.csv",
    ]:
        source = root / name
        target = raw_dir / name
        shutil.copy2(source, target)
        if sha256_file(source) != sha256_file(target):
            raise ValueError(f"Raw copy hash mismatch for {name}")
        copied[name] = sha256_file(target)
    return copied


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def parse_dates(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    if result["date"].isna().any():
        raise ValueError("date contains unparsable values")
    return result


def validate_train(frame: pd.DataFrame) -> None:
    if list(frame.columns) != TRAIN_COLUMNS:
        raise ValueError("train-test.csv schema mismatch")
    if len(frame) != 48_000:
        raise ValueError("train-test.csv must contain 48,000 rows")
    _validate_common(frame, expect_target=True, id_prefix="TR")


def validate_validation(frame: pd.DataFrame) -> None:
    if list(frame.columns) != VALIDATION_COLUMNS:
        raise ValueError("validation.csv schema mismatch")
    if len(frame) != 12_000:
        raise ValueError("validation.csv must contain 12,000 rows")
    _validate_common(frame, expect_target=False, id_prefix="TE")


def validate_template(frame: pd.DataFrame) -> None:
    if list(frame.columns) != TEMPLATE_COLUMNS:
        raise ValueError("validation-predictions-template.csv schema mismatch")
    if len(frame) != 12_000:
        raise ValueError("template must contain 12,000 rows")
    ids = frame["load_id"].astype(str)
    expected = [f"TE-{i:06d}" for i in range(1, 12_001)]
    if ids.tolist() != expected:
        raise ValueError("template IDs are not the expected ordered TE IDs")


def validate_december_template(frame: pd.DataFrame) -> None:
    if list(frame.columns) != DECEMBER_COLUMNS:
        raise ValueError("december-chart-inputs.csv schema mismatch")
    if len(frame) != 31:
        raise ValueError("December template must contain 31 rows")
    parsed = pd.to_datetime(frame["date"], errors="coerce")
    expected = pd.date_range("2025-12-01", "2025-12-31", freq="D")
    if parsed.isna().any() or set(parsed) != set(expected):
        raise ValueError("December template must contain all December 2025 dates")
    fixed = {
        "pickup": "Lexington",
        "delivery": "Fort Wayne",
        "distance": 360.0,
        "equipment": "Dry Van",
        "weight": 32000.0,
    }
    for column, value in fixed.items():
        if column in ["distance", "weight"]:
            if not np.isclose(pd.to_numeric(frame[column]), value).all():
                raise ValueError(f"December {column} invariant failed")
        elif not frame[column].eq(value).all():
            raise ValueError(f"December {column} invariant failed")


def _validate_common(frame: pd.DataFrame, expect_target: bool, id_prefix: str) -> None:
    ids = frame["load_id"].astype(str)
    if ids.isna().any() or ids.duplicated().any():
        raise ValueError("missing or duplicate load_id")
    if not ids.str.startswith(f"{id_prefix}-").all():
        raise ValueError("unexpected load_id prefix")
    for column in ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon", "distance", "quote_signal"]:
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any() or not np.isfinite(values).all():
            raise ValueError(f"{column} contains invalid values")
    if (pd.to_numeric(frame["distance"], errors="coerce") <= 0).any():
        raise ValueError("distance must be positive")
    if expect_target:
        target = pd.to_numeric(frame["posted_rate"], errors="coerce")
        if target.isna().any() or not np.isfinite(target).all() or (target <= 0).any():
            raise ValueError("posted_rate must be finite and positive")


def load_all(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = parse_dates(read_csv(root / "train-test.csv"))
    validation = parse_dates(read_csv(root / "validation.csv"))
    template = read_csv(root / "validation-predictions-template.csv")
    december = parse_dates(read_csv(root / "december-chart-inputs.csv"))
    validate_train(train)
    validate_validation(validation)
    validate_template(template)
    validate_december_template(december)
    return train, validation, template, december


def validate_final_predictions(path: Path) -> None:
    frame = pd.read_csv(path)
    if list(frame.columns) != TEMPLATE_COLUMNS or len(frame) != 12_000:
        raise ValueError("validation_predictions.csv schema or row-count mismatch")
    validate_template(frame[["load_id", "predicted_rate"]].assign(predicted_rate=np.nan))
    values = pd.to_numeric(frame["predicted_rate"], errors="coerce")
    if values.isna().any() or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("validation predictions must be finite and positive")


def validate_final_december(path: Path) -> None:
    frame = pd.read_csv(path)
    validate_december_template(frame)
    values = pd.to_numeric(frame["predicted_rate"], errors="coerce")
    if values.isna().any() or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("December predictions must be finite and positive")
