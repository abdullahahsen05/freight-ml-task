from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TemporalFold:
    name: str
    train_end: str
    valid_start: str
    valid_end: str


def temporal_folds() -> list[TemporalFold]:
    return [
        TemporalFold("validate_july", "2025-06-30", "2025-07-01", "2025-07-31"),
        TemporalFold("validate_august", "2025-07-31", "2025-08-01", "2025-08-31"),
        TemporalFold("validate_september", "2025-08-31", "2025-09-01", "2025-09-30"),
        TemporalFold("validate_october", "2025-09-30", "2025-10-01", "2025-10-31"),
        TemporalFold("validate_sep_oct", "2025-08-31", "2025-09-01", "2025-10-31"),
    ]


def split_fold(frame: pd.DataFrame, fold: TemporalFold) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_end = pd.Timestamp(fold.train_end)
    valid_start = pd.Timestamp(fold.valid_start)
    valid_end = pd.Timestamp(fold.valid_end)
    train = frame[frame["date"] <= train_end].copy()
    valid = frame[(frame["date"] >= valid_start) & (frame["date"] <= valid_end)].copy()
    if train.empty or valid.empty:
        raise ValueError(f"{fold.name} produced an empty split")
    if train["date"].max() >= valid["date"].min():
        raise ValueError(f"{fold.name} leaks chronology")
    return train, valid


def fold_manifest(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fold in temporal_folds():
        train, valid = split_fold(frame, fold)
        rows.append(
            {
                "fold": fold.name,
                "train_start": train["date"].min().date().isoformat(),
                "train_end": train["date"].max().date().isoformat(),
                "valid_start": valid["date"].min().date().isoformat(),
                "valid_end": valid["date"].max().date().isoformat(),
                "train_rows": len(train),
                "valid_rows": len(valid),
            }
        )
    return pd.DataFrame(rows)
