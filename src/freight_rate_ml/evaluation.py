from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score


def prediction_floor(y_train: pd.Series) -> float:
    return max(1.0, float(y_train.quantile(0.005) * 0.5))


def apply_prediction_safety(pred: np.ndarray, y_train: pd.Series) -> np.ndarray:
    floor = prediction_floor(y_train)
    values = np.asarray(pred, dtype=float)
    values = np.where(np.isfinite(values), values, floor)
    return np.maximum(values, floor)


def metrics(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.sum(np.abs(y_true))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "r2": float(r2_score(y_true, y_pred)),
        "median_ae": float(median_absolute_error(y_true, y_pred)),
        "wape": float(np.sum(np.abs(y_true - y_pred)) / denom) if denom else float("nan"),
    }


def subgroup_metrics(frame: pd.DataFrame, y_true, y_pred, train_frame: pd.DataFrame) -> dict[str, float]:
    route_train = set(train_frame["pickup"].astype(str) + " -> " + train_frame["delivery"].astype(str))
    route_valid = frame["pickup"].astype(str) + " -> " + frame["delivery"].astype(str)
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    result: dict[str, float] = {}
    extreme_cut = float(train_frame["posted_rate"].quantile(0.99))
    masks = {
        "seen_route": route_valid.isin(route_train).to_numpy(),
        "unseen_route": (~route_valid.isin(route_train)).to_numpy(),
        "normal_target": y_true < extreme_cut,
        "extreme_target": y_true >= extreme_cut,
    }
    for name, mask in masks.items():
        if mask.any():
            result[f"{name}_mae"] = float(mean_absolute_error(y_true[mask], y_pred[mask]))
            result[f"{name}_rows"] = int(mask.sum())
        else:
            result[f"{name}_mae"] = float("nan")
            result[f"{name}_rows"] = 0
    return result
