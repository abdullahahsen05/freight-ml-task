from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from .config import SEED
from .evaluation import apply_prediction_safety
from .features import FreightFeatureBuilder


class GlobalMedianModel:
    def fit(self, X, y):
        self.value_ = float(np.median(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.value_, dtype=float)


class DistanceLinearBaseline:
    def fit(self, X, y):
        from sklearn.linear_model import LinearRegression

        self.model_ = LinearRegression().fit(pd.to_numeric(X["distance"]).to_numpy().reshape(-1, 1), y)
        return self

    def predict(self, X):
        return self.model_.predict(pd.to_numeric(X["distance"]).to_numpy().reshape(-1, 1))


class EquipmentRatePerMileBaseline:
    def fit(self, X, y):
        frame = X.copy()
        frame["rpm"] = np.asarray(y, dtype=float) / pd.to_numeric(frame["distance"], errors="coerce")
        self.global_rpm_ = float(frame["rpm"].median())
        self.rpm_ = frame.groupby("equipment")["rpm"].median().to_dict()
        return self

    def predict(self, X):
        rpm = X["equipment"].map(self.rpm_).fillna(self.global_rpm_).to_numpy(dtype=float)
        return rpm * pd.to_numeric(X["distance"], errors="coerce").to_numpy(dtype=float)


@dataclass
class CandidateSpec:
    name: str
    family: str
    include_market: bool
    negative_weight: str = "abs"
    log_target: bool = False


class CandidateModel:
    def __init__(self, spec: CandidateSpec):
        self.spec = spec

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.y_train_ = pd.Series(y).astype(float).reset_index(drop=True)
        if self.spec.family == "global_median":
            self.estimator_ = GlobalMedianModel().fit(X, y)
            return self
        if self.spec.family == "distance_linear":
            self.estimator_ = DistanceLinearBaseline().fit(X, y)
            return self
        if self.spec.family == "equipment_rpm":
            self.estimator_ = EquipmentRatePerMileBaseline().fit(X, y)
            return self

        self.feature_builder_ = FreightFeatureBuilder(
            include_market=self.spec.include_market,
            negative_weight=self.spec.negative_weight,
        ).fit(X, y)
        transformed = self.feature_builder_.transform(X)
        spec = self.feature_builder_.spec
        if self.spec.family == "ridge":
            preprocessor = ColumnTransformer(
                [
                    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), spec.numeric),
                    ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=5), spec.categorical),
                ]
            )
            regressor = Ridge(alpha=20.0, random_state=SEED)
        elif self.spec.family == "hgb":
            preprocessor = ColumnTransformer(
                [
                    ("num", SimpleImputer(strategy="median"), spec.numeric),
                    ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-2), spec.categorical),
                ],
                verbose_feature_names_out=False,
            )
            regressor = HistGradientBoostingRegressor(
                loss="absolute_error",
                learning_rate=0.06,
                max_iter=140,
                max_leaf_nodes=31,
                l2_regularization=0.04,
                random_state=SEED,
            )
        elif self.spec.family == "extra_trees":
            preprocessor = ColumnTransformer(
                [
                    ("num", SimpleImputer(strategy="median"), spec.numeric),
                    ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-2), spec.categorical),
                ],
                verbose_feature_names_out=False,
            )
            regressor = ExtraTreesRegressor(
                n_estimators=90,
                min_samples_leaf=3,
                max_features=0.75,
                random_state=SEED,
                n_jobs=-1,
            )
        elif self.spec.family == "random_forest":
            preprocessor = ColumnTransformer(
                [
                    ("num", SimpleImputer(strategy="median"), spec.numeric),
                    ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-2), spec.categorical),
                ],
                verbose_feature_names_out=False,
            )
            regressor = RandomForestRegressor(
                n_estimators=80,
                min_samples_leaf=4,
                max_features=0.75,
                random_state=SEED,
                n_jobs=-1,
            )
        else:
            raise ValueError(f"Unknown family: {self.spec.family}")
        target = np.log1p(y) if self.spec.log_target else np.asarray(y, dtype=float)
        self.estimator_ = Pipeline([("preprocessor", preprocessor), ("regressor", regressor)])
        self.estimator_.fit(transformed, target)
        return self

    def predict(self, X: pd.DataFrame):
        if self.spec.family in {"global_median", "distance_linear", "equipment_rpm"}:
            raw = self.estimator_.predict(X)
        else:
            transformed = self.feature_builder_.transform(X)
            raw = self.estimator_.predict(transformed)
            if self.spec.log_target:
                raw = np.expm1(raw)
        return apply_prediction_safety(raw, self.y_train_)

    def save(self, path):
        dump(self, path)


def full_candidate_specs() -> list[CandidateSpec]:
    return [
        CandidateSpec("global_median", "global_median", True),
        CandidateSpec("distance_linear", "distance_linear", True),
        CandidateSpec("equipment_median_rpm", "equipment_rpm", True),
        CandidateSpec("ridge_log_full_abs_weight", "ridge", True, "abs", True),
        CandidateSpec("hgb_mae_full_abs_weight", "hgb", True, "abs", False),
        CandidateSpec("hgb_mae_full_missing_bad_weight", "hgb", True, "missing", False),
        CandidateSpec("extra_trees_full_abs_weight", "extra_trees", True, "abs", False),
    ]


def december_candidate_specs() -> list[CandidateSpec]:
    return [
        CandidateSpec("december_equipment_median_rpm", "equipment_rpm", False),
        CandidateSpec("december_ridge_log_abs_weight", "ridge", False, "abs", True),
        CandidateSpec("december_hgb_mae_abs_weight", "hgb", False, "abs", False),
        CandidateSpec("december_extra_trees_abs_weight", "extra_trees", False, "abs", False),
    ]


def timed_fit_predict(model: CandidateModel, train: pd.DataFrame, valid: pd.DataFrame):
    y_train = train["posted_rate"].astype(float)
    start = time.perf_counter()
    model.fit(train, y_train)
    fit_seconds = time.perf_counter() - start
    start = time.perf_counter()
    pred = model.predict(valid)
    predict_seconds = time.perf_counter() - start
    return pred, fit_seconds, predict_seconds
