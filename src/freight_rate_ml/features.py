from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


NUMERIC_FULL = [
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "distance",
    "weight_clean",
    "weight_was_missing",
    "weight_was_negative",
    "market_index",
    "market_index_was_missing",
    "quote_signal",
    "market_x_distance",
    "quote_x_distance",
    "quote_x_market",
    "month",
    "day",
    "dayofweek",
    "dayofyear",
    "week",
    "quarter",
    "is_weekend",
    "days_since_start",
    "dow_sin",
    "dow_cos",
    "annual_sin",
    "annual_cos",
    "lat_delta",
    "lon_delta",
    "abs_lat_delta",
    "abs_lon_delta",
    "haversine_miles",
    "distance_to_geo_ratio",
    "bearing_sin",
    "bearing_cos",
    "same_city",
    "pickup_frequency",
    "delivery_frequency",
    "route_frequency",
    "weight_per_mile",
]

NUMERIC_CHART = [c for c in NUMERIC_FULL if c not in ["market_index", "market_index_was_missing", "quote_signal", "market_x_distance", "quote_x_distance", "quote_x_market"]]
CATEGORICAL_FULL = ["pickup", "delivery", "equipment", "route", "city_pair"]
CATEGORICAL_CHART = CATEGORICAL_FULL


@dataclass
class FeatureSpec:
    numeric: list[str]
    categorical: list[str]


class FreightFeatureBuilder(BaseEstimator, TransformerMixin):
    def __init__(self, include_market: bool = True, negative_weight: str = "abs"):
        self.include_market = include_market
        self.negative_weight = negative_weight

    def fit(self, X: pd.DataFrame, y=None):
        frame = X.copy()
        frame["date"] = pd.to_datetime(frame["date"])
        self.start_date_ = frame["date"].min()
        weight = pd.to_numeric(frame["weight"], errors="coerce")
        if self.negative_weight == "missing":
            weight = weight.mask(weight < 0)
        elif self.negative_weight == "abs":
            weight = weight.abs()
        else:
            raise ValueError("negative_weight must be 'abs' or 'missing'")
        self.weight_median_ = float(weight.median())
        market = pd.to_numeric(frame.get("market_index", np.nan), errors="coerce")
        self.market_median_ = float(market.median()) if market.notna().any() else 1.0
        self.pickup_frequency_ = frame["pickup"].value_counts(normalize=True).to_dict()
        self.delivery_frequency_ = frame["delivery"].value_counts(normalize=True).to_dict()
        self.route_frequency_ = _route_key(frame).value_counts(normalize=True).to_dict()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = X.copy()
        frame["date"] = pd.to_datetime(frame["date"])
        result = pd.DataFrame(index=frame.index)
        for col in ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon", "distance"]:
            result[col] = pd.to_numeric(frame[col], errors="coerce")

        raw_weight = pd.to_numeric(frame["weight"], errors="coerce")
        result["weight_was_missing"] = raw_weight.isna().astype(float)
        result["weight_was_negative"] = (raw_weight < 0).fillna(False).astype(float)
        if self.negative_weight == "missing":
            clean_weight = raw_weight.mask(raw_weight < 0)
        else:
            clean_weight = raw_weight.abs()
        result["weight_clean"] = clean_weight.fillna(self.weight_median_)

        if self.include_market:
            raw_market = pd.to_numeric(frame["market_index"], errors="coerce")
            result["market_index_was_missing"] = raw_market.isna().astype(float)
            result["market_index"] = raw_market.fillna(self.market_median_)
            quote = pd.to_numeric(frame["quote_signal"], errors="coerce")
            result["quote_signal"] = quote
            result["market_x_distance"] = result["market_index"] * result["distance"]
            result["quote_x_distance"] = result["quote_signal"] * result["distance"]
            result["quote_x_market"] = result["quote_signal"] * result["market_index"]

        result["month"] = frame["date"].dt.month.astype(float)
        result["day"] = frame["date"].dt.day.astype(float)
        result["dayofweek"] = frame["date"].dt.dayofweek.astype(float)
        result["dayofyear"] = frame["date"].dt.dayofyear.astype(float)
        result["week"] = frame["date"].dt.isocalendar().week.astype(float)
        result["quarter"] = frame["date"].dt.quarter.astype(float)
        result["is_weekend"] = (frame["date"].dt.dayofweek >= 5).astype(float)
        result["days_since_start"] = (frame["date"] - self.start_date_).dt.days.astype(float)
        result["dow_sin"] = np.sin(2 * np.pi * result["dayofweek"] / 7)
        result["dow_cos"] = np.cos(2 * np.pi * result["dayofweek"] / 7)
        result["annual_sin"] = np.sin(2 * np.pi * result["dayofyear"] / 365.25)
        result["annual_cos"] = np.cos(2 * np.pi * result["dayofyear"] / 365.25)

        result["lat_delta"] = result["delivery_lat"] - result["pickup_lat"]
        result["lon_delta"] = result["delivery_lon"] - result["pickup_lon"]
        result["abs_lat_delta"] = result["lat_delta"].abs()
        result["abs_lon_delta"] = result["lon_delta"].abs()
        result["haversine_miles"] = _haversine(result["pickup_lat"], result["pickup_lon"], result["delivery_lat"], result["delivery_lon"])
        result["distance_to_geo_ratio"] = result["distance"] / result["haversine_miles"].replace(0, np.nan)
        result["distance_to_geo_ratio"] = result["distance_to_geo_ratio"].replace([np.inf, -np.inf], np.nan).fillna(1.0)
        bearing = np.arctan2(result["lon_delta"], result["lat_delta"])
        result["bearing_sin"] = np.sin(bearing)
        result["bearing_cos"] = np.cos(bearing)
        result["same_city"] = frame["pickup"].eq(frame["delivery"]).astype(float)
        route = _route_key(frame)
        result["pickup_frequency"] = frame["pickup"].map(self.pickup_frequency_).fillna(0.0)
        result["delivery_frequency"] = frame["delivery"].map(self.delivery_frequency_).fillna(0.0)
        result["route_frequency"] = route.map(self.route_frequency_).fillna(0.0)
        result["weight_per_mile"] = result["weight_clean"] / result["distance"].replace(0, np.nan)
        result["weight_per_mile"] = result["weight_per_mile"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

        result["pickup"] = frame["pickup"].astype(str)
        result["delivery"] = frame["delivery"].astype(str)
        result["equipment"] = frame["equipment"].astype(str)
        result["route"] = route
        result["city_pair"] = np.where(frame["pickup"].astype(str) < frame["delivery"].astype(str), frame["pickup"].astype(str) + "|" + frame["delivery"].astype(str), frame["delivery"].astype(str) + "|" + frame["pickup"].astype(str))
        columns = (NUMERIC_FULL if self.include_market else NUMERIC_CHART) + (CATEGORICAL_FULL if self.include_market else CATEGORICAL_CHART)
        return result[columns]

    @property
    def spec(self) -> FeatureSpec:
        return FeatureSpec(
            numeric=NUMERIC_FULL if self.include_market else NUMERIC_CHART,
            categorical=CATEGORICAL_FULL if self.include_market else CATEGORICAL_CHART,
        )


def _route_key(frame: pd.DataFrame) -> pd.Series:
    return frame["pickup"].astype(str) + " -> " + frame["delivery"].astype(str)


def _haversine(lat1, lon1, lat2, lon2):
    radius = 3958.7613
    phi1 = np.radians(lat1.astype(float))
    phi2 = np.radians(lat2.astype(float))
    d_phi = np.radians(lat2.astype(float) - lat1.astype(float))
    d_lam = np.radians(lon2.astype(float) - lon1.astype(float))
    a = np.sin(d_phi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(d_lam / 2) ** 2
    return radius * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def build_city_coordinate_lookup(*frames: pd.DataFrame) -> dict[str, dict[str, float]]:
    rows = []
    for frame in frames:
        rows.append(frame[["pickup", "pickup_lat", "pickup_lon"]].rename(columns={"pickup": "city", "pickup_lat": "lat", "pickup_lon": "lon"}))
        rows.append(frame[["delivery", "delivery_lat", "delivery_lon"]].rename(columns={"delivery": "city", "delivery_lat": "lat", "delivery_lon": "lon"}))
    coords = pd.concat(rows, ignore_index=True)
    grouped = coords.groupby("city")[["lat", "lon"]].agg(["min", "max", "mean"])
    inconsistent = grouped[((grouped[("lat", "max")] - grouped[("lat", "min")]).abs() > 1e-6) | ((grouped[("lon", "max")] - grouped[("lon", "min")]).abs() > 1e-6)]
    if not inconsistent.empty:
        raise ValueError(f"Inconsistent city coordinates: {list(inconsistent.index[:10])}")
    return {city: {"lat": float(row[("lat", "mean")]), "lon": float(row[("lon", "mean")])} for city, row in grouped.iterrows()}


def enrich_december(december: pd.DataFrame, lookup: dict[str, dict[str, float]]) -> pd.DataFrame:
    frame = december.copy()
    pickup = frame["pickup"].iloc[0]
    delivery = frame["delivery"].iloc[0]
    if pickup not in lookup or delivery not in lookup:
        raise ValueError("December cities are missing from coordinate lookup")
    frame["pickup_lat"] = lookup[pickup]["lat"]
    frame["pickup_lon"] = lookup[pickup]["lon"]
    frame["delivery_lat"] = lookup[delivery]["lat"]
    frame["delivery_lon"] = lookup[delivery]["lon"]
    return frame
