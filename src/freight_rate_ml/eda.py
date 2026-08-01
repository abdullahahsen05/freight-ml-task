from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def write_data_audit(train: pd.DataFrame, validation: pd.DataFrame, template: pd.DataFrame, december: pd.DataFrame, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    audit = {
        "train-test.csv": _profile(train, "posted_rate"),
        "validation.csv": _profile(validation, None),
        "validation-predictions-template.csv": _profile(template, None),
        "december-chart-inputs.csv": _profile(december, None),
    }
    train_routes = set(train["pickup"].astype(str) + " -> " + train["delivery"].astype(str))
    valid_routes = set(validation["pickup"].astype(str) + " -> " + validation["delivery"].astype(str))
    train_cities = set(train["pickup"]).union(set(train["delivery"]))
    valid_cities = set(validation["pickup"]).union(set(validation["delivery"]))
    audit["train_validation_overlap"] = {
        "unseen_validation_cities": sorted(valid_cities - train_cities),
        "unseen_validation_routes": len(valid_routes - train_routes),
        "train_routes": len(train_routes),
        "validation_routes": len(valid_routes),
        "template_matches_validation_order": template["load_id"].tolist() == validation["load_id"].tolist(),
    }
    audit["market_index_shift"] = {
        "train_mean": float(train["market_index"].mean()),
        "validation_mean": float(validation["market_index"].mean()),
        "difference": float(validation["market_index"].mean() - train["market_index"].mean()),
    }
    (output_dir / "data_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    pd.DataFrame(_flatten_profile(audit)).to_csv(output_dir / "data_profile.csv", index=False)
    drift = _drift_table(train, validation)
    drift.to_csv(output_dir / "train_validation_drift.csv", index=False)
    _write_findings(audit, drift, output_dir / "data_quality_findings.md")
    _charts(train, validation, output_dir)
    return audit


def _profile(frame: pd.DataFrame, target: str | None) -> dict:
    date_range = None
    if "date" in frame:
        dates = pd.to_datetime(frame["date"], errors="coerce")
        date_range = [dates.min().date().isoformat(), dates.max().date().isoformat()]
    result = {
        "shape": [int(frame.shape[0]), int(frame.shape[1])],
        "columns": list(frame.columns),
        "missing": {c: int(v) for c, v in frame.isna().sum().items()},
        "duplicate_rows": int(frame.duplicated().sum()),
        "date_range": date_range,
    }
    if "load_id" in frame:
        result["duplicate_load_ids"] = int(frame["load_id"].duplicated().sum())
    if "weight" in frame:
        weight = pd.to_numeric(frame["weight"], errors="coerce")
        result["negative_weights"] = int((weight < 0).sum())
        result["min_weight"] = float(weight.min())
    if "equipment" in frame:
        result["equipment_counts"] = {str(k): int(v) for k, v in frame["equipment"].value_counts().items()}
    if "pickup" in frame and "delivery" in frame:
        result["pickup_cities"] = int(frame["pickup"].nunique())
        result["delivery_cities"] = int(frame["delivery"].nunique())
        result["routes"] = int((frame["pickup"].astype(str) + " -> " + frame["delivery"].astype(str)).nunique())
    if target:
        y = pd.to_numeric(frame[target], errors="coerce")
        result["target_summary"] = {k: float(v) for k, v in y.describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).items()}
        result["rate_per_mile_summary"] = {k: float(v) for k, v in (y / frame["distance"]).describe(percentiles=[0.01, 0.5, 0.99]).items()}
    return result


def _flatten_profile(audit: dict) -> list[dict]:
    rows = []
    for dataset, values in audit.items():
        if isinstance(values, dict) and "shape" in values:
            rows.append({"dataset": dataset, "metric": "rows", "value": values["shape"][0]})
            rows.append({"dataset": dataset, "metric": "columns", "value": values["shape"][1]})
            for column, count in values.get("missing", {}).items():
                rows.append({"dataset": dataset, "metric": f"missing_{column}", "value": count})
            for key in ["negative_weights", "duplicate_rows", "duplicate_load_ids", "pickup_cities", "delivery_cities", "routes"]:
                if key in values:
                    rows.append({"dataset": dataset, "metric": key, "value": values[key]})
    return rows


def _drift_table(train: pd.DataFrame, validation: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon", "distance", "weight", "market_index", "quote_signal"]:
        a = pd.to_numeric(train[column], errors="coerce")
        b = pd.to_numeric(validation[column], errors="coerce")
        rows.append(
            {
                "column": column,
                "train_mean": a.mean(),
                "validation_mean": b.mean(),
                "mean_difference": b.mean() - a.mean(),
                "train_missing_rate": a.isna().mean(),
                "validation_missing_rate": b.isna().mean(),
                "train_p50": a.median(),
                "validation_p50": b.median(),
            }
        )
    return pd.DataFrame(rows)


def _write_findings(audit: dict, drift: pd.DataFrame, path: Path) -> None:
    overlap = audit["train_validation_overlap"]
    train = audit["train-test.csv"]
    validation = audit["validation.csv"]
    lines = [
        "# Data Quality Findings",
        "",
        f"- Training data has {train['shape'][0]:,} rows and validation has {validation['shape'][0]:,} rows.",
        f"- Training dates span {train['date_range'][0]} to {train['date_range'][1]}; validation dates span {validation['date_range'][0]} to {validation['date_range'][1]}.",
        f"- Missing values: training weight={train['missing']['weight']}, training market_index={train['missing']['market_index']}, validation weight={validation['missing']['weight']}, validation market_index={validation['missing']['market_index']}.",
        f"- Negative weights are present: training={train['negative_weights']}, validation={validation['negative_weights']}. The pipeline benchmarks absolute-value repair and missing-value imputation with flags.",
        f"- Validation contains unseen cities: {', '.join(overlap['unseen_validation_cities']) or 'none'}; exact unseen validation routes={overlap['unseen_validation_routes']}.",
        f"- Market index shifts from mean {audit['market_index_shift']['train_mean']:.3f} in training to {audit['market_index_shift']['validation_mean']:.3f} in validation.",
        "- Target rates are right-skewed with rare high rate-per-mile rows, so temporal MAE is the primary selection metric and RMSE/tail metrics are reported as diagnostics.",
        "- December chart rows lack market_index and quote_signal. A separate chart-compatible model excludes those fields and is backtested under the same constraint.",
        "- Root CSVs are preserved under data/raw with hash checks; pipeline commands read the original root files and never modify the supplied inputs.",
        "- City-to-coordinate consistency is validated before December enrichment.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _charts(train: pd.DataFrame, validation: pd.DataFrame, output_dir: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    y = train["posted_rate"].astype(float)
    _hist(y, output_dir / "target_distribution.png", "Posted Rate Distribution", "posted_rate")
    _hist(np.log1p(y), output_dir / "log_target_distribution.png", "Log Posted Rate Distribution", "log1p(posted_rate)")
    fig, ax = plt.subplots(figsize=(8, 5), dpi=140)
    sample = train.sample(min(8000, len(train)), random_state=42)
    ax.scatter(sample["distance"], sample["posted_rate"], s=6, alpha=0.22)
    ax.set_title("Posted Rate vs Distance")
    ax.set_xlabel("Distance (miles)")
    ax.set_ylabel("Posted rate ($)")
    fig.tight_layout()
    fig.savefig(output_dir / "rate_vs_distance.png")
    plt.close(fig)
    _hist(y / train["distance"], output_dir / "rate_per_mile_distribution.png", "Rate Per Mile Distribution", "posted_rate / distance")
    fig, ax = plt.subplots(figsize=(8, 5), dpi=140)
    train.assign(rate_per_mile=y / train["distance"]).boxplot(column="rate_per_mile", by="equipment", ax=ax, showfliers=False)
    ax.set_title("Rate Per Mile by Equipment")
    ax.set_xlabel("Equipment")
    ax.set_ylabel("Rate per mile")
    fig.suptitle("")
    fig.tight_layout()
    fig.savefig(output_dir / "rpm_by_equipment.png")
    plt.close(fig)
    weekly = train.set_index("date").resample("W").agg(posted_rate=("posted_rate", "median"), market_index=("market_index", "median"))
    fig, ax = plt.subplots(figsize=(9, 5), dpi=140)
    weekly["posted_rate"].plot(ax=ax, label="Median posted rate")
    ax2 = ax.twinx()
    weekly["market_index"].plot(ax=ax2, color="#C3562A", label="Median market index")
    ax.set_title("Weekly Rate and Market Index Trend")
    ax.set_ylabel("Median posted rate")
    ax2.set_ylabel("Median market index")
    fig.tight_layout()
    fig.savefig(output_dir / "weekly_rate_market_trend.png")
    plt.close(fig)
    missing = []
    for name, frame in [("train", train), ("validation", validation)]:
        temp = frame.copy()
        temp["month"] = temp["date"].dt.to_period("M").astype(str)
        for column in ["weight", "market_index"]:
            rates = temp.groupby("month")[column].apply(lambda s: s.isna().mean()).reset_index(name="missing_rate")
            rates["dataset"] = name
            rates["column"] = column
            missing.append(rates)
    missing_df = pd.concat(missing)
    fig, ax = plt.subplots(figsize=(9, 5), dpi=140)
    for (dataset, column), group in missing_df.groupby(["dataset", "column"]):
        ax.plot(group["month"], group["missing_rate"], marker="o", label=f"{dataset} {column}")
    ax.set_title("Missingness by Month")
    ax.set_ylabel("Missing rate")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "missingness_by_month.png")
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=140)
    for ax, column in zip(axes, ["distance", "weight", "market_index"]):
        ax.hist(pd.to_numeric(train[column], errors="coerce").dropna(), bins=45, alpha=0.55, label="train")
        ax.hist(pd.to_numeric(validation[column], errors="coerce").dropna(), bins=45, alpha=0.55, label="validation")
        ax.set_title(column)
    axes[0].legend()
    fig.suptitle("Train vs Validation Numeric Distributions")
    fig.tight_layout()
    fig.savefig(output_dir / "train_validation_numeric_distributions.png")
    plt.close(fig)
    routes = (train["pickup"].astype(str) + " -> " + train["delivery"].astype(str)).value_counts().head(20)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=140)
    routes.sort_values().plot(kind="barh", ax=ax)
    ax.set_title("Top 20 Training Routes")
    ax.set_xlabel("Rows")
    fig.tight_layout()
    fig.savefig(output_dir / "top_routes.png")
    plt.close(fig)
    _hist(y.clip(upper=y.quantile(0.995)), output_dir / "target_tail_view.png", "Posted Rate Tail View (clipped at p99.5 for readability)", "posted_rate")


def _hist(series: pd.Series, path: Path, title: str, xlabel: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5), dpi=140)
    ax.hist(pd.Series(series).dropna(), bins=60, color="#245C69", alpha=0.86)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Rows")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
