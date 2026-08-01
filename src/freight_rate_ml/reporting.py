from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def package_versions() -> dict[str, str]:
    versions = {"python": sys.version.split()[0], "platform": platform.platform()}
    for package in ["pandas", "numpy", "sklearn", "matplotlib", "reportlab"]:
        try:
            module = __import__(package)
            versions[package] = getattr(module, "__version__", "unknown")
        except Exception:
            versions[package] = "not installed"
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        commit = "not a git repository"
    versions["git_commit"] = commit
    return versions


def generate_report(root: Path) -> Path:
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    output = reports / "freight_rate_assessment_report.pdf"
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(output), pagesize=letter, rightMargin=0.55 * inch, leftMargin=0.55 * inch, topMargin=0.55 * inch, bottomMargin=0.55 * inch)
    elements = []
    metrics = pd.read_csv(root / "artifacts" / "metrics" / "model_rankings.csv")
    folds = pd.read_csv(root / "artifacts" / "metrics" / "folds.csv")
    dec_metrics = pd.read_csv(root / "artifacts" / "metrics" / "december_model_results.csv")
    audit = json.loads((root / "artifacts" / "eda" / "data_audit.json").read_text(encoding="utf-8"))
    selected = json.loads((root / "artifacts" / "metrics" / "selected_model.json").read_text(encoding="utf-8"))
    final_run = json.loads((root / "artifacts" / "manifests" / "final_run.json").read_text(encoding="utf-8"))
    versions = package_versions()

    def h(text):
        elements.append(Paragraph(text, styles["Heading2"]))

    def p(text):
        elements.append(Paragraph(text, styles["BodyText"]))
        elements.append(Spacer(1, 0.08 * inch))

    elements.append(Paragraph("Freight Rate Prediction Assessment", styles["Title"]))
    p(f"Generated with seed {final_run['seed']} on {final_run['generated_at']} using Python {versions['python']}. Git commit: {versions['git_commit']}.")
    h("Executive Summary")
    p(
        f"The final full-feature model is {selected['full_model']} selected by mean out-of-time MAE. "
        f"The fixed December chart uses {selected['december_model']}, a separate model trained without market_index or quote_signal because those inputs are absent from the chart file. "
        "No hidden validation target values were used or inferred."
    )
    h("Data Overview and Quality")
    train = audit["train-test.csv"]
    valid = audit["validation.csv"]
    p(
        f"Training data contains {train['shape'][0]:,} rows from {train['date_range'][0]} through {train['date_range'][1]}; "
        f"validation contains {valid['shape'][0]:,} rows from {valid['date_range'][0]} through {valid['date_range'][1]}. "
        f"Training has {train['missing']['weight']} missing weights, {train['missing']['market_index']} missing market_index values, and {train['negative_weights']} negative weights."
    )
    p(
        f"Validation includes {len(audit['train_validation_overlap']['unseen_validation_cities'])} cities not observed in labeled training and "
        f"{audit['train_validation_overlap']['unseen_validation_routes']:,} unseen exact pickup-delivery routes. "
        f"Market index mean shifts from {audit['market_index_shift']['train_mean']:.3f} to {audit['market_index_shift']['validation_mean']:.3f}."
    )
    _add_image(elements, root / "artifacts" / "eda" / "weekly_rate_market_trend.png", 6.8, 3.6)
    _add_image(elements, root / "artifacts" / "eda" / "train_validation_numeric_distributions.png", 6.8, 2.5)
    h("Validation Design")
    p("Model selection uses chronological expanding-window validation. Each fold trains only on earlier dates than its validation period, mirroring the final future prediction task.")
    _add_table(elements, folds[["fold", "train_start", "train_end", "valid_start", "valid_end", "train_rows", "valid_rows"]], max_rows=8)
    h("Feature Engineering and Treatments")
    p(
        "Features include date seasonality, geography from coordinates, haversine distance, route and city-pair categories, frequency features fitted inside each training fold, equipment, distance, weight flags, and market/quote interactions for the full model. "
        "Negative weights were benchmarked as absolute-value repair and as missing-with-flag; missing market_index values are imputed from fold training medians with a missingness flag."
    )
    h("Model Comparison")
    p("Baselines include global median, distance-only linear regression, equipment median rate-per-mile, and ridge regression. Nonlinear candidates include histogram gradient boosting and extra trees with controlled deterministic settings.")
    table = metrics[["model", "mean_mae", "recent_mae", "mean_rmse", "mean_r2", "mean_wape"]].head(8).copy()
    _add_table(elements, table, max_rows=8)
    h("Error Analysis and Limitations")
    p(
        "The primary metric is MAE because posted rates are strongly right-skewed with rare high-price tails that can dominate RMSE. "
        "RMSE, WAPE, seen/unseen-route MAE, and normal/extreme target subsets were retained as diagnostics. "
        "The main limitation is that final validation labels are hidden, so all model decisions rely on labeled temporal backtests only."
    )
    _add_image(elements, root / "artifacts" / "metrics" / "selected_residuals.png", 6.8, 3.2)
    _add_image(elements, root / "artifacts" / "metrics" / "selected_error_by_equipment.png", 6.8, 3.2)
    h("Fixed December Prediction")
    dec_summary = dec_metrics.groupby("model", as_index=False)["mae"].mean().sort_values("mae").head(5)
    p("The December chart input lacks exogenous market fields, so model choice for this file used reduced-feature temporal backtesting under the same missing-feature contract.")
    _add_table(elements, dec_summary.rename(columns={"mae": "mean_temporal_mae"}), max_rows=5)
    p("The following image is the exact chart generated by the supplied score.py from the completed December prediction CSV.")
    _add_image(elements, root / "scorer_results" / "candidate_december.png", 6.8, 3.1)
    h("Reproducibility")
    p("From a clean checkout, install requirements and run: python scripts/run_pipeline.py --project-root . Then validate with: python scripts/validate_outputs.py --project-root . and python score.py --predictions validation_predictions.csv --december-predictions december_chart_inputs.csv.")
    p("Human-only remaining steps are publishing the GitHub repository, recording the Loom walkthrough, pasting the Loom URL, and submitting the assessment.")
    doc.build(elements)
    return output


def write_loom_materials(root: Path) -> None:
    reports = root / "reports"
    script = """# Loom Script - 2-3 Minute Walkthrough

## 0:00-0:20 Objective and Data
This repository predicts spot freight rates. I use `train-test.csv` as labeled chronological development data, then generate the required 12,000 predictions for `validation.csv` and the fixed December chart file.

## 0:20-0:50 EDA and Quality Issues
The labeled period runs January through October 2025, while final validation runs November and December. I found missing weights, missing market index values, negative physical weights, strong right-skew in posted rates, and unseen cities/routes in the future validation file.

## 0:50-1:25 Validation and Model Choice
Because the final data is future-dated, model selection uses expanding temporal folds. I compare simple baselines, ridge regression, histogram gradient boosting, and extra trees. The selected full model is chosen by mean out-of-time MAE, with RMSE, WAPE, tail behavior, and recent-fold performance as diagnostics.

## 1:25-2:05 Code Walkthrough
The main command is `python scripts/run_pipeline.py --project-root .`. The important modules are `data.py` for schema checks and raw preservation, `features.py` for fold-safe features, `validation.py` for chronological folds, `models.py` for candidates, and `pipeline.py` for orchestration.

## 2:05-2:30 Outputs and Reproducibility
The pipeline writes `validation_predictions.csv`, `december_chart_inputs.csv`, the scorer chart under `scorer_results/`, metrics under `artifacts/metrics/`, and the final PDF report under `reports/`. The Loom URL should be added here after recording.

Loom URL: TODO
"""
    checklist = """# Loom Checklist

- Show `README.md` setup and one-command run instructions.
- Show `artifacts/eda/data_quality_findings.md`.
- Show `artifacts/metrics/model_rankings.csv` and selected model JSON.
- Show `src/freight_rate_ml/features.py`, `models.py`, and `pipeline.py`.
- Show `validation_predictions.csv` and `december_chart_inputs.csv`.
- Show `scorer_results/candidate_december.png`.
- Show `reports/freight_rate_assessment_report.pdf`.

Loom URL after recording: TODO
"""
    (reports / "LOOM_SCRIPT.md").write_text(script, encoding="utf-8")
    (reports / "LOOM_CHECKLIST.md").write_text(checklist, encoding="utf-8")


def _add_table(elements, frame: pd.DataFrame, max_rows: int = 10) -> None:
    display = frame.head(max_rows).copy()
    for col in display.columns:
        if pd.api.types.is_numeric_dtype(display[col]):
            display[col] = display[col].map(lambda x: f"{x:,.3f}" if pd.notna(x) else "")
    data = [list(display.columns)] + display.astype(str).values.tolist()
    table = Table(data, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#245C69")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 7.2),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#D9E2E4")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F8F9")]),
            ]
        )
    )
    elements.append(table)
    elements.append(Spacer(1, 0.12 * inch))


def _add_image(elements, path: Path, width_inches: float, height_inches: float) -> None:
    if path.is_file() and path.stat().st_size > 0:
        elements.append(Image(str(path), width=width_inches * inch, height=height_inches * inch))
        elements.append(Spacer(1, 0.12 * inch))
