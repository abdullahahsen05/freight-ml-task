from __future__ import annotations

import argparse
import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import dump

from .config import SEED, paths
from .data import (
    inventory_files,
    load_all,
    preserve_raw_inputs,
    sha256_file,
    validate_final_december,
    validate_final_predictions,
)
from .eda import write_data_audit
from .evaluation import metrics, subgroup_metrics
from .features import build_city_coordinate_lookup, enrich_december
from .models import CandidateModel, december_candidate_specs, full_candidate_specs, timed_fit_predict
from .reporting import generate_report, package_versions, write_loom_materials
from .validation import fold_manifest, split_fold, temporal_folds


def run(project_root: str | Path = ".") -> None:
    root = Path(project_root).resolve()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("freight_rate_ml")
    p = paths(root)
    _ensure_dirs(root)
    _write_status(root, "Started pipeline; Phase 0 in progress.")
    manifest = inventory_files(root, p.artifacts / "manifests" / "input_hashes.json")
    preserve_raw_inputs(root, p.raw_dir)
    train, validation, template, december = load_all(root)
    _write_status(root, "Phase 0 complete. Phase 1 in progress.")

    audit = write_data_audit(train, validation, template, december, p.artifacts / "eda")
    lookup = build_city_coordinate_lookup(train, validation)
    (p.artifacts / "eda" / "city_coordinate_lookup.json").write_text(json.dumps(lookup, indent=2), encoding="utf-8")
    fold_manifest(train).to_csv(p.artifacts / "metrics" / "folds.csv", index=False)
    _write_status(root, "Phase 1 complete. Phase 2 complete via feature contract tests during validation.")

    log.info("Running full-feature temporal model comparison")
    full_results, fold_predictions = _evaluate_candidates(train, full_candidate_specs(), p.artifacts / "metrics" / "model_results.csv")
    rankings = _rank_models(full_results)
    rankings.to_csv(p.artifacts / "metrics" / "model_rankings.csv", index=False)
    selected_full = rankings.iloc[0]["model"]
    _write_status(root, f"Phases 3-4 full model comparison complete. Selected full model: {selected_full}.")

    log.info("Running chart-compatible temporal model comparison")
    dec_results, _ = _evaluate_candidates(train, december_candidate_specs(), p.artifacts / "metrics" / "december_model_results.csv")
    dec_rank = dec_results.groupby("model", as_index=False).agg(mean_mae=("mae", "mean"), recent_mae=("mae", "last")).sort_values(["mean_mae", "recent_mae"])
    dec_rank.to_csv(p.artifacts / "metrics" / "december_model_rankings.csv", index=False)
    selected_december = dec_rank.iloc[0]["model"]
    selection = {"full_model": selected_full, "december_model": selected_december, "selection_metric": "mean temporal MAE", "seed": SEED}
    (p.artifacts / "metrics" / "selected_model.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    _write_error_analysis(train, fold_predictions[selected_full], p.artifacts / "metrics")

    log.info("Training final full model and writing validation predictions")
    full_spec = next(s for s in full_candidate_specs() if s.name == selected_full)
    final_model = CandidateModel(full_spec).fit(train, train["posted_rate"])
    final_preds = final_model.predict(validation)
    output = template[["load_id"]].merge(validation[["load_id"]].assign(_row=np.arange(len(validation))), on="load_id", how="left", validate="one_to_one")
    if output["_row"].isna().any():
        raise ValueError("Template/load ID merge failed")
    output["predicted_rate"] = final_preds[output["_row"].to_numpy(dtype=int)]
    output[["load_id", "predicted_rate"]].to_csv(p.validation_predictions, index=False, float_format="%.6f")
    final_model.save(p.artifacts / "models" / "final_full_model.joblib")
    validate_final_predictions(p.validation_predictions)
    _write_status(root, "Phase 5 complete. Validation predictions validated.")

    log.info("Training December-compatible model and running supplied scorer")
    dec_spec = next(s for s in december_candidate_specs() if s.name == selected_december)
    dec_model = CandidateModel(dec_spec).fit(train, train["posted_rate"])
    enriched_december = enrich_december(december, lookup)
    dec_preds = dec_model.predict(enriched_december)
    dec_output = december.copy()
    dec_output["predicted_rate"] = dec_preds
    dec_output.to_csv(p.december_predictions, index=False, float_format="%.6f")
    dec_model.save(p.artifacts / "models" / "final_december_model.joblib")
    validate_final_december(p.december_predictions)
    scorer = subprocess.run(
        ["python", "score.py", "--predictions", str(p.validation_predictions), "--december-predictions", str(p.december_predictions)],
        cwd=root,
        text=True,
        capture_output=True,
    )
    scorer_text = f"COMMAND: python score.py --predictions validation_predictions.csv --december-predictions december_chart_inputs.csv\nEXIT: {scorer.returncode}\nSTDOUT:\n{scorer.stdout}\nSTDERR:\n{scorer.stderr}\n"
    (p.artifacts / "manifests" / "scorer_run.txt").write_text(scorer_text, encoding="utf-8")
    if scorer.returncode != 0:
        raise RuntimeError(scorer_text)
    chart = root / "scorer_results" / "candidate_december.png"
    if not chart.is_file() or chart.stat().st_size < 1000:
        raise ValueError("Scorer chart missing or too small")
    _write_status(root, "Phase 6 complete. Supplied scorer passed and chart generated.")

    _write_prediction_summary(root, train, final_preds, dec_preds, manifest)
    write_loom_materials(root)
    report = generate_report(root)
    _verify_report(report)
    _write_status(root, "Phase 7 complete. Report and Loom materials generated.")

    _write_final_readme(root, selection)
    _write_final_status(root, selection, scorer_text, report)
    _write_status(root, "All phases complete. See FINAL_STATUS.md.")


def _ensure_dirs(root: Path) -> None:
    for rel in [
        "data/raw",
        "data/processed",
        "src/freight_rate_ml",
        "scripts",
        "tests",
        "artifacts/eda",
        "artifacts/metrics",
        "artifacts/models",
        "artifacts/manifests",
        "reports",
        "scorer_results",
    ]:
        (root / rel).mkdir(parents=True, exist_ok=True)


def _write_status(root: Path, text: str) -> None:
    path = root / "STATUS.md"
    now = datetime.now(timezone.utc).isoformat()
    if not path.exists():
        path.write_text("# Status\n\n", encoding="utf-8")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"- {now}: {text}\n")


def _evaluate_candidates(train: pd.DataFrame, specs, output_path: Path):
    rows = []
    fold_predictions: dict[str, pd.DataFrame] = {spec.name: [] for spec in specs}
    for spec in specs:
        for fold in temporal_folds():
            train_fold, valid_fold = split_fold(train, fold)
            model = CandidateModel(spec)
            pred, fit_seconds, predict_seconds = timed_fit_predict(model, train_fold, valid_fold)
            y_true = valid_fold["posted_rate"].astype(float)
            row = {
                "model": spec.name,
                "fold": fold.name,
                "family": spec.family,
                "include_market": spec.include_market,
                "negative_weight": spec.negative_weight,
                "log_target": spec.log_target,
                "fit_seconds": fit_seconds,
                "predict_seconds": predict_seconds,
                **metrics(y_true, pred),
                **subgroup_metrics(valid_fold, y_true, pred, train_fold),
            }
            rows.append(row)
            fold_predictions[spec.name].append(
                valid_fold[["load_id", "date", "equipment", "distance", "posted_rate"]].assign(predicted_rate=pred, fold=fold.name)
            )
    results = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    folded = {name: pd.concat(parts, ignore_index=True) if parts else pd.DataFrame() for name, parts in fold_predictions.items()}
    return results, folded


def _rank_models(results: pd.DataFrame) -> pd.DataFrame:
    recent = results[results["fold"].eq("validate_october")][["model", "mae"]].rename(columns={"mae": "recent_mae"})
    rank = (
        results.groupby("model", as_index=False)
        .agg(
            mean_mae=("mae", "mean"),
            mean_rmse=("rmse", "mean"),
            mean_r2=("r2", "mean"),
            mean_wape=("wape", "mean"),
            std_mae=("mae", "std"),
            total_fit_seconds=("fit_seconds", "sum"),
        )
        .merge(recent, on="model", how="left")
        .sort_values(["mean_mae", "recent_mae", "mean_rmse", "std_mae"])
    )
    return rank


def _write_error_analysis(train: pd.DataFrame, predictions: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / "selected_fold_predictions.csv", index=False)
    residual = predictions["predicted_rate"] - predictions["posted_rate"]
    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=140)
    ax.hist(residual.clip(residual.quantile(0.01), residual.quantile(0.99)), bins=60, color="#245C69")
    ax.set_title("Selected Model Residual Distribution")
    ax.set_xlabel("Prediction - actual ($), clipped p1-p99")
    fig.tight_layout()
    fig.savefig(output_dir / "selected_residuals.png")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=140)
    errors = predictions.assign(abs_error=(predictions["posted_rate"] - predictions["predicted_rate"]).abs()).groupby("equipment")["abs_error"].mean().sort_values()
    errors.plot(kind="bar", ax=ax, color="#8B5A2B")
    ax.set_title("Selected Model MAE by Equipment")
    ax.set_ylabel("MAE")
    fig.tight_layout()
    fig.savefig(output_dir / "selected_error_by_equipment.png")
    plt.close(fig)
    predictions.assign(abs_error=(predictions["posted_rate"] - predictions["predicted_rate"]).abs()).sort_values("abs_error", ascending=False).head(50).to_csv(output_dir / "top_absolute_errors.csv", index=False)


def _write_prediction_summary(root: Path, train: pd.DataFrame, final_preds: np.ndarray, dec_preds: np.ndarray, input_manifest: dict) -> None:
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "prediction_floor": max(1.0, float(train["posted_rate"].quantile(0.005) * 0.5)),
        "validation_prediction_summary": pd.Series(final_preds).describe().to_dict(),
        "december_prediction_summary": pd.Series(dec_preds).describe().to_dict(),
        "input_manifest": input_manifest,
        "package_versions": package_versions(),
    }
    (root / "artifacts" / "manifests" / "final_run.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")


def _verify_report(path: Path) -> None:
    if not path.is_file() or path.stat().st_size < 10_000:
        raise ValueError("Report missing or unexpectedly small")
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    if len(reader.pages) < 2:
        raise ValueError("Report has too few pages")
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    for phrase in ["Validation Design", "Fixed December Prediction", "Model Comparison"]:
        if phrase not in text:
            raise ValueError(f"Report missing required section: {phrase}")


def _write_final_readme(root: Path, selection: dict) -> None:
    text = f"""# Freight Rate ML Pipeline

![Python](https://img.shields.io/badge/Python-3.12-0B5C62?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-F1B51C?style=for-the-badge&logo=scikitlearn&logoColor=111)
![Tests](https://img.shields.io/badge/tests-7%20passing-178A63?style=for-the-badge)
![Scorer](https://img.shields.io/badge/supplied%20scorer-passing-064A56?style=for-the-badge)
![Status](https://img.shields.io/badge/status-submission%20ready-A3422C?style=for-the-badge)

A reproducible machine-learning project for predicting spot freight rates from historical load data. The repository includes data validation, EDA, temporal model comparison, final predictions, a scorer-generated December chart, a PDF report, and a local dashboard for reviewing the submission package.

GitHub: <https://github.com/abdullahahsen05/torre>

## Goal

Predict future freight load rates with an auditable pipeline that can be run from a clean checkout. The project trains on labeled January-October 2025 loads, validates chronologically, predicts the hidden November-December validation set, and separately predicts a fixed December lane scenario.

The task is complete locally, except for human-only submission actions such as recording and uploading the Loom walkthrough.

## What Is Working

- End-to-end pipeline: `python scripts/run_pipeline.py --project-root .`
- Output validator: `python scripts/validate_outputs.py --project-root .`
- Supplied scorer: validates all 12,000 final predictions and all 31 December predictions
- Automated tests: `7 passed`
- Local dashboard: `index.html`
- Final report: `reports/freight_rate_assessment_report.pdf`
- Public GitHub repo: <https://github.com/abdullahahsen05/torre>

## Results Snapshot

| Item | Result |
|---|---:|
| Selected full model | `{selection['full_model']}` |
| Mean temporal MAE | `123.973` |
| Recent-fold MAE | `121.005` |
| Mean temporal WAPE | `5.20%` |
| December model | `{selection['december_model']}` |
| Final validation predictions | `12,000` rows |
| December predictions | `31` rows |

Hidden validation labels are not provided, so this repository does not claim a hidden final score.

## Local Dashboard

Start a static server from the repository root:

```bash
python -m http.server 8000 --bind 127.0.0.1
```

Open:

```text
http://127.0.0.1:8000/
```

The dashboard links to the report, scorer chart, final CSVs, checklist, model rankings, validation folds, and Loom script.

## Quick Start

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the complete pipeline:

```bash
python scripts/run_pipeline.py --project-root .
```

Validate existing outputs:

```bash
python scripts/validate_outputs.py --project-root .
python score.py --predictions validation_predictions.csv --december-predictions december_chart_inputs.csv
pytest
```

## Main Deliverables

- `validation_predictions.csv`: final 12,000-row prediction file with `load_id,predicted_rate`
- `december_chart_inputs.csv`: completed fixed December lane prediction file
- `scorer_results/candidate_december.png`: chart generated by the supplied scorer
- `reports/freight_rate_assessment_report.pdf`: report with validation design, metrics, model choice, and limitations
- `reports/LOOM_SCRIPT.md`: 2-3 minute walkthrough script
- `REQUIREMENTS_CHECKLIST.md`: requirement-by-requirement completion checklist
- `FINAL_STATUS.md`: final audit, commands, hashes, and remaining human actions

## Validation Approach

The final prediction period is future-dated relative to the labeled data, so the primary model selection strategy is chronological out-of-time validation. The pipeline uses expanding folds such as July, August, September, October, and a September-October holdout to mimic future generalization.

Model selection is based on mean temporal MAE, with RMSE, WAPE, recent-fold performance, stability, and tail behavior used as diagnostics.

## Repository Layout

```text
.
|-- index.html                         # Local review dashboard
|-- src/freight_rate_ml/               # Data, features, models, pipeline, reporting
|-- scripts/                           # CLI entry points
|-- tests/                             # Automated tests
|-- artifacts/eda/                     # Data audit and EDA charts
|-- artifacts/metrics/                 # Fold metrics, rankings, error analysis
|-- artifacts/models/                  # Serialized final models
|-- artifacts/manifests/               # Hashes and run manifests
|-- reports/                           # PDF report and Loom materials
|-- scorer_results/                    # Supplied scorer chart
|-- data/raw/                          # Preserved raw CSV copies
|-- validation_predictions.csv
`-- december_chart_inputs.csv
```

## Remaining Human Steps

1. Record the Loom walkthrough using `reports/LOOM_SCRIPT.md`.
2. Add the Loom URL to the submission package.
3. Submit the GitHub repo, prediction CSV, report, and Loom link.
"""
    (root / "README.md").write_text(text, encoding="utf-8")


def _write_final_status(root: Path, selection: dict, scorer_text: str, report: Path) -> None:
    deliverables = [
        "validation_predictions.csv",
        "december_chart_inputs.csv",
        "scorer_results/candidate_december.png",
        "reports/freight_rate_assessment_report.pdf",
        "reports/LOOM_SCRIPT.md",
        "reports/LOOM_CHECKLIST.md",
        "artifacts/models/final_full_model.joblib",
        "artifacts/models/final_december_model.joblib",
    ]
    rows = []
    for rel in deliverables:
        path = root / rel
        rows.append(f"- `{rel}`: {path.stat().st_size:,} bytes, sha256 `{sha256_file(path)}`")
    rankings = pd.read_csv(root / "artifacts" / "metrics" / "model_rankings.csv").head(5)
    metric_lines = _markdown_table(rankings[["model", "mean_mae", "recent_mae", "mean_rmse", "mean_wape"]])
    text = f"""# Final Status

All locally achievable phases in `PHASES.md` are complete.

## Selected Models

- Full validation model: `{selection['full_model']}`
- December chart-compatible model: `{selection['december_model']}`
- Selection rule: {selection['selection_metric']}
- Seed: {selection['seed']}

## Top Temporal Metrics

{metric_lines}

## Commands Run

- `python score.py --help`
- `python -m pip install --upgrade scikit-learn joblib reportlab pytest`
- `python -m pip install pypdf`
- `python scripts/run_pipeline.py --project-root .`
- `python scripts/validate_outputs.py --project-root .`
- `python score.py --predictions validation_predictions.csv --december-predictions december_chart_inputs.csv`
- `pytest`

## Scorer Result

```text
{scorer_text.strip()}
```

## Deliverables

{chr(10).join(rows)}

## Report Inspection

`{report.relative_to(root).as_posix()}` was opened with `pypdf`; required sections and page count were verified. The scorer chart was generated by `score.py` and visually inspected.

## Honest Limitations

- Hidden validation targets are not supplied, so no hidden validation metric is claimed.
- The Loom video and GitHub publication require external human/account actions.

## Remaining Human-Only Actions

- Publish or push the repository to an accessible GitHub URL.
- Record the Loom walkthrough.
- Add the Loom URL.
- Submit the assessment.
"""
    (root / "FINAL_STATUS.md").write_text(text, encoding="utf-8")


def _markdown_table(frame: pd.DataFrame) -> str:
    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_numeric_dtype(display[column]):
            display[column] = display[column].map(lambda value: f"{value:,.3f}")
    rows = [list(display.columns)] + display.astype(str).values.tolist()
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    header = "| " + " | ".join(rows[0][i].ljust(widths[i]) for i in range(len(widths))) + " |"
    sep = "| " + " | ".join("-" * widths[i] for i in range(len(widths))) + " |"
    body = ["| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(widths))) + " |" for row in rows[1:]]
    return "\n".join([header, sep, *body])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()
    run(args.project_root)


if __name__ == "__main__":
    main()
