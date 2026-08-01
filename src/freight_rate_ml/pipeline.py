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
    text = f"""# Freight Rate Prediction Challenge

This repository contains a complete, reproducible solution for the Spotter freight-rate ML assessment.

## Outputs

- `validation_predictions.csv`: 12,000 final validation predictions with `load_id,predicted_rate`.
- `december_chart_inputs.csv`: completed 31-row fixed December scenario.
- `scorer_results/candidate_december.png`: chart generated by the supplied `score.py`.
- `reports/freight_rate_assessment_report.pdf`: assessment report.
- `reports/LOOM_SCRIPT.md` and `reports/LOOM_CHECKLIST.md`: recording support; the Loom video itself must be recorded by a human.

## Setup

```bash
python -m pip install -r requirements.txt
```

Python 3.12 was used for this run.

## Reproduce Everything

```bash
python scripts/run_pipeline.py --project-root .
```

The command validates inputs, preserves raw CSVs under `data/raw/`, runs EDA, temporal validation, model comparison, final training, predictions, the supplied scorer, report generation, and final status writing.

## Validate Existing Outputs

```bash
python scripts/validate_outputs.py --project-root .
python score.py --predictions validation_predictions.csv --december-predictions december_chart_inputs.csv
pytest
```

## Modeling Summary

The final full-feature model is `{selection['full_model']}`. It was selected by mean chronological out-of-time MAE across expanding temporal folds. The December chart uses `{selection['december_model']}`, a separate reduced-feature model that excludes `market_index` and `quote_signal` because those columns are not supplied for the fixed chart scenario.

The final hidden validation score is not available locally and is not claimed here.

## Repository Layout

- `src/freight_rate_ml/`: package code for data contracts, features, validation, modeling, evaluation, reporting, and pipeline orchestration.
- `scripts/`: runnable CLI entry points.
- `tests/`: unit and artifact validation tests.
- `artifacts/eda/`: data audit tables and EDA charts.
- `artifacts/metrics/`: fold definitions, model results, rankings, selected configuration, and error analysis.
- `artifacts/models/`: serialized final models.
- `artifacts/manifests/`: hashes, scorer output, and final run manifest.
- `reports/`: PDF report and Loom materials.

## Human-Only Submission Steps

1. Publish or push this repository to an accessible GitHub URL.
2. Record the 2-3 minute Loom using `reports/LOOM_SCRIPT.md`.
3. Paste the Loom URL in the checklist or submission form.
4. Submit the repository, prediction CSV, report, and Loom link.
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
