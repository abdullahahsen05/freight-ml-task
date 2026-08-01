# Loom Script - 2-3 Minute Walkthrough

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
