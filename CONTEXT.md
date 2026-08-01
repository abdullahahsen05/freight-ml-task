# CONTEXT.md - Freight Rate ML Assessment

## 1. Mission

Build a complete, reproducible, GitHub-ready machine-learning solution for the Spotter Freight Rate Prediction Challenge. The solution must inspect and clean the supplied data, choose and justify a validation strategy, train and compare models, generate all required predictions, pass the supplied scorer, produce the required December chart, generate a professional PDF or DOCX report, and document how to run everything from a clean checkout.

This is an end-to-end delivery task, not an exploratory notebook exercise. The final repository must be runnable, testable, and submission-ready.

## 2. Codex execution behavior

Codex must work autonomously through every phase in `PHASES.md`.

- Read this file, `PLAN.md`, `PHASES.md`, the assessment PDF, `readme.md`, and `score.py` before changing code.
- Do not stop after a phase to ask for approval.
- Do not provide phase-by-phase status messages and wait for a reply.
- After a phase gate passes, continue immediately to the next phase.
- Diagnose and repair ordinary errors yourself: dependency issues, path issues, failed tests, model failures, report-generation problems, scorer failures, and formatting errors are not reasons to stop.
- Stop only after the final definition of done is satisfied, or when blocked by a genuinely external requirement that cannot be resolved inside the repository, such as publishing to a GitHub account or recording/uploading the Loom video. Even then, complete every local artifact first.
- Never fabricate metrics, charts, data findings, or successful command output. Every reported result must come from an executed pipeline.
- Keep a machine-readable run manifest and a human-readable final status so work can be audited.

## 3. Source-of-truth order

When instructions appear inconsistent, use this priority:

1. `freight-rate-ml-assessment.pdf`
2. `score.py`, because it defines exact file validation rules
3. `readme.md`
4. `CONTEXT.md`, `PLAN.md`, and `PHASES.md`
5. Reasonable engineering judgment documented in the repository

Do not edit the supplied assessment PDF. Preserve an untouched copy of all original input files.

## 4. Original files present in the ZIP

The supplied archive currently contains these files at the project root:

| File | Purpose |
|---|---|
| `freight-rate-ml-assessment.pdf` | Official assessment brief |
| `readme.md` | Short setup and submission instructions |
| `score.py` | Mandatory output validator and December chart generator |
| `requirements.txt` | Requirements for the provided scorer only; it is not a complete ML environment |
| `train-test.csv` | Labeled development data |
| `validation.csv` | Unlabeled final-prediction data |
| `validation-predictions-template.csv` | Required validation ID template |
| `december-chart-inputs.csv` | Fixed 31-row December scenario to complete |

The supplied README and PDF refer to a `data/` directory, but the ZIP places the CSV files in the root. The implementation must handle this deliberately. Either normalize the layout while preserving raw copies or implement robust path discovery. The final README must show commands that actually work in the final repository.

## 5. Official required deliverables

The assessment requires:

1. An accessible GitHub repository containing code, dependencies, and run instructions.
2. `validation_predictions.csv` with exactly these two columns in this order:
   - `load_id`
   - `predicted_rate`
3. A completed December prediction CSV retaining the original seven columns in this order:
   - `pickup`
   - `delivery`
   - `distance`
   - `equipment`
   - `weight`
   - `date`
   - `predicted_rate`
4. The chart created by the supplied `score.py` at `scorer_results/candidate_december.png`.
5. A PDF or DOCX report containing, at minimum:
   - The train/validation split and validation approach
   - The fixed December chart produced by `score.py`
6. A 2-3 minute Loom walkthrough covering:
   - Key EDA findings
   - Data-quality issues and treatment
   - Model-selection reasoning
   - Training and validation split
   - Important code paths

The Loom itself requires a human recording and upload. Codex must still generate a polished `reports/LOOM_SCRIPT.md` and `reports/LOOM_CHECKLIST.md` so the only remaining human action is recording and pasting the link.

## 6. Exact scorer contract

`score.py` is authoritative for output formatting.

### Validation predictions

- Exactly 12,000 rows
- Exactly two columns in this order: `load_id,predicted_rate`
- IDs must be the complete set `TE-000001` through `TE-012000`
- No missing or duplicate IDs
- `predicted_rate` must be numeric, finite, and strictly positive

### December predictions

- Exactly 31 rows
- One row for every date from `2025-12-01` through `2025-12-31`
- Exact column order: `pickup,delivery,distance,equipment,weight,date,predicted_rate`
- Fixed values on every row:
  - pickup: `Lexington`
  - delivery: `Fort Wayne`
  - distance: `360`
  - equipment: `Dry Van`
  - weight: `32000`
- Unique valid dates
- Numeric, finite, strictly positive predictions

The scorer creates `scorer_results/candidate_december.png`. Final hidden validation metrics are calculated after submission and are not exposed by the scorer. Do not claim a hidden score.

## 7. Initial dataset profile

The following facts were observed directly from the supplied CSV files. Codex must reproduce and verify them in its own data-audit outputs before relying on them.

### `train-test.csv`

- Shape: 48,000 rows x 14 columns
- Date range: `2025-01-01` through `2025-10-31`
- Target: `posted_rate`
- Features:
  - `load_id`
  - `pickup`, `delivery`
  - `pickup_lat`, `pickup_lon`, `delivery_lat`, `delivery_lon`
  - `distance`
  - `equipment`
  - `weight`
  - `date`
  - `market_index`
  - `quote_signal`
- Missing values:
  - `weight`: 300
  - `market_index`: 374
- Invalid-looking physical values:
  - 292 negative weights
  - minimum observed weight: -47,500
- Equipment classes:
  - Dry Van: 27,202
  - Reefer: 12,045
  - Flatbed: 8,753
- 64 pickup cities and 64 delivery cities
- 4,014 observed pickup-delivery pairs
- Target range is strongly right-skewed:
  - minimum approximately 57.22
  - median approximately 2,030.76
  - maximum approximately 25,533
- Rare extreme rate-per-mile values exist. These may be legitimate simulated spikes, injected anomalies, or both; do not silently delete them.

### `validation.csv`

- Shape: 12,000 rows x 13 columns
- Date range: `2025-11-01` through `2025-12-31`
- Same input features as training, excluding `posted_rate`
- Missing values:
  - `weight`: 165
  - `market_index`: 249
- 145 negative weights
- The template IDs exactly match the validation IDs and order
- Eight cities appear in validation but not in labeled training:
  - Allentown
  - Charlotte
  - Chicago
  - Jackson
  - Knoxville
  - Laredo
  - Norfolk
  - San Diego
- At least 736 validation routes are unseen as exact pickup-delivery pairs in training
- The market-index distribution shifts materially relative to training:
  - training mean approximately 1.083
  - validation mean approximately 0.927
- Equipment proportions remain broadly similar

These facts make a random split insufficient as the only validation method. The final set is a future time period, contains distribution shift, and includes unseen cities/routes.

### `december-chart-inputs.csv`

- Shape: 31 rows x 7 columns
- Dates: every day in December 2025
- All non-date load fields are fixed
- `predicted_rate` is entirely missing
- It does not contain coordinates, `market_index`, or `quote_signal`

This schema mismatch must be handled explicitly. Do not invent arbitrary constants without documenting and validating the decision.

## 8. Key modeling implications

### 8.1 Temporal generalization

The final prediction period follows the labeled period chronologically. Model selection must therefore emphasize out-of-time validation. Random train/test splitting may be used only as a secondary diagnostic and must never be the sole basis for model choice.

Use multiple expanding-window or rolling temporal folds that mimic forecasting into future dates. Include a final holdout with a horizon similar to November-December when practical. Record fold boundaries and row counts.

### 8.2 Unseen cities and routes

A model that relies only on raw pickup/delivery category memorization will fail on unseen cities. Keep geographic numeric features and derive robust geographic/lane features. Categorical route features may still be useful, but they must be combined with features that generalize to unseen entities.

### 8.3 Missing and invalid weight values

Negative freight weights are physically invalid but have magnitudes resembling normal weights. Benchmark at least these treatments inside the training folds:

- convert negative weight to absolute value and add a `weight_was_negative` flag
- treat negative values as missing and impute, with a flag

Never decide using the final unlabeled outcomes. Missing values must be imputed from training-fold statistics only, unless the selected model handles missing values natively and the behavior is documented.

### 8.4 Missing market index

`market_index` is strongly date-dependent and also shifts over time. Compare robust strategies such as:

- native missing-value handling plus a missingness flag
- training-fold date-level median/mean imputation with fallback to global median
- an imputation model trained inside each fold

Avoid leakage from future dates into past fold preprocessing.

### 8.5 Target outliers

Rare extreme rates can dominate RMSE while being difficult or impossible to predict if they are stochastic. Evaluate at least MAE, RMSE, R-squared, and a carefully handled percentage metric. Use MAE as the primary model-selection metric unless experiments give a documented reason to change it. Include performance on normal and extreme subsets. Consider log-target training, Huber/MAE objectives, clipping only at prediction-time to defensible training-derived bounds, and robust ensembling. Never delete target outliers without analysis.

### 8.6 December chart feature mismatch

The December input has only fields that are present or derivable from the fixed scenario. Use a transparent chart-prediction strategy. Preferred approach:

1. Derive Lexington and Fort Wayne coordinates from the consistent city-coordinate mapping in supplied train/validation features.
2. Derive all normal date, route, distance, geography, equipment, and weight features.
3. Train and validate a chart-compatible secondary model that excludes `market_index` and `quote_signal`, because those fields are not supplied for the fixed scenario.
4. Simulate this missing-feature condition during temporal validation and report its performance.

An alternative forecast of `market_index` or `quote_signal` is allowed only if it is implemented, validated using historical backtesting, and clearly documented. Do not fill future exogenous features with unexplained constants.

The full-feature selected model should still be used for `validation.csv`, where those columns are available except for ordinary missing values.

## 9. Expected repository architecture

Codex may refine names, but the final repository should remain understandable and modular. A suitable target is:

```text
.
├── CONTEXT.md
├── PLAN.md
├── PHASES.md
├── README.md
├── requirements.txt
├── pyproject.toml                 # optional but preferred
├── score.py                      # supplied, unchanged unless a copy is retained
├── freight-rate-ml-assessment.pdf
├── data/
│   ├── raw/                      # untouched original CSV files
│   └── processed/                # generated only
├── src/
│   └── freight_rate_ml/
│       ├── __init__.py
│       ├── config.py
│       ├── data.py
│       ├── validation.py
│       ├── features.py
│       ├── models.py
│       ├── evaluation.py
│       ├── predict.py
│       ├── reporting.py
│       └── pipeline.py
├── scripts/
│   ├── run_pipeline.py
│   └── validate_outputs.py
├── tests/
├── artifacts/
│   ├── eda/
│   ├── metrics/
│   ├── models/
│   └── manifests/
├── scorer_results/
│   └── candidate_december.png
├── reports/
│   ├── freight_rate_assessment_report.pdf
│   ├── LOOM_SCRIPT.md
│   └── LOOM_CHECKLIST.md
├── validation_predictions.csv
└── december_chart_inputs.csv     # completed submission version
```

Generated caches, virtual environments, and unnecessary binary artifacts must be ignored by Git. Keep final model artifacts only when needed for reproducibility and reasonably sized.

## 10. Reproducibility and engineering constraints

- Python scripts, not notebook-only execution, must produce the final deliverables.
- A single documented command must run the complete pipeline from raw data to final report.
- Use deterministic seeds everywhere practical.
- Record Python and package versions.
- Make all filesystem paths relative to the repository root or configurable.
- Preserve raw input files. Generated files must go to explicit output locations.
- Use logging instead of scattered prints for pipeline progress.
- Fail early with clear schema and invariant errors.
- Prevent target leakage by keeping `posted_rate` out of all features and fitting preprocessors within folds.
- Do not use `validation.csv` labels because no labels are supplied. Do not infer or fabricate them.
- Using unlabeled validation feature distributions for descriptive drift analysis is allowed, but model selection must not depend on hidden outcomes.
- Keep runtime and memory appropriate for a normal laptop. Prefer reproducible, moderately tuned models over enormous searches.
- All final CSV numeric output should be written with adequate precision and without an index column.

## 11. Metrics and selection policy

At minimum, calculate per-fold and aggregate:

- MAE
- RMSE
- R-squared
- Median absolute error
- MAPE or WAPE with explicit safeguards and definition

Primary selection: mean out-of-time MAE.

Tie-breakers, in order:

1. Performance on the most recent fold
2. RMSE and tail robustness
3. Stability across folds
4. Simplicity and reproducibility
5. Runtime

An ensemble may be selected only if it improves the predefined out-of-time metric and does not create excessive complexity. Record all candidate results in CSV or JSON, not only in narrative text.

## 12. Definition of done

The project is complete only when all of these are true:

- All phases in `PHASES.md` are completed without skipped gates.
- The raw inputs are preserved and discoverable.
- The full pipeline runs successfully from a clean environment using documented commands.
- EDA outputs and a data-quality summary exist.
- Temporal validation folds are implemented and documented.
- Baselines and multiple serious candidates were evaluated.
- The selected model and any ensemble choice are supported by executed metrics.
- `validation_predictions.csv` has exactly 12,000 valid rows and passes `score.py` validation.
- The completed December CSV has exactly 31 valid rows and passes `score.py` validation.
- `scorer_results/candidate_december.png` exists and is visually readable.
- A professional PDF or DOCX report exists and includes the exact scorer-generated chart.
- The report states the validation design, metrics, data issues, model choice, and limitations.
- `reports/LOOM_SCRIPT.md` and `reports/LOOM_CHECKLIST.md` exist.
- Automated tests pass.
- The README contains accurate setup, training, prediction, scoring, and reproduction instructions.
- A final audit verifies no missing required file, no non-positive prediction, no malformed schema, no accidental target leakage, and no undocumented manual step except recording/uploading the Loom and publishing the repository.
- `FINAL_STATUS.md` summarizes commands run, selected model, measured metrics, output paths, test results, scorer result, remaining human-only actions, and any honest limitations.

