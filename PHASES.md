# PHASES.md - Continuous Execution Checklist

## Execution rule

Complete phases in order. Do not pause, ask for confirmation, or wait for a user response between phases. Each phase has a gate. If the gate fails, remain in that phase, diagnose the cause, repair it, rerun the gate, and only then continue. Mark progress in `STATUS.md`, but status logging must not interrupt execution.

The only acceptable early stop is a truly external blocker that cannot be solved locally. Missing packages, code defects, failed models, path mismatches, report failures, and scorer errors are local problems and must be fixed rather than reported as blockers.

---

## Phase 0 - Bootstrap, instruction review, and preservation

### Tasks

- Read in full:
  - `CONTEXT.md`
  - `PLAN.md`
  - `PHASES.md`
  - `freight-rate-ml-assessment.pdf`
  - `readme.md`
  - `score.py`
- Inventory every file and record size and hash.
- Create `STATUS.md` with all phases initially incomplete.
- Establish a clean project layout.
- Preserve exact raw copies of all supplied CSV inputs.
- Decide how root files and the README's `data/` paths will be reconciled.
- Initialize package/module structure, configuration, logging, `.gitignore`, and tests directory.
- Expand dependencies only as needed for the intended implementation.
- Confirm the Python interpreter and package manager commands that will be documented.

### Required outputs

- `STATUS.md`
- `artifacts/manifests/input_hashes.json`
- clean source/test/artifact directory structure
- preserved raw inputs
- updated dependency specification

### Gate

- Every supplied file can be located and hashed.
- Raw CSV copies match original hashes.
- `python score.py --help` succeeds.
- The package imports or initial smoke test succeeds.

Immediately continue to Phase 1.

---

## Phase 1 - Schema validation, data audit, and EDA

### Tasks

- Implement reusable schema validators for labeled, validation, template, and December files.
- Load dates explicitly and validate ID uniqueness.
- Reproduce row/column counts, date ranges, missing counts, category counts, and target statistics from `CONTEXT.md`.
- Detect and quantify:
  - negative weights
  - missing weights and market index
  - target/rate-per-mile extremes
  - duplicate rows/features
  - unseen validation cities and routes
  - numeric distribution shift
  - category-frequency shift
  - city-to-coordinate consistency
- Generate EDA tables and charts listed in `PLAN.md`.
- Write a concise data-quality findings document that states proposed treatments but distinguishes findings from final model choices.

### Required outputs

- `artifacts/eda/data_audit.json`
- `artifacts/eda/data_profile.csv`
- `artifacts/eda/train_validation_drift.csv`
- `artifacts/eda/data_quality_findings.md`
- EDA PNG files
- unit tests for schema and basic audit logic

### Gate

- Validators pass on supplied files.
- Observed facts are internally consistent.
- EDA artifacts are non-empty and readable.
- Tests for Phase 1 pass.
- No raw input file was modified.

Immediately continue to Phase 2.

---

## Phase 2 - Fold-safe preprocessing and feature engineering

### Tasks

- Implement deterministic date, geographic, route, equipment, weight, and interaction features.
- Implement at least two candidate negative-weight treatments with flags.
- Implement missing-value handling without future leakage.
- Build city-coordinate lookup and consistency checks.
- Implement categorical handling that supports unseen values.
- Ensure `load_id` and `posted_rate` are excluded from feature matrices.
- Make feature generation work on:
  - labeled data
  - validation data
  - enriched December data
- Add serialization support for preprocessors/models.

### Required outputs

- feature/preprocessing modules
- feature-list artifact or schema
- tests for dates, geography, missingness, negative weights, unseen categories, and target exclusion

### Gate

- A sample train/validation transform completes with identical feature contract.
- No leakage test fails.
- Enriched December rows can be transformed by the chart-compatible feature pipeline.
- Phase 2 tests pass.

Immediately continue to Phase 3.

---

## Phase 3 - Temporal validation framework and baselines

### Tasks

- Implement chronological expanding or rolling folds.
- Save exact fold date boundaries and row counts.
- Assert every fold has training dates strictly before validation dates.
- Implement global median, distance-based, equipment rate-per-mile, and regularized linear baselines.
- Calculate all required metrics per fold and aggregate.
- Add seen/unseen route or city diagnostic metrics.
- Save baseline predictions for error inspection.

### Required outputs

- `artifacts/metrics/folds.csv`
- baseline rows in `artifacts/metrics/model_results.csv`
- baseline ranking summary
- validation/fold tests

### Gate

- At least three useful chronological folds execute.
- Fold chronology assertions pass.
- Baseline metrics are finite and reproducible.
- The model-results artifact contains one row per candidate/fold/metric grouping as designed.

Immediately continue to Phase 4.

---

## Phase 4 - Strong models, controlled tuning, and error analysis

### Tasks

- Train at least two serious nonlinear model families.
- Compare raw-target and robust/log-target variants where appropriate.
- Compare negative-weight and missing-market strategies.
- Use bounded deterministic tuning, not an uncontrolled exhaustive search.
- Track fit and prediction time.
- Rank models by temporal MAE and defined tie-breakers.
- Evaluate a simple ensemble of top diverse candidates only if justified.
- Perform residual and subgroup analysis.
- Make one constrained improvement pass based on observed errors.
- Select the final full-feature model configuration.
- Select the final chart-compatible model configuration using reduced-feature temporal backtesting.

### Required outputs

- completed `artifacts/metrics/model_results.csv`
- `artifacts/metrics/model_rankings.csv`
- `artifacts/metrics/december_model_results.csv`
- selected configuration JSON/YAML files
- error-analysis charts/tables
- serialized best fold model(s) if useful

### Gate

- Multiple candidates completed all required folds or have a documented technical reason for exclusion.
- Selection is based on executed metrics, not preference.
- The recent fold does not reveal an unaddressed catastrophic regression.
- The chart-compatible model was evaluated under its actual feature constraints.
- All metrics and selection artifacts are reproducible.

Immediately continue to Phase 5.

---

## Phase 5 - Final training and 12,000 validation predictions

### Tasks

- Refit the selected full-feature pipeline on all labeled rows.
- Generate predictions for every validation row.
- Merge predictions to the supplied template by `load_id` with one-to-one assertions.
- Preserve exact template order and required column order.
- Apply only documented prediction safety constraints.
- Save final model/preprocessor artifacts and a run manifest.
- Write `validation_predictions.csv` with no index.

### Required outputs

- final full-feature model artifact(s)
- `validation_predictions.csv`
- prediction summary statistics
- `artifacts/manifests/final_run.json`

### Gate

- Exactly 12,000 rows.
- Exactly `load_id,predicted_rate` in that order.
- IDs equal the expected set and contain no duplicates.
- All predictions are numeric, finite, and strictly positive.
- Internal validator passes.

Immediately continue to Phase 6.

---

## Phase 6 - December predictions and supplied scorer

### Tasks

- Enrich the fixed rows with the verified city-coordinate map and derived features.
- Refit the selected chart-compatible model on all labeled data.
- Predict all 31 December dates.
- Preserve all original fixed fields and exact column order.
- Save the completed submission file as `december_chart_inputs.csv`.
- Run the supplied scorer using the final validation and December files.
- Capture scorer stdout/stderr and exit status in an artifact.
- Inspect `scorer_results/candidate_december.png` visually.
- If the chart is empty, corrupt, implausibly flat due to a bug, or malformed, diagnose and correct the upstream December pipeline, then rerun the scorer.

### Required outputs

- final chart-compatible model artifact(s)
- completed `december_chart_inputs.csv`
- `scorer_results/candidate_december.png`
- `artifacts/manifests/scorer_run.txt` or JSON

### Gate

- Exactly 31 rows and seven required columns in order.
- Exact dates and fixed scenario values pass.
- All December predictions are finite and positive.
- `score.py` exits successfully and reports validation of 12,000 and 31 predictions.
- The scorer chart exists and has been visually inspected.

Immediately continue to Phase 7.

---

## Phase 7 - Automated assessment report and Loom materials

### Tasks

- Generate the report from saved EDA, metrics, configurations, and scorer chart.
- Include exact temporal split dates and candidate metrics.
- Explain data-quality treatments and model choice.
- Distinguish full-feature and chart-compatible models.
- Include limitations and avoid any claim about hidden final metrics.
- Embed the exact scorer-generated December chart.
- Create the 2-3 minute Loom script and checklist.
- Render or inspect every PDF page, or inspect the DOCX output, to verify layout.

### Required outputs

- `reports/freight_rate_assessment_report.pdf` or `.docx`
- `reports/LOOM_SCRIPT.md`
- `reports/LOOM_CHECKLIST.md`
- optional report source template

### Gate

- Report opens successfully and is non-empty.
- Required sections are present.
- Metrics match machine-readable artifacts.
- The exact December chart is present and legible.
- No clipped tables, blank pages, broken images, or invented results.
- Loom script fits approximately 2-3 minutes when read normally.

Immediately continue to Phase 8.

---

## Phase 8 - README, tests, reproducibility, and repository hardening

### Tasks

- Finalize README with commands that match the actual implementation.
- Add setup, clean run, faster rerun, scoring, testing, report, and Loom instructions.
- Add/complete unit and integration tests.
- Add output-validation CLI.
- Ensure deterministic seeds and version recording.
- Clean temporary files and update `.gitignore`.
- Run a clean or near-clean reproduction in a fresh environment if feasible.
- Confirm no absolute local paths are embedded.
- Confirm repository does not depend on hidden notebooks or manual edits.

### Required outputs

- final `README.md`
- full tests
- clean dependency files
- `.gitignore`
- optional `Makefile` or task runner

### Gate

- README commands execute successfully.
- Tests pass.
- Output validator passes.
- Scorer still passes after cleanup.
- Report still opens and contains the chart.
- A new user can identify the one-command pipeline.

Immediately continue to Phase 9.

---

## Phase 9 - Final audit and completion record

### Tasks

- Run the complete final audit from `PLAN.md`.
- Re-run tests, output validation, and scorer.
- Record file hashes and sizes for final deliverables.
- Verify final predictions do not contain NaN, infinity, zero, negatives, or index columns.
- Verify the selected metrics and model configuration are reflected consistently in report and README.
- Verify no phase remains incomplete in `STATUS.md`.
- Write `FINAL_STATUS.md` with full results and honest limitations.
- Clearly list only human/external remaining actions:
  - create or publish the GitHub repository if Codex lacks account access
  - record the Loom video
  - paste the Loom URL
  - submit the deliverables

### Required outputs

- `FINAL_STATUS.md`
- final hashes/manifests
- all required submission files

### Final gate

The work may stop only when:

- all local phases are marked complete
- all tests pass
- both CSVs pass exact schema/invariant validation
- supplied scorer succeeds
- chart is inspected
- report is inspected
- README is accurate
- final status is written

Do not end with a plan for future coding. End with a completed implementation and a concise handoff of human-only actions.

