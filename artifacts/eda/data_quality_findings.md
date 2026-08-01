# Data Quality Findings

- Training data has 48,000 rows and validation has 12,000 rows.
- Training dates span 2025-01-01 to 2025-10-31; validation dates span 2025-11-01 to 2025-12-31.
- Missing values: training weight=300, training market_index=374, validation weight=165, validation market_index=249.
- Negative weights are present: training=292, validation=145. The pipeline benchmarks absolute-value repair and missing-value imputation with flags.
- Validation contains unseen cities: Allentown, Charlotte, Chicago, Jackson, Knoxville, Laredo, Norfolk, San Diego; exact unseen validation routes=736.
- Market index shifts from mean 1.083 in training to 0.927 in validation.
- Target rates are right-skewed with rare high rate-per-mile rows, so temporal MAE is the primary selection metric and RMSE/tail metrics are reported as diagnostics.
- December chart rows lack market_index and quote_signal. A separate chart-compatible model excludes those fields and is backtested under the same constraint.
- Root CSVs are preserved under data/raw with hash checks; pipeline commands read the original root files and never modify the supplied inputs.
- City-to-coordinate consistency is validated before December enrichment.
