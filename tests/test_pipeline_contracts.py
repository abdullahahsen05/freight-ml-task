from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from freight_rate_ml.data import load_all, validate_final_december, validate_final_predictions
from freight_rate_ml.features import FreightFeatureBuilder, build_city_coordinate_lookup, enrich_december
from freight_rate_ml.validation import split_fold, temporal_folds


def test_input_schemas_load():
    train, validation, template, december = load_all(ROOT)
    assert train.shape == (48000, 14)
    assert validation.shape == (12000, 13)
    assert template["load_id"].tolist() == validation["load_id"].tolist()
    assert december.shape == (31, 7)


def test_fold_chronology():
    train, *_ = load_all(ROOT)
    for fold in temporal_folds():
        train_fold, valid_fold = split_fold(train, fold)
        assert train_fold["date"].max() < valid_fold["date"].min()


def test_feature_contract_and_no_target_leakage():
    train, validation, _, december = load_all(ROOT)
    builder = FreightFeatureBuilder(include_market=True, negative_weight="abs").fit(train.head(500))
    features = builder.transform(validation.head(20))
    assert "posted_rate" not in features.columns
    assert "load_id" not in features.columns
    assert np.isfinite(features[builder.spec.numeric].to_numpy(dtype=float)).all()
    chart_builder = FreightFeatureBuilder(include_market=False, negative_weight="abs").fit(train.head(500))
    lookup = build_city_coordinate_lookup(train, validation)
    enriched = enrich_december(december, lookup)
    chart_features = chart_builder.transform(enriched)
    assert "market_index" not in chart_features.columns
    assert len(chart_features) == 31


def test_city_coordinate_lookup_consistent():
    train, validation, *_ = load_all(ROOT)
    lookup = build_city_coordinate_lookup(train, validation)
    assert "Lexington" in lookup
    assert "Fort Wayne" in lookup


def test_final_artifacts_when_present():
    predictions = ROOT / "validation_predictions.csv"
    december = ROOT / "december_chart_inputs.csv"
    if predictions.exists() and december.exists():
        validate_final_predictions(predictions)
        validate_final_december(december)


def test_scorer_when_outputs_present():
    if (ROOT / "validation_predictions.csv").exists() and (ROOT / "december_chart_inputs.csv").exists():
        result = subprocess.run(
            ["python", "score.py", "--predictions", "validation_predictions.csv", "--december-predictions", "december_chart_inputs.csv"],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr


def test_report_and_chart_when_present():
    chart = ROOT / "scorer_results" / "candidate_december.png"
    report = ROOT / "reports" / "freight_rate_assessment_report.pdf"
    if chart.exists():
        assert chart.stat().st_size > 1000
    if report.exists():
        assert report.stat().st_size > 10000
