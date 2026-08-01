from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from freight_rate_ml.data import load_all, validate_final_december, validate_final_predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    load_all(root)
    validate_final_predictions(root / "validation_predictions.csv")
    validate_final_december(root / "december_chart_inputs.csv")
    required = [
        root / "scorer_results" / "candidate_december.png",
        root / "reports" / "freight_rate_assessment_report.pdf",
        root / "artifacts" / "metrics" / "model_rankings.csv",
        root / "artifacts" / "eda" / "data_audit.json",
        root / "README.md",
        root / "index.html",
    ]
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise SystemExit(f"Missing required artifacts: {missing}")
    scorer = subprocess.run(
        ["python", "score.py", "--predictions", "validation_predictions.csv", "--december-predictions", "december_chart_inputs.csv"],
        cwd=root,
        text=True,
        capture_output=True,
    )
    if scorer.returncode != 0:
        raise SystemExit(scorer.stdout + scorer.stderr)
    print("Output validation passed.")
    print(scorer.stdout.strip())


if __name__ == "__main__":
    main()
