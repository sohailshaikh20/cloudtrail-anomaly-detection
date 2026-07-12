"""
explain_shap.py  —  Phase 5, part 1 (global explanation)

Loads the trained Tier 3 XGBoost model and computes SHAP values for every
labelled session, then produces the GLOBAL explanation: which features drive
the model's attack predictions overall.

Outputs:
  results/figures/shap_global_bar.png
  results/figures/shap_global_beeswarm.png
  results/shap_global_importance.csv
Run:
  python src/explain_shap.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
LABELLED = ROOT / "data" / "processed" / "labelled.parquet"
MODEL_FILE = ROOT / "models" / "tier3_xgboost.json"
FIG_DIR = ROOT / "results" / "figures"
CSV_OUT = ROOT / "results" / "shap_global_importance.csv"

FEATURE_COLS = [
    "api_call_count", "api_calls_per_min", "api_diversity",
    "error_rate", "write_read_ratio",
    "n_source_ips", "n_regions", "night_fraction",
    "iam_escalation_flag",
]


def main() -> None:
    model = XGBClassifier()
    model.load_model(MODEL_FILE)

    df = pd.read_parquet(LABELLED)
    data = df[df["label"].isin([0, 1])].copy()
    X = data[FEATURE_COLS]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    mean_abs = np.abs(shap_values).mean(axis=0)
    importance = pd.Series(mean_abs, index=FEATURE_COLS).sort_values(ascending=False)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    importance.to_csv(CSV_OUT, header=["mean_abs_shap"])

    print("Global SHAP importance (mean |SHAP| across sessions):\n")
    for name, val in importance.items():
        bar = "#" * int(val / importance.max() * 30) if importance.max() > 0 else ""
        print(f"  {name:20s} {val:8.4f}  {bar}")

    plt.figure()
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "shap_global_bar.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "shap_global_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nSaved:")
    print(f"  {FIG_DIR / 'shap_global_bar.png'}")
    print(f"  {FIG_DIR / 'shap_global_beeswarm.png'}")
    print(f"  {CSV_OUT}")


if __name__ == "__main__":
    main()
