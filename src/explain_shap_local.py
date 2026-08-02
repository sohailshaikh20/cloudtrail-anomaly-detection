"""
explain_shap_local.py  —  Phase 5, part 2 (local explanations)

For individual sessions, shows how each feature pushed the model from its
baseline up to (or down from) the final attack score. The analyst-facing view.

Picks three illustrative sessions:
  * confidently-flagged attack   (highest xgb_proba)
  * confidently-normal session   (lowest  xgb_proba)
  * borderline session           (xgb_proba closest to 0.5)

Outputs: results/figures/shap_local_<case>.png
Run:     python src/explain_shap_local.py
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
SCORED = ROOT / "data" / "processed" / "tier3_scored.parquet"
MODEL_FILE = ROOT / "models" / "tier3_xgboost.json"
FIG_DIR = ROOT / "results" / "figures"

FEATURE_COLS = [
    "api_call_count", "api_calls_per_min", "api_diversity",
    "error_rate", "write_read_ratio",
    "n_source_ips", "n_regions", "night_fraction",
    "iam_escalation_flag",
]


def explain_one(explainer, X, row_idx, case_name, proba):
    sv = explainer(X.iloc[[row_idx]])

    contribs = pd.Series(sv.values[0], index=FEATURE_COLS).sort_values(
        key=lambda s: s.abs(), ascending=False
    )
    print(f"\n=== {case_name}  (model attack-probability = {proba:.3f}) ===")
    print("  feature contributions (SHAP), largest first:")
    for name, val in contribs.items():
        arrow = "attack" if val > 0 else "normal"
        print(f"    {name:20s} {val:+7.3f}  -> pushes {arrow}   (value={X.iloc[row_idx][name]:.3g})")

    plt.figure()
    shap.plots.waterfall(sv[0], show=False, max_display=len(FEATURE_COLS))
    plt.tight_layout()
    out = FIG_DIR / f"shap_local_{case_name}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved -> {out}")


def main() -> None:
    model = XGBClassifier()
    model.load_model(MODEL_FILE)

    df = pd.read_parquet(SCORED).reset_index(drop=True)
    X = df[FEATURE_COLS]

    explainer = shap.TreeExplainer(model)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    idx_attack = df["xgb_proba"].idxmax()
    idx_normal = df["xgb_proba"].idxmin()
    idx_border = (df["xgb_proba"] - 0.5).abs().idxmin()

    explain_one(explainer, X, idx_attack, "attack", df.loc[idx_attack, "xgb_proba"])
    explain_one(explainer, X, idx_normal, "normal", df.loc[idx_normal, "xgb_proba"])
    explain_one(explainer, X, idx_border, "borderline", df.loc[idx_border, "xgb_proba"])

    print("\nDone. Three local explanation figures saved in results/figures/.")


if __name__ == "__main__":
    main()
