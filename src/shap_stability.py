"""
shap_stability.py  —  Phase 5, part 3 (SHAP stability analysis)

Research-gap contribution: is SHAP's explanation of the CloudTrail model STABLE?

(1) GLOBAL RANKING STABILITY across random subsamples (Spearman rank correlation).
(2) LOCAL EXPLANATION STABILITY under tiny perturbations (top-feature agreement).

Outputs:
  results/figures/shap_stability_global.png
  results/shap_stability_summary.txt
Run:
  python src/shap_stability.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from scipy.stats import spearmanr
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
LABELLED = ROOT / "data" / "processed" / "labelled.parquet"
MODEL_FILE = ROOT / "models" / "tier3_xgboost.json"
FIG_DIR = ROOT / "results" / "figures"
TXT_OUT = ROOT / "results" / "shap_stability_summary.txt"

FEATURE_COLS = [
    "api_call_count", "api_calls_per_min", "api_diversity",
    "error_rate", "write_read_ratio",
    "n_source_ips", "n_regions", "night_fraction",
    "iam_escalation_flag",
]


def global_importance(explainer, X):
    sv = explainer.shap_values(X)
    return np.abs(sv).mean(axis=0)


def main() -> None:
    model = XGBClassifier()
    model.load_model(MODEL_FILE)

    df = pd.read_parquet(LABELLED)
    data = df[df["label"].isin([0, 1])].copy().reset_index(drop=True)
    X = data[FEATURE_COLS]

    explainer = shap.TreeExplainer(model)
    lines = []
    def out(s=""):
        print(s); lines.append(s)

    out("(1) GLOBAL ranking stability across 10 random subsamples")
    out("    (Spearman rank correlation of feature importance order; 1.0 = identical)\n")
    rng = np.random.default_rng(42)
    n = len(X)
    sub_size = max(200, int(0.5 * n))
    rankings = []
    for i in range(10):
        idx = rng.choice(n, size=sub_size, replace=False)
        imp = global_importance(explainer, X.iloc[idx])
        rankings.append(pd.Series(imp, index=FEATURE_COLS).rank(ascending=False))

    corrs = []
    for i in range(len(rankings)):
        for j in range(i + 1, len(rankings)):
            rho, _ = spearmanr(rankings[i], rankings[j])
            corrs.append(rho)
    corrs = np.array(corrs)
    out(f"    mean pairwise Spearman : {corrs.mean():.3f}")
    out(f"    min  pairwise Spearman : {corrs.min():.3f}")
    out(f"    max  pairwise Spearman : {corrs.max():.3f}")

    avg_rank = pd.concat(rankings, axis=1).mean(axis=1).sort_values()
    out("\n    average feature rank across runs (1 = most important):")
    for name, r in avg_rank.items():
        out(f"      {name:20s} {r:4.1f}")

    if corrs.mean() > 0.9:
        verdict1 = "HIGHLY STABLE"
    elif corrs.mean() > 0.7:
        verdict1 = "STABLE"
    else:
        verdict1 = "UNSTABLE"
    out(f"\n    => Global ranking is {verdict1} (mean Spearman {corrs.mean():.3f}).")

    out("\n(2) LOCAL explanation stability under small perturbations")
    out("    (does each session's TOP feature stay the same after +-2% noise?)\n")
    sample_idx = rng.choice(n, size=min(300, n), replace=False)
    Xs = X.iloc[sample_idx].reset_index(drop=True)
    base_sv = explainer.shap_values(Xs)
    base_top = np.abs(base_sv).argmax(axis=1)

    agree = 0
    trials = 5
    for t in range(trials):
        noise = 1 + rng.normal(0, 0.02, size=Xs.shape)
        Xp = Xs * noise
        pert_sv = explainer.shap_values(Xp)
        pert_top = np.abs(pert_sv).argmax(axis=1)
        agree += (base_top == pert_top).mean()
    top_stability = agree / trials
    out(f"    top-feature agreement after perturbation : {top_stability*100:.1f}%")

    if top_stability > 0.9:
        verdict2 = "HIGHLY STABLE"
    elif top_stability > 0.75:
        verdict2 = "STABLE"
    else:
        verdict2 = "MODERATELY STABLE"
    out(f"    => Local explanations are {verdict2}.")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rank_df = pd.concat(rankings, axis=1)
    order = avg_rank.index.tolist()
    means = rank_df.loc[order].mean(axis=1)
    stds = rank_df.loc[order].std(axis=1)
    plt.figure(figsize=(8, 5))
    ypos = np.arange(len(order))
    plt.barh(ypos, means.values, xerr=stds.values, color="#1C7293", ecolor="#B23A2E", capsize=4)
    plt.yticks(ypos, order)
    plt.gca().invert_yaxis()
    plt.xlabel("Average SHAP importance rank across subsamples (lower = more important)")
    plt.title(f"SHAP global-ranking stability (mean Spearman = {corrs.mean():.3f})")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "shap_stability_global.png", dpi=150, bbox_inches="tight")
    plt.close()

    out(f"\nSaved figure -> {FIG_DIR / 'shap_stability_global.png'}")
    TXT_OUT.parent.mkdir(parents=True, exist_ok=True)
    TXT_OUT.write_text("\n".join(lines))
    out(f"Saved summary -> {TXT_OUT}")


if __name__ == "__main__":
    main()
