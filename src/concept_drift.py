"""
concept_drift.py  —  Phase 6 (concept-drift experiment)

Tests whether the unsupervised detector degrades over time. Splits by TIME using
config's drift_split_date: train on sessions before it, test on sessions after.
Uses the label-free unsupervised detector so this measures DRIFT, not leakage.

Outputs:
  results/figures/concept_drift.png
  results/concept_drift_summary.txt
Run:
  python src/concept_drift.py
"""

from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parent.parent
LABELLED = ROOT / "data" / "processed" / "labelled.parquet"
CONFIG = ROOT / "config.yaml"
FIG_DIR = ROOT / "results" / "figures"
TXT_OUT = ROOT / "results" / "concept_drift_summary.txt"

FEATURE_COLS = [
    "api_call_count", "api_calls_per_min", "api_diversity",
    "error_rate", "write_read_ratio",
    "n_source_ips", "n_regions", "night_fraction",
    "iam_escalation_flag",
]
LOG_COLS = ["api_call_count", "api_calls_per_min", "api_diversity",
            "n_source_ips", "n_regions", "write_read_ratio"]


def load_cfg():
    with open(CONFIG) as f:
        return yaml.safe_load(f)


def prep(X):
    X = X.copy()
    for c in LOG_COLS:
        X[c] = np.log1p(X[c])
    return X


def fit_score(train_X, eval_X, eval_y, cfg):
    p2 = cfg.get("tier2_isolation_forest", {})
    scaler = StandardScaler().fit(train_X)
    iforest = IsolationForest(
        n_estimators=p2.get("n_estimators", 200),
        contamination=p2.get("contamination", "auto"),
        random_state=p2.get("random_state", 42), n_jobs=-1,
    )
    iforest.fit(scaler.transform(train_X))
    score = -iforest.decision_function(scaler.transform(eval_X))
    return (roc_auc_score(eval_y, score) if len(set(eval_y)) > 1 else float("nan"),
            average_precision_score(eval_y, score))


def main():
    cfg = load_cfg()
    split_date = pd.to_datetime(cfg.get("evaluation", {}).get("drift_split_date", "2019-01-01")).date()

    df = pd.read_parquet(LABELLED)
    df = df[df["label"].isin([0, 1])].copy()
    df["session_date"] = pd.to_datetime(df["session_date"]).dt.date

    before = df[df["session_date"] < split_date]
    after = df[df["session_date"] >= split_date]

    lines = []
    def out(s=""):
        print(s); lines.append(s)

    out(f"Concept-drift experiment (temporal split at {split_date})\n")
    out(f"  train period (before): {len(before):,} sessions "
        f"({(before['label']==1).mean()*100:.0f}% attack)")
    out(f"  test  period (after) : {len(after):,} sessions "
        f"({(after['label']==1).mean()*100:.0f}% attack)\n")

    if len(before) < 50 or len(after) < 50:
        out("  [warn] one period is too small for a reliable comparison.")

    Xb, yb = prep(before[FEATURE_COLS]), before["label"].values
    Xa, ya = prep(after[FEATURE_COLS]), after["label"].values

    auc_in, ap_in = fit_score(Xb, Xb, yb, cfg)
    auc_out, ap_out = fit_score(Xb, Xa, ya, cfg)

    out(f"  {'evaluation':28s} {'AUROC':>7s} {'AP':>7s}")
    out("  " + "-" * 46)
    out(f"  {'fit before / score before':28s} {auc_in:>7.3f} {ap_in:>7.3f}")
    out(f"  {'fit before / score after':28s} {auc_out:>7.3f} {ap_out:>7.3f}")

    d_auc = auc_in - auc_out
    out(f"\n  drift gap (AUROC): {d_auc:+.3f}")
    if abs(d_auc) < 0.05:
        out("  => Detector is temporally ROBUST: little concept drift.")
    elif d_auc >= 0.05:
        out("  => Performance DROPS on later data: concept drift is present,")
        out("     motivating periodic re-fitting / the adaptive component.")
    else:
        out("  => Performance is higher on later data (attack mix shifted).")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4.6))
    labels = ["fit before\nscore before", "fit before\nscore after"]
    aucs = [auc_in, auc_out]; aps = [ap_in, ap_out]
    x = np.arange(2); w = 0.35
    plt.bar(x - w/2, aucs, w, label="AUROC", color="#1C7293")
    plt.bar(x + w/2, aps, w, label="Avg Precision", color="#0E7C66")
    plt.xticks(x, labels); plt.ylim(0, 1.0); plt.ylabel("Score")
    plt.title(f"Concept drift - temporal split at {split_date}")
    plt.legend(frameon=False)
    for i, v in enumerate(aucs): plt.text(i - w/2, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    for i, v in enumerate(aps): plt.text(i + w/2, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "concept_drift.png", dpi=150, bbox_inches="tight")
    plt.close()

    out(f"\nSaved -> {FIG_DIR / 'concept_drift.png'}")
    TXT_OUT.write_text("\n".join(lines))
    out(f"Saved -> {TXT_OUT}")


if __name__ == "__main__":
    main()
