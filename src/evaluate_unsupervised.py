"""
evaluate_unsupervised.py  —  Phase 6 (evaluation figures)

Publication-quality ROC and Precision-Recall curves for the two unsupervised
detectors (Isolation Forest, Extended IF), plotted together. Leakage-free
headline results.

Reads data/processed/tier2_scored.parquet (if_score, eif_score, label).
Outputs:
  results/figures/roc_unsupervised.png
  results/figures/pr_unsupervised.png
  results/unsupervised_eval_summary.txt
Run:
  python src/evaluate_unsupervised.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_curve, auc, precision_recall_curve,
                             average_precision_score)

ROOT = Path(__file__).resolve().parent.parent
SCORED = ROOT / "data" / "processed" / "tier2_scored.parquet"
FIG_DIR = ROOT / "results" / "figures"
TXT_OUT = ROOT / "results" / "unsupervised_eval_summary.txt"

TEAL = "#1C7293"; SEA = "#0E7C66"; GREY = "#9AA6B5"


def main():
    df = pd.read_parquet(SCORED)
    ev = df[df["label"].isin([0, 1])].copy()
    y = ev["label"].values

    models = []
    if "if_score" in ev:  models.append(("Isolation Forest", ev["if_score"].values, TEAL))
    if "eif_score" in ev: models.append(("Extended IF", ev["eif_score"].values, SEA))
    if not models:
        raise SystemExit("No if_score/eif_score columns found in tier2_scored.parquet")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    lines = []
    def out(s=""):
        print(s); lines.append(s)

    out("Unsupervised evaluation (leakage-free) — evaluated on labelled sessions")
    out(f"  sessions scored: {len(ev):,}\n")

    plt.figure(figsize=(6.4, 5.2))
    for name, score, col in models:
        fpr, tpr, _ = roc_curve(y, score)
        a = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=col, lw=2.2, label=f"{name} (AUROC = {a:.3f})")
        out(f"  {name:20s} AUROC = {a:.3f}")
    plt.plot([0, 1], [0, 1], "--", color=GREY, lw=1.2, label="Random (0.5)")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC — Unsupervised Detectors")
    plt.legend(loc="lower right", frameon=False)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "roc_unsupervised.png", dpi=150, bbox_inches="tight")
    plt.close()

    out("")
    baseline = (y == 1).mean()
    plt.figure(figsize=(6.4, 5.2))
    for name, score, col in models:
        prec, rec, _ = precision_recall_curve(y, score)
        ap = average_precision_score(y, score)
        plt.plot(rec, prec, color=col, lw=2.2, label=f"{name} (AP = {ap:.3f})")
        out(f"  {name:20s} Average Precision = {ap:.3f}")
    plt.axhline(baseline, ls="--", color=GREY, lw=1.2, label=f"Baseline ({baseline:.2f})")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision-Recall — Unsupervised Detectors")
    plt.legend(loc="lower left", frameon=False)
    plt.ylim(0, 1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "pr_unsupervised.png", dpi=150, bbox_inches="tight")
    plt.close()

    out(f"\n  PR baseline (attack prevalence) = {baseline:.3f}")
    out(f"\nSaved:")
    out(f"  {FIG_DIR / 'roc_unsupervised.png'}")
    out(f"  {FIG_DIR / 'pr_unsupervised.png'}")
    TXT_OUT.write_text("\n".join(lines))
    out(f"  {TXT_OUT}")


if __name__ == "__main__":
    main()
