"""
tier2_significance.py  —  statistical significance of EIF vs IF

Runs both detectors across many random seeds, reports mean +/- std for AUROC
and AP, and tests whether the per-seed AUROC difference is significant.

Output: results/significance_summary.txt
Run:    python src/tier2_significance.py
"""

from pathlib import Path
import yaml
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
LABELLED = ROOT / "data" / "processed" / "labelled.parquet"
CONFIG = ROOT / "config.yaml"
TXT_OUT = ROOT / "results" / "significance_summary.txt"

FEATURE_COLS = [
    "api_call_count", "api_calls_per_min", "api_diversity",
    "error_rate", "write_read_ratio",
    "n_source_ips", "n_regions", "night_fraction",
    "iam_escalation_flag",
]
LOG_COLS = ["api_call_count", "api_calls_per_min", "api_diversity",
            "n_source_ips", "n_regions", "write_read_ratio"]

N_SEEDS = 20


def load_cfg():
    with open(CONFIG) as f:
        return yaml.safe_load(f)


def ci95(vals):
    n = len(vals)
    if n < 2:
        return float("nan")
    return stats.t.ppf(0.975, n - 1) * np.std(vals, ddof=1) / np.sqrt(n)


def main():
    cfg = load_cfg()
    p2 = cfg.get("tier2_isolation_forest", {})
    n_estimators = p2.get("n_estimators", 200)

    df = pd.read_parquet(LABELLED)
    ev = df[df["label"].isin([0, 1])].copy()
    y = ev["label"].values

    X = ev[FEATURE_COLS].copy()
    for c in LOG_COLS:
        X[c] = np.log1p(X[c])
    Xs = StandardScaler().fit_transform(X)

    if_auc, if_ap, eif_auc, eif_ap = [], [], [], []
    have_eif = True

    for seed in range(N_SEEDS):
        iforest = IsolationForest(n_estimators=n_estimators,
                                  random_state=seed, n_jobs=-1)
        iforest.fit(Xs)
        s_if = -iforest.decision_function(Xs)
        if_auc.append(roc_auc_score(y, s_if))
        if_ap.append(average_precision_score(y, s_if))

        try:
            from isotree import IsolationForest as ExtIF
            eif = ExtIF(ntrees=n_estimators, ndim=2, random_seed=seed)
            eif.fit(Xs)
            s_eif = eif.predict(Xs)
            eif_auc.append(roc_auc_score(y, s_eif))
            eif_ap.append(average_precision_score(y, s_eif))
        except Exception as e:
            have_eif = False
            print(f"[warn] EIF unavailable: {e}")
            break

    lines = []
    def out(s=""):
        print(s); lines.append(s)

    out(f"Statistical significance: IF vs EIF across {N_SEEDS} random seeds\n")
    if_auc = np.array(if_auc); if_ap = np.array(if_ap)
    out(f"  Isolation Forest  AUROC = {if_auc.mean():.4f} +/- {if_auc.std(ddof=1):.4f}"
        f"  (95% CI +/-{ci95(if_auc):.4f})")
    out(f"  Isolation Forest  AP    = {if_ap.mean():.4f} +/- {if_ap.std(ddof=1):.4f}")

    if have_eif:
        eif_auc = np.array(eif_auc); eif_ap = np.array(eif_ap)
        out(f"  Extended IF       AUROC = {eif_auc.mean():.4f} +/- {eif_auc.std(ddof=1):.4f}"
            f"  (95% CI +/-{ci95(eif_auc):.4f})")
        out(f"  Extended IF       AP    = {eif_ap.mean():.4f} +/- {eif_ap.std(ddof=1):.4f}")

        diff = eif_auc - if_auc
        out(f"\n  mean AUROC difference (EIF - IF) = {diff.mean():+.4f}")

        t_stat, t_p = stats.ttest_rel(eif_auc, if_auc)
        try:
            w_stat, w_p = stats.wilcoxon(eif_auc, if_auc)
        except Exception:
            w_p = float("nan")
        d = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else float("inf")

        out(f"  paired t-test:      t = {t_stat:.3f}, p = {t_p:.2e}")
        out(f"  Wilcoxon test:      p = {w_p:.2e}")
        out(f"  Cohen's d (paired): {d:.2f}")

        out("")
        if t_p < 0.05:
            out("  => The EIF advantage over IF is STATISTICALLY SIGNIFICANT (p < 0.05),")
            out("     consistent across seeds. The single-run result is confirmed.")
        else:
            out("  => The difference is NOT statistically significant at p < 0.05;")
            out("     the claim should be softened accordingly.")

    TXT_OUT.parent.mkdir(parents=True, exist_ok=True)
    TXT_OUT.write_text("\n".join(lines))
    out(f"\nSaved -> {TXT_OUT}")


if __name__ == "__main__":
    main()
