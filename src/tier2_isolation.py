"""
tier2_isolation.py  —  Phase 4, Tier 2 (unsupervised detection)

Trains Isolation Forest (scikit-learn) and Extended Isolation Forest (isotree)
on the session features. Both are UNSUPERVISED: they never see the labels
during training. Labels are used ONLY afterwards to score the detectors.

Output: data/processed/tier2_scored.parquet + metrics for IF and EIF.
Run:    python src/tier2_isolation.py
"""

from pathlib import Path
import yaml
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
LABELLED = ROOT / "data" / "processed" / "labelled.parquet"
OUT = ROOT / "data" / "processed" / "tier2_scored.parquet"
CONFIG = ROOT / "config.yaml"

FEATURE_COLS = [
    "api_call_count", "api_calls_per_min", "api_diversity",
    "error_rate", "write_read_ratio",
    "n_source_ips", "n_regions", "night_fraction",
    "iam_escalation_flag",
]


def load_cfg():
    with open(CONFIG) as f:
        return yaml.safe_load(f)


def report(name, y_true, y_pred, score):
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, score)
    except Exception:
        auc = float("nan")
    print(f"  {name}")
    print(f"    precision : {p:.3f}")
    print(f"    recall    : {r:.3f}")
    print(f"    F1        : {f:.3f}")
    print(f"    AUROC     : {auc:.3f}")
    return f


def main() -> None:
    cfg = load_cfg()
    p2 = cfg.get("tier2_isolation_forest", {})
    n_estimators = p2.get("n_estimators", 200)
    random_state = p2.get("random_state", 42)
    contamination = p2.get("contamination", "auto")

    df = pd.read_parquet(LABELLED)

    X = df[FEATURE_COLS].copy()
    for c in ["api_call_count", "api_calls_per_min", "api_diversity",
              "n_source_ips", "n_regions", "write_read_ratio"]:
        X[c] = np.log1p(X[c])
    Xs = StandardScaler().fit_transform(X)

    # --- Standard Isolation Forest ---
    iforest = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    iforest.fit(Xs)
    if_score = -iforest.decision_function(Xs)
    if_pred = (iforest.predict(Xs) == -1).astype(int)
    df["if_score"] = if_score
    df["if_flag"] = if_pred

    # --- Extended Isolation Forest (isotree) ---
    eif_ok = True
    try:
        from isotree import IsolationForest as ExtIF
        eif = ExtIF(ntrees=n_estimators, ndim=2, random_seed=random_state)
        eif.fit(Xs)
        eif_raw = eif.predict(Xs)
        df["eif_score"] = eif_raw
        thr = np.quantile(eif_raw, 1 - if_pred.mean())
        df["eif_flag"] = (eif_raw >= thr).astype(int)
    except Exception as e:
        eif_ok = False
        print(f"[warn] Extended IF unavailable ({e}); skipping EIF.\n")

    ev = df[df["label"].isin([0, 1])]
    y = ev["label"]

    print("Tier 2 (unsupervised) — evaluated on labelled sessions")
    print(f"  sessions scored : {len(ev):,}\n")
    report("Isolation Forest", y, ev["if_flag"], ev["if_score"])
    if eif_ok:
        print()
        report("Extended Isolation Forest", y, ev["eif_flag"], ev["eif_score"])

    print("\n  (Tier 1 baseline F1 was 0.377 — compare above.)")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
