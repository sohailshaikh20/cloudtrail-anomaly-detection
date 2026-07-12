"""
tier3_xgboost.py  —  Phase 4, Tier 3 (supervised classification)

Trains an XGBoost classifier on the labelled sessions. Unlike Tiers 1-2, this
tier LEARNS from the labels. To keep the evaluation honest:

  * only the 9 behavioural features are used (never the actor identity);
  * data is split into train/test; metrics are on the held-out test set;
  * class imbalance (70% attack) is handled with scale_pos_weight.

Output: models/tier3_xgboost.json, data/processed/tier3_scored.parquet, metrics.
Run:    python src/tier3_xgboost.py
"""

from pathlib import Path
import yaml
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix)
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
LABELLED = ROOT / "data" / "processed" / "labelled.parquet"
OUT = ROOT / "data" / "processed" / "tier3_scored.parquet"
MODEL_DIR = ROOT / "models"
MODEL_FILE = MODEL_DIR / "tier3_xgboost.json"
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


def main() -> None:
    cfg = load_cfg()
    p3 = cfg.get("tier3_xgboost", {})
    ev_cfg = cfg.get("evaluation", {})
    test_size = ev_cfg.get("test_size", 0.30)
    random_state = ev_cfg.get("random_state", 42)

    df = pd.read_parquet(LABELLED)
    data = df[df["label"].isin([0, 1])].copy()
    X = data[FEATURE_COLS]
    y = data["label"].astype(int)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    n_pos = int((y_tr == 1).sum())
    n_neg = int((y_tr == 0).sum())
    scale_pos_weight = (n_neg / n_pos) if n_pos else 1.0

    model = XGBClassifier(
        n_estimators=p3.get("n_estimators", 400),
        max_depth=p3.get("max_depth", 6),
        learning_rate=p3.get("learning_rate", 0.05),
        random_state=random_state,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        n_jobs=-1,
    )
    model.fit(X_tr, y_tr)

    proba = model.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)

    p = precision_score(y_te, pred, zero_division=0)
    r = recall_score(y_te, pred, zero_division=0)
    f = f1_score(y_te, pred, zero_division=0)
    auc = roc_auc_score(y_te, proba)
    tn, fp, fn, tp = confusion_matrix(y_te, pred, labels=[0, 1]).ravel()

    print("Tier 3 (XGBoost, supervised) — held-out test set")
    print(f"  train size : {len(X_tr):,}   test size : {len(X_te):,}")
    print(f"  scale_pos_weight : {scale_pos_weight:.3f}\n")
    print(f"  precision : {p:.3f}")
    print(f"  recall    : {r:.3f}")
    print(f"  F1        : {f:.3f}")
    print(f"  AUROC     : {auc:.3f}\n")
    print("  confusion matrix (test)")
    print(f"    true normal : TN={tn:5,}  FP={fp:5,}")
    print(f"    true attack : FN={fn:5,}  TP={tp:5,}\n")

    imp = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print("  top features (XGBoost gain importance):")
    for name, val in imp.head(5).items():
        print(f"    {name:20s}: {val:.3f}")

    print("\n  (Tier 1 F1 0.377 | Tier 2 IF AUROC 0.80, EIF 0.85 — compare above.)")
    if f > 0.98 or auc > 0.99:
        print("\n  [check] Scores are very high — verify the model isn't just")
        print("          separating identities via a proxy feature (discuss in write-up).")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(MODEL_FILE)
    data["xgb_proba"] = model.predict_proba(X)[:, 1]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(OUT, index=False)
    print(f"\nSaved model -> {MODEL_FILE}")
    print(f"Saved scores -> {OUT}")


if __name__ == "__main__":
    main()
