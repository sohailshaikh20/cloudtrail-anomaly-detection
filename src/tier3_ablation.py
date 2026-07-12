"""
tier3_ablation.py  —  Phase 4, ablation study

Tests whether Tier 3's performance depends on the iam_escalation_flag feature,
which we suspect may proxy the identity-based labels.

Trains TWO identical XGBoost models on the SAME train/test split:
  A) full         : all 9 features
  B) no_escalation: the 8 behavioural features, escalation flag removed

Run:  python src/tier3_ablation.py
"""

from pathlib import Path
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
LABELLED = ROOT / "data" / "processed" / "labelled.parquet"
CONFIG = ROOT / "config.yaml"

ALL_FEATURES = [
    "api_call_count", "api_calls_per_min", "api_diversity",
    "error_rate", "write_read_ratio",
    "n_source_ips", "n_regions", "night_fraction",
    "iam_escalation_flag",
]
NO_ESC = [c for c in ALL_FEATURES if c != "iam_escalation_flag"]


def load_cfg():
    with open(CONFIG) as f:
        return yaml.safe_load(f)


def train_eval(name, features, data, cfg):
    p3 = cfg.get("tier3_xgboost", {})
    ev = cfg.get("evaluation", {})
    test_size = ev.get("test_size", 0.30)
    rs = ev.get("random_state", 42)

    X = data[features]
    y = data["label"].astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=rs, stratify=y
    )
    n_pos = int((y_tr == 1).sum()); n_neg = int((y_tr == 0).sum())
    spw = (n_neg / n_pos) if n_pos else 1.0

    model = XGBClassifier(
        n_estimators=p3.get("n_estimators", 400),
        max_depth=p3.get("max_depth", 6),
        learning_rate=p3.get("learning_rate", 0.05),
        random_state=rs, scale_pos_weight=spw,
        eval_metric="logloss", n_jobs=-1,
    )
    model.fit(X_tr, y_tr)
    proba = model.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "name": name,
        "n_features": len(features),
        "precision": precision_score(y_te, pred, zero_division=0),
        "recall": recall_score(y_te, pred, zero_division=0),
        "f1": f1_score(y_te, pred, zero_division=0),
        "auroc": roc_auc_score(y_te, proba),
    }


def main() -> None:
    cfg = load_cfg()
    df = pd.read_parquet(LABELLED)
    data = df[df["label"].isin([0, 1])].copy()

    a = train_eval("A) full (9 features)", ALL_FEATURES, data, cfg)
    b = train_eval("B) no escalation flag (8)", NO_ESC, data, cfg)

    print("Tier 3 ablation — does performance depend on iam_escalation_flag?\n")
    hdr = f"  {'model':28s} {'feat':>4s} {'prec':>7s} {'recall':>7s} {'F1':>7s} {'AUROC':>7s}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for m in (a, b):
        print(f"  {m['name']:28s} {m['n_features']:>4d} "
              f"{m['precision']:>7.3f} {m['recall']:>7.3f} {m['f1']:>7.3f} {m['auroc']:>7.3f}")

    d_f1 = a["f1"] - b["f1"]
    d_auc = a["auroc"] - b["auroc"]
    print(f"\n  drop when flag removed:  F1 -{d_f1:.3f}   AUROC -{d_auc:.3f}\n")

    if d_f1 < 0.05:
        print("  => Small drop: behavioural features detect attacks on their own.")
        print("     Detection is robust; the flag was a convenience, not a crutch.")
    elif d_f1 < 0.20:
        print("  => Moderate drop: the flag helps but is not the whole story.")
    else:
        print("  => Large drop: the flag carried most of the signal — likely a")
        print("     proxy for the identity-based labels (label-leakage effect).")


if __name__ == "__main__":
    main()
