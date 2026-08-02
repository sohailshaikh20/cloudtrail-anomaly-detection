"""
tier3_leakage_check.py  —  Phase 4 / robustness

Tests whether Tier 3's high scores are inflated by IDENTITY LEAKAGE.

Concern: labels come from actor identity, and the same actor can appear in BOTH
train and test of a random split. The model might learn to recognise identities
rather than attack behaviour.

Test: same model, same data, two splits:
  A) RANDOM  - actors may appear in both train and test (current setup)
  B) GROUPED - each actor entirely in train OR test, never both

Small drop A->B  => generalises to unseen identities => no leakage.
Large drop A->B  => identity leakage was inflating scores.

Run:  python src/tier3_leakage_check.py
"""

from pathlib import Path
import yaml
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
LABELLED = ROOT / "data" / "processed" / "labelled.parquet"
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


def fit_eval(X_tr, X_te, y_tr, y_te, cfg):
    p3 = cfg.get("tier3_xgboost", {})
    rs = cfg.get("evaluation", {}).get("random_state", 42)
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
        "precision": precision_score(y_te, pred, zero_division=0),
        "recall": recall_score(y_te, pred, zero_division=0),
        "f1": f1_score(y_te, pred, zero_division=0),
        "auroc": roc_auc_score(y_te, proba) if len(set(y_te)) > 1 else float("nan"),
        "n_train": len(X_tr), "n_test": len(X_te),
    }


def main() -> None:
    cfg = load_cfg()
    df = pd.read_parquet(LABELLED)
    data = df[df["label"].isin([0, 1])].copy().reset_index(drop=True)
    X = data[FEATURE_COLS]
    y = data["label"].astype(int)
    groups = data["actor"]
    rs = cfg.get("evaluation", {}).get("random_state", 42)
    test_size = cfg.get("evaluation", {}).get("test_size", 0.30)

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_size, random_state=rs, stratify=y
    )
    a = fit_eval(Xtr, Xte, ytr, yte, cfg)

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=rs)
    tr_idx, te_idx = next(gss.split(X, y, groups))
    b = fit_eval(X.iloc[tr_idx], X.iloc[te_idx], y.iloc[tr_idx], y.iloc[te_idx], cfg)

    tr_actors = set(groups.iloc[tr_idx]); te_actors = set(groups.iloc[te_idx])
    overlap = tr_actors & te_actors

    print("Tier 3 leakage check — random split vs identity-grouped split\n")
    print(f"  total actors: {groups.nunique()}   "
          f"train actors: {len(tr_actors)}   test actors: {len(te_actors)}   "
          f"overlap: {len(overlap)}  (grouped split must be 0)\n")
    hdr = f"  {'split':16s} {'prec':>7s} {'recall':>7s} {'F1':>7s} {'AUROC':>7s}  {'test n':>7s}"
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    print(f"  {'A) random':16s} {a['precision']:>7.3f} {a['recall']:>7.3f} {a['f1']:>7.3f} {a['auroc']:>7.3f}  {a['n_test']:>7d}")
    print(f"  {'B) grouped':16s} {b['precision']:>7.3f} {b['recall']:>7.3f} {b['f1']:>7.3f} {b['auroc']:>7.3f}  {b['n_test']:>7d}")

    d_f1 = a["f1"] - b["f1"]
    d_auc = a["auroc"] - b["auroc"]
    print(f"\n  drop (random -> grouped):  F1 -{d_f1:.3f}   AUROC -{d_auc:.3f}\n")

    if d_f1 < 0.10:
        print("  => Small drop: detection generalises to UNSEEN identities.")
        print("     Identity leakage is NOT inflating the results. (Strong result.)")
    elif d_f1 < 0.30:
        print("  => Moderate drop: some identity dependence, but real signal remains.")
        print("     Report both numbers honestly; the grouped score is the fair one.")
    else:
        print("  => Large drop: scores were substantially inflated by identity leakage.")
        print("     The grouped-split score is the honest performance to report.")


if __name__ == "__main__":
    main()
