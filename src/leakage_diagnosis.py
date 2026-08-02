"""
leakage_diagnosis.py  —  which features leak identity, and can we recover?

(1) IDENTITY-PREDICTABILITY per feature (mutual information vs actor).
(2) RECOVERY TEST: re-run the identity-grouped split while progressively
    dropping the most identity-predictive features.

Run:  python src/leakage_diagnosis.py
"""

from pathlib import Path
import yaml
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, roc_auc_score, recall_score, precision_score
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


def grouped_eval(X, y, groups, cfg):
    p3 = cfg.get("tier3_xgboost", {})
    rs = cfg.get("evaluation", {}).get("random_state", 42)
    ts = cfg.get("evaluation", {}).get("test_size", 0.30)
    gss = GroupShuffleSplit(n_splits=1, test_size=ts, random_state=rs)
    tr, te = next(gss.split(X, y, groups))
    n_pos = int((y.iloc[tr] == 1).sum()); n_neg = int((y.iloc[tr] == 0).sum())
    spw = (n_neg / n_pos) if n_pos else 1.0
    m = XGBClassifier(
        n_estimators=p3.get("n_estimators", 400), max_depth=p3.get("max_depth", 6),
        learning_rate=p3.get("learning_rate", 0.05), random_state=rs,
        scale_pos_weight=spw, eval_metric="logloss", n_jobs=-1,
    )
    m.fit(X.iloc[tr], y.iloc[tr])
    proba = m.predict_proba(X.iloc[te])[:, 1]
    pred = (proba >= 0.5).astype(int)
    yte = y.iloc[te]
    return {
        "precision": precision_score(yte, pred, zero_division=0),
        "recall": recall_score(yte, pred, zero_division=0),
        "f1": f1_score(yte, pred, zero_division=0),
        "auroc": roc_auc_score(yte, proba) if len(set(yte)) > 1 else float("nan"),
    }


def main() -> None:
    cfg = load_cfg()
    df = pd.read_parquet(LABELLED)
    data = df[df["label"].isin([0, 1])].copy().reset_index(drop=True)
    X = data[FEATURE_COLS]
    y = data["label"].astype(int)
    groups = data["actor"]

    actor_id = LabelEncoder().fit_transform(groups)
    mi = mutual_info_classif(X, actor_id, random_state=42)
    mi_s = pd.Series(mi, index=FEATURE_COLS).sort_values(ascending=False)

    print("(1) How much each feature predicts IDENTITY (mutual information).")
    print("    Higher = more identity-specific = more leakage risk.\n")
    mx = mi_s.max() if mi_s.max() > 0 else 1
    for name, val in mi_s.items():
        bar = "#" * int(val / mx * 30)
        print(f"    {name:20s} {val:6.3f}  {bar}")

    print("\n(2) Grouped-split performance as we remove the most identity-predictive")
    print("    features (does honest performance recover?).\n")
    order = list(mi_s.index)
    print(f"    {'features kept':>14s}  {'dropped':22s} {'F1':>7s} {'AUROC':>7s} {'recall':>7s}")
    print("    " + "-" * 66)

    for k in range(0, len(order) - 1):
        dropped = order[:k]
        kept = [c for c in FEATURE_COLS if c not in dropped]
        res = grouped_eval(X[kept], y, groups, cfg)
        drop_lbl = ", ".join(d.replace("api_", "") for d in dropped) if dropped else "(none)"
        if len(drop_lbl) > 21:
            drop_lbl = drop_lbl[:19] + ".."
        print(f"    {len(kept):>14d}  {drop_lbl:22s} {res['f1']:>7.3f} {res['auroc']:>7.3f} {res['recall']:>7.3f}")

    print("\n  Interpretation:")
    print("   - If F1/AUROC RISE as leaky features are dropped -> removing identity-")
    print("     specific features reduces leakage; keep the cleaner feature set.")
    print("   - If they STAY LOW regardless -> the labels themselves are identity-bound;")
    print("     lean on unsupervised tiers + Stratus Red Team for honest supervised eval.")


if __name__ == "__main__":
    main()
