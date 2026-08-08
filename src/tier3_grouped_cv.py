"""
tier3_grouped_cv.py  —  Phase 4 / robust leakage evaluation

Hardens the leakage finding using CROSS-VALIDATION instead of a single split.
Runs k folds for BOTH schemes and reports mean +/- std:
  * STRATIFIED k-fold (random)  - actors may appear in both train and test.
  * GROUPED   k-fold (honest)   - each actor confined to one fold (unseen in test).

Outputs: results/grouped_cv_summary.txt
Run:     python src/tier3_grouped_cv.py
"""

from pathlib import Path
import yaml
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
LABELLED = ROOT / "data" / "processed" / "labelled.parquet"
CONFIG = ROOT / "config.yaml"
TXT_OUT = ROOT / "results" / "grouped_cv_summary.txt"

FEATURE_COLS = [
    "api_call_count", "api_calls_per_min", "api_diversity",
    "error_rate", "write_read_ratio",
    "n_source_ips", "n_regions", "night_fraction",
    "iam_escalation_flag",
]


def load_cfg():
    with open(CONFIG) as f:
        return yaml.safe_load(f)


def make_model(cfg, y_tr):
    p3 = cfg.get("tier3_xgboost", {})
    rs = cfg.get("evaluation", {}).get("random_state", 42)
    n_pos = int((y_tr == 1).sum()); n_neg = int((y_tr == 0).sum())
    spw = (n_neg / n_pos) if n_pos else 1.0
    return XGBClassifier(
        n_estimators=p3.get("n_estimators", 400), max_depth=p3.get("max_depth", 6),
        learning_rate=p3.get("learning_rate", 0.05), random_state=rs,
        scale_pos_weight=spw, eval_metric="logloss", n_jobs=-1,
    )


def score_fold(model, Xte, yte):
    proba = model.predict_proba(Xte)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "precision": precision_score(yte, pred, zero_division=0),
        "recall": recall_score(yte, pred, zero_division=0),
        "f1": f1_score(yte, pred, zero_division=0),
        "auroc": roc_auc_score(yte, proba) if len(set(yte)) > 1 else float("nan"),
    }


def run_cv(splitter, X, y, groups, cfg, use_groups):
    rows = []
    it = splitter.split(X, y, groups) if use_groups else splitter.split(X, y)
    for tr, te in it:
        m = make_model(cfg, y.iloc[tr])
        m.fit(X.iloc[tr], y.iloc[tr])
        rows.append(score_fold(m, X.iloc[te], y.iloc[te]))
    return pd.DataFrame(rows)


def summarise(name, dfres):
    line = f"  {name:20s}"
    for k in ["precision", "recall", "f1", "auroc"]:
        line += f"  {k[:4]}={dfres[k].mean():.3f}\u00B1{dfres[k].std():.3f}"
    return line


def main():
    cfg = load_cfg()
    df = pd.read_parquet(LABELLED)
    data = df[df["label"].isin([0, 1])].copy().reset_index(drop=True)
    X = data[FEATURE_COLS]; y = data["label"].astype(int); groups = data["actor"]
    rs = cfg.get("evaluation", {}).get("random_state", 42)

    n_actors = groups.nunique()
    k = min(5, n_actors)

    lines = []
    def out(s=""):
        print(s); lines.append(s)

    out(f"Tier 3 grouped cross-validation  ({k}-fold, {n_actors} actors)\n")
    out("  Reporting mean \u00B1 std across folds.\n")

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=rs)
    rand = run_cv(skf, X, y, groups, cfg, use_groups=False)

    gkf = GroupKFold(n_splits=k)
    grp = run_cv(gkf, X, y, groups, cfg, use_groups=True)

    out(summarise("A) random  (k-fold)", rand))
    out(summarise("B) grouped (k-fold)", grp))

    d_f1 = rand["f1"].mean() - grp["f1"].mean()
    d_auc = rand["auroc"].mean() - grp["auroc"].mean()
    out(f"\n  mean drop random -> grouped:  F1 -{d_f1:.3f}   AUROC -{d_auc:.3f}")

    out("\n  Interpretation:")
    if d_f1 > 0.3:
        out("   Large, consistent drop across folds => identity leakage is ROBUST,")
        out("   not a single-split artefact. Report grouped numbers as honest")
        out("   supervised performance; the gap is the methodological finding.")
    elif d_f1 > 0.1:
        out("   Moderate drop => some identity dependence; report both, prefer grouped.")
    else:
        out("   Small drop => supervised detection generalises to unseen identities.")

    TXT_OUT.parent.mkdir(parents=True, exist_ok=True)
    TXT_OUT.write_text("\n".join(lines))
    out(f"\nSaved -> {TXT_OUT}")


if __name__ == "__main__":
    main()
