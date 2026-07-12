"""
tier1_rules.py  —  Phase 4, Tier 1 (rule-based baseline)

A deterministic detector: flag a session as anomalous if it trips ANY rule.
No machine learning. This is the comparison floor the ML tiers must beat.

Rules (thresholds in config.yaml):
  R1 API burst       api_calls_per_min  >= api_burst_per_minute
  R2 High error rate error_rate         >= error_rate_threshold
  R3 IAM escalation  iam_escalation_flag == 1
  R4 Region spread   n_regions          >= region_threshold
  R5 IP rotation     n_source_ips       >= ip_threshold

Output: data/processed/tier1_scored.parquet + baseline precision/recall/F1.
Run:  python src/tier1_rules.py
"""

from pathlib import Path
import yaml
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

ROOT = Path(__file__).resolve().parent.parent
LABELLED = ROOT / "data" / "processed" / "labelled.parquet"
OUT = ROOT / "data" / "processed" / "tier1_scored.parquet"
CONFIG = ROOT / "config.yaml"


def load_cfg():
    with open(CONFIG) as f:
        return yaml.safe_load(f)


def apply_rules(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    t = cfg["tier1_rules"]
    burst = t.get("api_burst_per_minute", 50)
    err = t.get("error_rate_threshold", 0.30)
    region_thr = t.get("region_threshold", 4)
    ip_thr = t.get("ip_threshold", 10)

    df = df.copy()
    r1 = df["api_calls_per_min"] >= burst
    r2 = df["error_rate"] >= err
    r3 = df["iam_escalation_flag"] == 1
    r4 = df["n_regions"] >= region_thr
    r5 = df["n_source_ips"] >= ip_thr

    df["r1_burst"] = r1.astype(int)
    df["r2_errors"] = r2.astype(int)
    df["r3_escalation"] = r3.astype(int)
    df["r4_regions"] = r4.astype(int)
    df["r5_ips"] = r5.astype(int)

    df["rule_flag"] = (r1 | r2 | r3 | r4 | r5).astype(int)
    return df


def evaluate(df: pd.DataFrame) -> None:
    ev = df[df["label"].isin([0, 1])]
    y_true = ev["label"]
    y_pred = ev["rule_flag"]

    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    print("Tier 1 (rule-based baseline) — evaluated on labelled sessions")
    print(f"  sessions scored : {len(ev):,}")
    print(f"  precision       : {p:.3f}")
    print(f"  recall          : {r:.3f}")
    print(f"  F1              : {f:.3f}")
    print()
    print("  confusion matrix")
    print(f"    true normal : TN={tn:6,}  FP={fp:6,}")
    print(f"    true attack : FN={fn:6,}  TP={tp:6,}")
    print()
    print("  rule hit counts (how often each rule fired):")
    for col in ["r1_burst", "r2_errors", "r3_escalation", "r4_regions", "r5_ips"]:
        print(f"    {col:14s}: {int(df[col].sum()):,}")


def main() -> None:
    cfg = load_cfg()
    df = pd.read_parquet(LABELLED)
    df = apply_rules(df, cfg)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    evaluate(df)
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
