"""
features.py  —  Phase 2, step 2

Turns the per-event table (events.parquet) into a per-session feature table.
A "session" = one identity's activity on one calendar day (per user, per day).

Features per session:
  api_call_count, api_calls_per_min, api_diversity, error_rate,
  write_read_ratio, n_source_ips, n_regions, night_fraction, iam_escalation_flag

Output: data/processed/features.parquet
Run:    python src/features.py
"""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
EVENTS = ROOT / "data" / "processed" / "events.parquet"
OUT = ROOT / "data" / "processed" / "features.parquet"

WRITE_PREFIXES = (
    "Create", "Delete", "Put", "Update", "Attach", "Detach",
    "Modify", "Run", "Terminate", "Add", "Remove", "Associate",
    "Disassociate", "Authorize", "Revoke", "Set", "Start", "Stop",
)

ESCALATION_APIS = {
    "AttachRolePolicy", "AttachUserPolicy", "PutUserPolicy", "PutRolePolicy",
    "CreateAccessKey", "AssumeRole", "CreateUser", "CreateRole",
    "UpdateAssumeRolePolicy", "AddUserToGroup",
}


def is_write(name: str) -> bool:
    return isinstance(name, str) and name.startswith(WRITE_PREFIXES)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["eventTime"] = pd.to_datetime(df["eventTime"], utc=True)

    df["session_date"] = df["eventTime"].dt.date
    df["actor"] = df["identityArn"].fillna(df["identityType"]).fillna("unknown")

    df["is_error"] = df["errorCode"].notna()
    df["is_write"] = df["eventName"].apply(is_write)
    df["hour"] = df["eventTime"].dt.hour
    df["is_night"] = df["hour"].between(0, 5)

    groups = df.groupby(["actor", "session_date"], sort=False)

    feats = groups.agg(
        api_call_count=("eventName", "size"),
        api_diversity=("eventName", "nunique"),
        n_source_ips=("sourceIPAddress", "nunique"),
        n_regions=("awsRegion", "nunique"),
        error_rate=("is_error", "mean"),
        write_count=("is_write", "sum"),
        night_fraction=("is_night", "mean"),
        t_start=("eventTime", "min"),
        t_end=("eventTime", "max"),
    ).reset_index()

    span_min = (feats["t_end"] - feats["t_start"]).dt.total_seconds() / 60.0
    feats["api_calls_per_min"] = feats["api_call_count"] / span_min.replace(0, 1)

    read_count = feats["api_call_count"] - feats["write_count"]
    feats["write_read_ratio"] = feats["write_count"] / read_count.replace(0, 1)

    esc = (
        df.assign(is_esc=df["eventName"].isin(ESCALATION_APIS))
        .groupby(["actor", "session_date"])["is_esc"]
        .max()
        .reset_index(name="iam_escalation_flag")
    )
    feats = feats.merge(esc, on=["actor", "session_date"], how="left")
    feats["iam_escalation_flag"] = feats["iam_escalation_flag"].astype(int)

    cols = [
        "actor", "session_date",
        "api_call_count", "api_calls_per_min", "api_diversity",
        "error_rate", "write_read_ratio",
        "n_source_ips", "n_regions", "night_fraction",
        "iam_escalation_flag",
    ]
    return feats[cols]


def main() -> None:
    df = pd.read_parquet(EVENTS)
    feats = build_features(df)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(OUT, index=False)
    print(f"Done. {len(feats):,} sessions -> {OUT}")
    print(f"Features: {[c for c in feats.columns if c not in ('actor','session_date')]}")


if __name__ == "__main__":
    main()
