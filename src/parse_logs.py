"""
parse_logs.py  —  Phase 2, step 1

Reads all the gzipped CloudTrail JSON files in data/raw/flaws_cloudtrail_logs/,
extracts the fields we care about from every event, and saves one clean table
to data/processed/events.parquet.

Run:
    python src/parse_logs.py
"""

import gzip
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw" / "flaws_cloudtrail_logs"
OUT_FILE = ROOT / "data" / "processed" / "events.parquet"


def get(d, *keys, default=None):
    """Safely dig into nested dicts; return default if any key is missing."""
    cur = d
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur


def flatten_event(e: dict) -> dict:
    """Pull the fields we want out of one raw CloudTrail event into a flat row."""
    return {
        "eventTime": e.get("eventTime"),
        "eventName": e.get("eventName"),
        "eventSource": e.get("eventSource"),
        "awsRegion": e.get("awsRegion"),
        "sourceIPAddress": e.get("sourceIPAddress"),
        "userAgent": e.get("userAgent"),
        "eventType": e.get("eventType"),
        "errorCode": e.get("errorCode"),
        "errorMessage": e.get("errorMessage"),
        "identityType": get(e, "userIdentity", "type"),
        "identityArn": get(e, "userIdentity", "arn"),
        "identityAccountId": get(e, "userIdentity", "accountId"),
        "identityUserName": get(e, "userIdentity", "userName"),
        "accessKeyId": get(e, "userIdentity", "accessKeyId"),
        "mfaAuthenticated": get(
            e, "userIdentity", "sessionContext", "attributes", "mfaAuthenticated"
        ),
        "eventID": e.get("eventID"),
    }


def main() -> None:
    files = sorted(RAW_DIR.glob("*.json.gz"))
    if not files:
        raise SystemExit(f"No .json.gz files found in {RAW_DIR}")

    print(f"Found {len(files)} files. Parsing ...")
    rows = []
    for fp in tqdm(files, desc="files"):
        with gzip.open(fp, "rt") as f:
            records = json.load(f).get("Records", [])
        for e in records:
            rows.append(flatten_event(e))

    df = pd.DataFrame(rows)
    df["eventTime"] = pd.to_datetime(df["eventTime"], errors="coerce", utc=True)
    df = df.sort_values("eventTime").reset_index(drop=True)

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_FILE, index=False)

    print(f"\nDone. {len(df):,} events -> {OUT_FILE}")
    print(f"Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
