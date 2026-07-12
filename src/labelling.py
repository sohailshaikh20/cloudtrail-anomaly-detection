"""
labelling.py  —  Phase 3

Assigns a heuristic ground-truth label to each session, based on the ACTOR
identity (not on the model's features, to avoid circular reasoning).

Label meaning:
    1  = attack     (documented challenge / attacker identities)
    0  = normal     (documented legitimate owner, his tools, AWS services)
   -1  = excluded   (ambiguous identities we cannot confidently attribute;
                     kept in the data for unsupervised detection, but NOT
                     used for scoring or supervised training)

Grounding: Scott Piper's flaws.cloud release notes state the account's only
legitimate user was himself (mostly via root); 'backup' and 'Level6' are the
challenge IAM users; the 'flaws' role is used by EC2. Everything else is a mix
of ~10k external attackers and AWS service noise.

This is WEAK / HEURISTIC labelling and is declared as a limitation.

Output: data/processed/labelled.parquet
Run:    python src/labelling.py
"""

from pathlib import Path
import re
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FEATURES = ROOT / "data" / "processed" / "features.parquet"
OUT = ROOT / "data" / "processed" / "labelled.parquet"

ATTACK_PATTERNS = [
    r"user/backup",
    r"user/Level6",
    r"assumed-role/Level6",
    r"assumed-role/level6",
    r"assumed-role/level5",
    r"user/Level5",
    r"assumed-role/flaws",
]

NORMAL_PATTERNS = [
    r":root",
    r"user/piper",
    r"SummitRoute",
    r"SecurityMon?key",
    r"Cloudsploit",
    r"^AWSService$",
    r"^AWSAccount$",
    r"AWSServiceRoleFor",
    r"config-role",
    r"lambda_basic_execution",
]


def label_actor(actor: str) -> int:
    """Return 1 (attack), 0 (normal), or -1 (excluded/ambiguous)."""
    if not isinstance(actor, str):
        return -1
    for pat in ATTACK_PATTERNS:
        if re.search(pat, actor, flags=re.IGNORECASE):
            return 1
    for pat in NORMAL_PATTERNS:
        if re.search(pat, actor, flags=re.IGNORECASE):
            return 0
    return -1


def main() -> None:
    df = pd.read_parquet(FEATURES)
    df["label"] = df["actor"].apply(label_actor)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)

    n = len(df)
    counts = df["label"].value_counts().to_dict()
    n_attack = counts.get(1, 0)
    n_normal = counts.get(0, 0)
    n_excl = counts.get(-1, 0)
    labelled = n_attack + n_normal

    print(f"Done. {n:,} sessions -> {OUT}\n")
    print("Label breakdown:")
    print(f"  attack   (1): {n_attack:5,}")
    print(f"  normal   (0): {n_normal:5,}")
    print(f"  excluded (-1): {n_excl:5,}   (ambiguous; kept for detection, not scoring)")
    print()
    if labelled:
        print(f"Labelled set for evaluation: {labelled:,} sessions "
              f"({n_attack/labelled*100:.1f}% attack, {n_normal/labelled*100:.1f}% normal)")


if __name__ == "__main__":
    main()
