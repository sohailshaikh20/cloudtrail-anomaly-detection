# Explainable and Adaptive AI-Driven Anomaly Detection for AWS CloudTrail Logs

Capstone thesis - MSc Computer Science (Adaptive Cybersecurity), University of Galway.
Supervisor: Dr. Malika Bendechache.

A three-tier anomaly-detection pipeline for AWS CloudTrail audit logs, augmented
with SHAP-based explainability and evaluated on the public flaws.cloud dataset.

## Pipeline

```text
Raw CloudTrail JSON
   ↓
Feature engineering            → src/features.py
   ↓
Tier 1  Rule-based baseline    → src/tier1_rules.py
   ↓
Tier 2  Isolation Forest / EIF → src/tier2_isolation.py
   ↓
Tier 3  XGBoost classifier     → src/tier3_xgboost.py
   ↓
SHAP explanation layer         → src/explain_shap.py
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/download_data.py
```

## Project structure

| Path          | Purpose                                    |
|---------------|--------------------------------------------|
| `src/`        | The reproducible pipeline (Python modules) |
| `notebooks/`  | Exploration and figures only               |
| `data/`       | Dataset (git-ignored, never committed)     |
| `results/`    | Metrics and figures                        |
| `config.yaml` | All paths and parameters                   |
| `tests/`      | Sanity tests                               |

## Dataset

flaws.cloud CloudTrail logs by Scott Piper (Summit Route): ~1.94M events,
Feb 2017 - Oct 2020, 9,402 unique IPs, 1,242 distinct AWS APIs. The only known
public real-world CloudTrail dataset. Not committed to the repo — run
`src/download_data.py` to fetch it.

## Progress

- [x] Phase 0 — Environment & repo skeleton
- [x] Phase 1 — Data acquisition & validation
- [x] Phase 2 — Parsing & feature engineering
- [x] Phase 3 — Heuristic labelling
- [x] Phase 4 — Three-tier detection
- [x] Phase 5 — SHAP explainability
- [x] Phase 6 — Evaluation & concept-drift experiment

## Use of AI

AI tools were used as a productivity aid for scaffolding and boilerplate, in line
with University of Galway guidelines. All design decisions, feature engineering,
and analysis are the author's own.
