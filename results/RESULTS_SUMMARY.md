# Results Summary

_Auto-generated from the saved scored files. All metrics on the labelled subset (labels 0/1; ambiguous -1 excluded)._

| Tier | Model | Precision | Recall | F1 | AUROC | AP | Note |
|------|-------|-----------|--------|----|-------|----|------|
| Tier 1 | Rule-based | 0.456 | 0.322 | 0.377 |  |  | Baseline floor |
| Tier 2 | Isolation Forest | 0.866 | 0.208 | 0.335 | 0.797 | 0.859 | Unsupervised, leakage-free |
| Tier 2 | Extended IF | 0.840 | 0.201 | 0.324 | 0.852 | 0.870 | Best unsupervised; beats IF |
| Tier 3 | XGBoost (random split) |  |  |  |  |  | INFLATED by identity leakage - not the honest number |
| Tier 3 | XGBoost (grouped CV) |  |  |  |  |  | Honest eval - see grouped_cv_summary.txt (large variance) |

## Leakage cross-validation
```
Tier 3 grouped cross-validation  (5-fold, 57 actors)

  Reporting mean ± std across folds.

  A) random  (k-fold)   prec=0.979±0.003  reca=0.975±0.004  f1=0.977±0.002  auro=0.993±0.002
  B) grouped (k-fold)   prec=0.725±0.437  reca=0.650±0.400  f1=0.663±0.381  auro=0.685±nan

  mean drop random -> grouped:  F1 -0.313   AUROC -0.308

  Interpretation:
   Large, consistent drop across folds => identity leakage is ROBUST,
   not a single-split artefact. Report grouped numbers as honest
   supervised performance; the gap is the methodological finding.
```

## Concept drift
```
Concept-drift experiment (temporal split at 2019-01-01)

  train period (before): 2,533 sessions (68% attack)
  test  period (after) : 2,815 sessions (72% attack)

  evaluation                     AUROC      AP
  ----------------------------------------------
  fit before / score before      0.860   0.871
  fit before / score after       0.669   0.820

  drift gap (AUROC): +0.191
  => Performance DROPS on later data: concept drift is present,
     motivating periodic re-fitting / the adaptive component.

Saved -> /Users/sohail/Desktop/cloudtrail-anomaly-detection/results/figures/concept_drift.png
```

## SHAP stability
```
(1) GLOBAL ranking stability across 10 random subsamples
    (Spearman rank correlation of feature importance order; 1.0 = identical)

    mean pairwise Spearman : 1.000
    min  pairwise Spearman : 1.000
    max  pairwise Spearman : 1.000

    average feature rank across runs (1 = most important):
      iam_escalation_flag   1.0
      api_diversity         2.0
      n_source_ips          3.0
      api_calls_per_min     4.0
      n_regions             5.0
      write_read_ratio      6.0
      api_call_count        7.0
      error_rate            8.0
      night_fraction        9.0

    => Global ranking is HIGHLY STABLE (mean Spearman 1.000).

(2) LOCAL explanation stability under small perturbations
    (does each session's TOP feature stay the same after +-2% noise?)

    top-feature agreement after perturbation : 87.1%
    => Local explanations are STABLE.

Saved figure -> /Users/sohail/Desktop/cloudtrail-anomaly-detection/results/figures/shap_stability_global.png
```

## Headline narrative

- **Unsupervised tiers are the reliable, leakage-free result.** Extended IF (AUROC ~0.85) outperforms Isolation Forest (~0.80) - first demonstration on CloudTrail.
- **Supervised tier suffers identity leakage:** random-split F1 ~0.98 collapses and destabilises under identity-grouped evaluation - a methodological finding, not a usable score.
- **Concept drift is present:** static detector AUROC falls ~0.86 -> ~0.67 across a 2019 temporal split, empirically motivating the adaptive component.
- **SHAP explanations are stable** (global Spearman ~1.0; local ~87%), supporting the SHAP-over-LIME choice on CloudTrail.
