# Reproducibility and forensic audit

## Legacy audit decision: REVISION REQUIRED

The v1.0 archive preserved the historical analysis, but it was not internally submission-ready. The revised pipeline corrects the following material discrepancies.

1. `data/processed/decision_matrix.csv` contained 299 rows and omitted taxonomy rarity, whereas the legacy model outputs were produced from 214 complete cases.
2. `dataset_summary.csv` reported 299 complete cases even though the notebook produced 214 when rerun from the frozen raw file.
3. The notebook depended on files in its current working directory and could not run from the repository root as documented.
4. `api_urls.csv` contained thousands of blank semicolon-padded rows rather than a clean project index.
5. The legacy Random Forest result used in-sample impurity importance without repeated out-of-sample model comparison.
6. Removing current TRL did not remove current-state dependence because the retained remaining-gap variable still equalled target TRL minus current TRL.
7. The legacy MARCOS score used the mean of two utility ratios and min-max values close to zero, producing numerically unstable scores rather than the standard MARCOS utility function.
8. The README duplicated major sections and did not identify the exact working directory or generated outputs.

## Frozen-data verification

Run:

```bash
python scripts/run_analysis.py
```

Expected terminal result:

```json
{"status": "ok", "primary_n": 201, "secondary_n": 196}
```

The exact raw-data SHA-256 and every output hash are recorded in `outputs/run_manifest.json`. A second run with the same environment and seed must produce identical CSV hashes.

## Analytical safeguards

- A single `RepeatedKFold` object defines identical outer splits for all learners.
- Elastic Net tuning occurs inside each outer training set.
- Predictive metrics are calculated from complete out-of-fold predictions for each repeat.
- Permutation importance is computed only on the held-out outer fold with negative MAE as the scoring rule.
- Negative importance estimates are clipped to zero before unit-sum normalisation; the frequency of clipping is reported.
- Importance magnitudes are labelled outcome-conditioned proxy weights, not preferences or causal effects.
- Benefit/cost directions are stored separately from the learned importance magnitudes.
- Random seeds, sample sizes, package versions and output hashes are recorded automatically.

## Additional diagnostics

- `leakage_diagnostic.csv` contrasts the legacy current-state specification, the incomplete current-TRL-only deletion and the leakage-aware baseline specification.
- `generalisation_diagnostics.csv` reports programme-grouped five-fold results and an earliest-80%/latest-20% chronological transport diagnostic.
- `outcome_weight_agreement.csv` and `outcome_ranking_sensitivity.csv` quantify the consequences of changing the outcome from raw progress to target-normalised attainment.
- `monte_carlo_weight_perturbation.csv` and `rank_acceptability.csv` contain all 5,000 bounded perturbation draws.
- `full_results.zip` is the deterministic compressed archive of every generated CSV; it preserves the large draw-level and project-level files without bloating the Git tree.

## Scope limitation

The repository contains one cross-sectional TechPort snapshot, not historical versioned records. Fields described as plausibly baseline-available may have been updated on the live platform after project initiation. The study is therefore retrospective and explanatory; the temporal split is a transport diagnostic, not evidence of a deployable prospective forecasting system.
