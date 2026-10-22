# Reproducibility and forensic audit

## Version 3 entry point

Run `python scripts/reproduce_all.py` for the current article. This invokes both
the descriptive component and the nested extension, runs tests and creates
`outputs/revision_manifest.json` and `outputs/complete_results.zip`.

The v2 entry point and audit below are preserved for provenance, but
`run_analysis.py` alone does not reproduce the v3 shortlist evidence. Current
article source and reference metadata are in `manuscript/`. The curated
manuscript is not automatically rewritten when numerical outputs are regenerated.

Nested weights use three inner training-pool folds per outer split. Outer
candidate outcomes never enter weight learning. Taxonomy counts and scales are
training-only; boundary ties use fractional allocation. Strict start-date
transport uses 1 October 2021 without splitting tied dates, but does not prove
historical field or outcome availability. Test-fold variability is descriptive.

The baseline raw snapshot hash is unchanged:
`904795730b4dd9df8a5d7e804729835ef852c646dae85fe85e66aab79e983d90`.

All PNGs are strict opaque exports with atomic file replacement. The previous
truncated-image-tolerance setting has been removed. SVG metadata are fixed for
deterministic re-exports. A source-only release does not contain all generated
large files until the pipeline has been run.

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
