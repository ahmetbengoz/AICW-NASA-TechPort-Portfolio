# Learning outcome-conditioned criterion importance for technology portfolio screening

This repository is the reproducibility package for a retrospective NASA TechPort study of supervised-learning-derived proxy-importance weights and their propagation through multi-criteria decision-making (MCDM) rankings.

The revised study does **not** present Random Forest impurity importance as a new artificial-intelligence algorithm, a preference weight, or a causal effect. It compares Elastic Net, Random Forest, Extra Trees and histogram gradient boosting under identical repeated cross-validation splits; derives non-negative unit-sum profiles from held-out permutation importance; and tests sensitivity to learner choice, outcome definition, aggregation method and bounded weight perturbations.

## Main empirical design

- Frozen source snapshot: 299 released NASA TechPort project records.
- Primary sample: 201 completed projects with complete baseline and outcome fields.
- Primary outcome: realised TRL progress (`current TRL - initial TRL`).
- Secondary outcome: realised progress divided by planned TRL gain for the 196 projects with positive planned gain.
- Plausibly baseline-available predictors: initial TRL, planned duration, planned TRL gain, taxonomy level, portfolio taxonomy rarity and number of other organisations.
- Learners: Elastic Net, Random Forest, Extra Trees and histogram gradient boosting; mean prediction is retained as a performance-only baseline.
- Validation: five repeats of five-fold cross-validation, plus programme-grouped and chronological transport diagnostics.
- Importance: permutation importance on each held-out outer fold, clipped at zero and normalised only after out-of-sample evaluation.
- Downstream ranking: corrected WASPAS and MARCOS implementations.
- Robustness: learner and outcome agreement, conventional profiles, cross-method agreement and 5,000 bounded +/-20% weight perturbations.

## Reproduce the complete analysis

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python scripts/run_analysis.py
```

The command rewrites every analytical CSV in `outputs/`, every publication figure in `figures/`, and `outputs/run_manifest.json`, which records seeds, software versions and SHA-256 hashes.

## Repository structure

```text
data/raw/                         frozen TechPort snapshot and clean API URL index
data/data_dictionary.csv          field definitions, timing and analytical roles
scripts/run_analysis.py           single-command deterministic pipeline
notebooks/                        notebook entry point for interactive reproduction
outputs/                          generated samples, models, weights, rankings and diagnostics
figures/                          generated 600-dpi PNG and editable SVG figures
manuscript/supplementary/         methodological documentation
REPRODUCIBILITY.md                audit trail and verification instructions
ZENODO_RELEASE.md                 versioned archival-release procedure
```

Running the pipeline creates `outputs/full_results.zip`, which contains the complete
generated CSV set, including the two large deterministic draw/ranking files excluded
as loose files from version control. It also regenerates the high-resolution PNG
figures; the scalable SVG counterparts and all expected hashes remain versioned.

## Important interpretation boundary

Held-out permutation importance measures the predictive reliance of a fitted learner under a specified outcome and data distribution. It does not by itself identify stakeholder preferences, benefit/cost direction, causal importance or a universally transferable weighting rule. The MCDM directions in this repository are declared modelling assumptions used to demonstrate downstream propagation.

## Data source and licence

The frozen snapshot was derived from the public [NASA TechPort API](https://techport.nasa.gov/help/api). Repository code is released under the MIT License. NASA source records remain subject to the terms of the source platform.

## Archived versions

- Legacy v1.0 archive: [Zenodo 10.5281/zenodo.18913422](https://doi.org/10.5281/zenodo.18913422).
- The v1.0 archive documents the superseded single-Random-Forest analysis and must not be used to reproduce the revised manuscript.
- The next GitHub release will create a new versioned Zenodo record while preserving the v1.0 record.

## Citation

Use the metadata in `CITATION.cff`. Replace the provisional manuscript fields with the final journal citation after publication.
