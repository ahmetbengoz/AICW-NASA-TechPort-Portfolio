# Auditing learned criterion weights in technology portfolio screening

Reproducibility materials for an empirical audit of the conversion from
predictive feature importance to multi-criteria project scores. Version 3.0.0
separates predictive signal, importance stability and independent shortlist
outcomes. It does not propose a new AI or MCDM algorithm.

## Main finding

The frozen NASA TechPort batch contains 299 released records, 201 completed
complete-case projects and 196 projects with positive planned gain for the
attainment outcome. It is not a representative census of the TechPort inventory.

Under training-only preprocessing, a planned-gain-only linear baseline obtains
random-CV raw-progress R² **0.576**, compared with **0.543** for the six-feature
Random Forest and **−0.050** when gain is removed. The forest has lower MAE than
the linear baseline, so no model is best under every loss.

In independently evaluated random candidate cohorts, selecting the top 20% by
planned gain gives mean raw-progress lift **1.259** TRL levels. Nested learned
WASPAS and MARCOS scores give **1.234** and **1.254**. Under attainment, learned
scores give lift **−0.032** and **−0.029**, while direct forest prediction gives
**0.118**. Programme and chronological results differ; no universally superior
screening policy is established.

Predictive importance, a reproducible coefficient vector and high local rank
stability therefore do not by themselves establish useful project selection.

## Reproduce version 3

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python scripts/reproduce_all.py
```

The command runs the descriptive analysis, nested extension, 100 label
permutations per outcome, invariant tests and image-integrity checks. Expect
several minutes, depending on hardware. The numerical environment is pinned.
Reference metadata can be independently refreshed with
`python scripts/audit_references.py`; that optional command needs internet.

The legacy descriptive component alone is `python scripts/run_analysis.py`.
It does **not** reproduce the independent shortlist results by itself.

## Two evidential layers

1. **Descriptive comparison:** Elastic Net, Random Forest, Extra Trees and
   histogram gradient boosting share five repeats of five-fold CV. Held-out
   permutation importance is propagated through WASPAS and MARCOS; outcome
   sensitivity and 5,000 local weight perturbations are reported.
2. **Independent shortlist evaluation:** each outer training pool contains a
   three-fold inner importance experiment. Candidate outcomes enter only after
   weights, category counts, scales and policies have been frozen. Random,
   programme-separated and strictly start-date-separated cohorts are evaluated.

Both layers are retrospective. The descriptive fixed-batch rarity feature and
the nested training-only rarity feature are deliberately distinguished. The
strict date split does not establish that training projects were completed or
their fields frozen at a historical decision date.

## Outputs

| Location | Contents |
| --- | --- |
| `data/raw/` | Frozen released records and API index |
| `scripts/reproduce_all.py` | Complete v3 entry point |
| `scripts/run_analysis.py` | Descriptive analysis with inner-pipeline Elastic Net tuning |
| `scripts/run_validation_extension.py` | Independent shortlists, ablations and label randomisation |
| `outputs/validation_extension/` | Nested profiles, cohort results, ablations and falsification tables |
| `outputs/revision_manifest.json` | SHA-256 hashes of all v3 analytical outputs and figures |
| `outputs/complete_results.zip` | Generated archive of every CSV, including unversioned detailed predictions |
| `figures/` | Seven SVG figure sources and regenerated opaque 600-DPI PNGs |
| `tests/` | Scientific and software invariants |
| `manuscript/final_blinded_manuscript.md` | Current curated article source; filename retained for compatibility |
| `manuscript/reference_audit.json` | Verified DOI metadata, including corrected author lists |

Large detailed predictions, split memberships and raster figures are generated
by the entry point rather than all stored as loose Git objects. Compact
scientific results and SVG sources are versioned. Private submission letters,
author-contact files and internal assessments are not part of public releases.

## Interpretation boundaries

- Importance is predictive reliance, not preference, signed effect or causal importance.
- Shortlist lift is an observed proxy-outcome diagnostic, not welfare, financial return or the effect of funding a project.
- Directions for taxonomy, maturity and organisation counts are illustrative assumptions, not NASA judgements.
- Repeated folds overlap. Reported partition variability is not independent-replication confidence.
- Fixed-universe MARCOS ordering is weighted-additive ordering; agreement with another partly additive rule is not independent validation.
- The code strictly rejects truncated PNGs and writes complete opaque images atomically; it never silently pads missing rows.

## Archival versions

The existing Zenodo concept is
[10.5281/zenodo.18913421](https://doi.org/10.5281/zenodo.18913421).
The original version
[10.5281/zenodo.18913422](https://doi.org/10.5281/zenodo.18913422)
and the v2 history do not contain this nested audit. Cite the exact v3 repository
commit until a new versioned archive is actually minted. No new DOI is guessed
or claimed. See [ZENODO_RELEASE.md](ZENODO_RELEASE.md).

Code is MIT-licensed. NASA source records retain their source terms. NASA did not
participate in the study and the rankings are not NASA recommendations.
