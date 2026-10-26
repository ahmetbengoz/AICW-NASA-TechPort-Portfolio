# Version 3 archival release

## Version 3.0.1 patch release

Prepared tag: `v3.0.1`, targeting the current `main` branch.

Release title: `v3.0.1 — Archival cleanup and citation metadata update`

This maintenance release removes an internal working document from the source
tree and updates release and citation metadata. The dataset, analytical code,
numerical results and figures are unchanged from v3.0.0. Zenodo assigns a new
version DOI after publication; `10.5281/zenodo.23667988` remains the DOI of
v3.0.0 and must not be assigned to v3.0.1.

## Published version 3.0.0

Published tag: `v3.0.0`

Release title: `v3.0.0 — Nested shortlist validation and structural baseline audit`

Zenodo version DOI: [10.5281/zenodo.23667988](https://doi.org/10.5281/zenodo.23667988).

GitHub release: [v3.0.0](https://github.com/ahmetbengoz/AICW-NASA-TechPort-Portfolio/releases/tag/v3.0.0).

Released on 26 October 2026 from merged scientific revision commit
`0bfe2a5383390add5f3ed58286bec4a8dae8812a`. Zenodo reports this version as
published with the DOI above. Subsequent documentation changes on `main` do
not alter the immutable source release.

## Public release description

Version 3.0.0 provides the frozen NASA TechPort batch, complete descriptive and
nested validation code, structural ablations, independent shortlist outcomes,
training-only transformations, label-randomisation controls and seven figure
sources for “Auditing learned criterion weights in technology portfolio
screening”. It shows why predictive importance and locally stable weights do
not establish improved screening. The study is retrospective and does not infer
stakeholder utility, causal effects or NASA policy.

Changes include nested importance extraction, gain-only and gain-deletion
baselines, conventional comparator profiles, direct-prediction ranking,
fractional cutoff ties, random/programme/start-date transport, strict image
exports and invariant tests. Earlier analyses are retained in history.

## Reproduction and archive contents

1. Cite the version DOI `10.5281/zenodo.23667988` for these analyses. Earlier versions remain separately identifiable.
2. Run `python scripts/reproduce_all.py` on the released source. Retain `outputs/revision_manifest.json`.
3. The GitHub source archive contains the scientific source, compact results and SVGs. Large CSVs and PNGs are regenerated. Do not claim they are all loose files in the source archive.
4. If adding generated assets to Zenodo, attach `outputs/complete_results.zip` and the seven verified figures, not the private journal-submission package.
5. Use software, version 3.0.0, MIT for code, English, and the two creators in `CITATION.cff`.
6. The DOI identifies v3.0.0. Version 1 (`10.5281/zenodo.18913422`) and version 2 (`10.5281/zenodo.21894218`) do not contain the nested shortlist extension.

Never publish cover letters, author-contact title pages, Turkish internal audit
documents or the complete private submission ZIP as release assets.
