# Version 3 archival release plan

Proposed tag: `v3.0.0`

Proposed title: `v3.0.0 — Nested shortlist validation and structural baseline audit`

Target the merged revision commit, not the old v2 commit. Confirm successful
`reproduce-analysis` CI before creating a release. This file is a plan; it does
not assert that a GitHub release or new Zenodo version already exists.

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

## Archive checks

1. Use the existing concept DOI `10.5281/zenodo.18913421`; do not replace its older versions.
2. Run `python scripts/reproduce_all.py` on the released source. Retain `outputs/revision_manifest.json`.
3. The GitHub source archive contains the scientific source, compact results and SVGs. Large CSVs and PNGs are regenerated. Do not claim they are all loose files in the source archive.
4. If adding generated assets to Zenodo, attach `outputs/complete_results.zip` and the seven verified figures, not the private journal-submission package.
5. Use software, version 3.0.0, MIT for code, English, and the two creators in `CITATION.cff`.
6. Insert a new version DOI only after Zenodo returns it and its contents have been checked. Until then cite the exact repository commit.

Never publish cover letters, author-contact title pages, Turkish internal audit
documents or the complete private submission ZIP as release assets.
