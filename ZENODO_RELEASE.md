# Zenodo v2 release plan

1. Freeze and merge the reviewed `v2.0.0` repository state.
2. Create the GitHub release `v2.0.0` with the title **Leakage-aware comparative learning and MCDM revision**.
3. Confirm that the existing GitHub-Zenodo integration creates a new version under concept DOI `10.5281/zenodo.18913421` rather than replacing record `10.5281/zenodo.18913422`.
4. Verify that the Zenodo archive contains `scripts/run_analysis.py`, the frozen raw snapshot, `data_dictionary.csv`, pinned dependencies, all generated outputs, all final figures and `run_manifest.json`.
5. Use version `2.0.0`, software as the resource type, MIT as the code licence, and the two manuscript authors as creators.
6. Describe v1.0 as superseded for manuscript reproduction while retaining it for provenance.
7. Insert the newly minted version DOI into the unblinded title page, cover letter and post-review data-availability statement.
8. Download the Zenodo ZIP, verify its checksum and run the analysis once from the extracted archive before final journal submission.

## Proposed description

This release contains the frozen NASA TechPort snapshot, leakage-aware repeated-cross-validation pipeline, held-out permutation proxy weights, corrected WASPAS and MARCOS analyses, robustness outputs and publication figures for the manuscript “Learning outcome-conditioned criterion importance for technology portfolio screening”. Version 2.0 supersedes the v1.0 analytical results but preserves the earlier record for provenance.
