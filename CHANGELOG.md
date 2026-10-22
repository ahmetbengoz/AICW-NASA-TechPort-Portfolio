# Changelog

## v3.0.0 - independent shortlist and structural-baseline audit

- Reframed the contribution as a retrospective empirical audit, not a new weighting algorithm.
- Added nested training-only importance extraction and independent candidate-cohort shortlists.
- Added gain-only and headroom linear baselines, gain-deletion ablation, direct prediction and conventional weighting comparators.
- Re-estimated category rarity and criterion scales within the relevant training sets.
- Evaluated 10%, 20% and 30% shortlists with fractional boundary ties across random, programme and strictly start-date-separated splits.
- Added 100 label-randomisation controls per outcome and explicit noncausal interpretation.
- Moved standardisation inside Elastic Net inner validation; updated affected results.
- Explained the fixed-universe MARCOS/weighted-additive rank equivalence.
- Removed truncated-PNG tolerance and introduced strict atomic opaque image exports.
- Added invariant tests, deterministic SVG metadata and a complete v3 reproduction entry point.
- Verified and corrected reference metadata; added relevant 2025–2026 work without claiming the first general importance-audit protocol.
- No new Zenodo version DOI is claimed until a release has actually been archived.

## v2.0.0 - leakage-aware comparative revision

- Repositioned the study from a new-AI-method claim to an outcome-conditioned proxy-importance benchmarking protocol.
- Restricted the primary analysis to 201 completed, complete-case projects.
- Added a target-normalised outcome for 196 projects.
- Replaced the single fitted Random Forest with repeated cross-validation of Elastic Net, Random Forest, Extra Trees and histogram gradient boosting.
- Replaced in-sample impurity importance with held-out permutation importance and fold-level stability intervals.
- Added leakage, programme-grouped and chronological diagnostics.
- Corrected the MARCOS implementation and added explicit FUCOM, MEREC, standard-deviation and equal-weight comparison profiles.
- Added learner, outcome, ranking-method and Monte Carlo weight-perturbation robustness analyses.
- Rebuilt all publication figures at 600 dpi with editable SVG counterparts.
- Added pinned environments, a data dictionary, a run manifest, a CI workflow and one-command instructions.
- Preserved the v1.0 Zenodo archive as a legacy record instead of overwriting it.
