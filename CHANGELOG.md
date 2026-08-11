# Changelog

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
