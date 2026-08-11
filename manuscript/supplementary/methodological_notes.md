# Methodological notes

## Interpretation

The revised workflow estimates outcome-conditioned predictive-reliance profiles. A normalised held-out permutation profile is called a proxy weight because it can be propagated through an MCDM calculation, not because it is equivalent to a stakeholder preference, marginal utility, causal effect or universal criterion weight.

## Samples and outcomes

The frozen dataset contains 299 released records. The primary sample contains 201 completed projects with complete TRL, date, taxonomy and organisation fields. The primary outcome is current TRL minus initial TRL. The secondary sample contains 196 projects with positive planned TRL gain; its outcome is realised progress divided by planned gain and clipped to the zero-one interval.

The legacy current-state specification is retained only as a leakage diagnostic. The revised ex-ante-oriented predictor set excludes both current TRL and remaining TRL gap because the latter is also a function of current TRL.

## Learners and validation

Elastic Net, Random Forest, Extra Trees and histogram gradient boosting are evaluated on identical five-times-repeated five-fold partitions. Elastic Net hyperparameters are selected within each outer training fold. A mean-prediction model provides a performance floor but does not generate a proxy weight.

For each outer fold, predictive performance is recorded as R-squared, MAE and RMSE. Complete out-of-fold predictions are assembled for each repeat, so the reported repeat-level metrics refer to all observations rather than averages of unstable small-fold R-squared values. Paired Wilcoxon tests use the five repeat-level results and Holm adjustment; they are treated as descriptive because of the small number of repeats.

## Held-out permutation profiles

Permutation importance is computed on the outer test fold using the increase in MAE after a feature is permuted. Each fold uses ten permutations. Negative sampling estimates are set to zero. The remaining magnitudes are divided by their sum. If all magnitudes are non-positive, an equal profile is used and the event is flagged. The released output reports raw importance, clipping frequency, normalised fold profiles, means, standard deviations and percentile intervals.

## MCDM propagation

Predictive importance supplies magnitude only. Separate illustrative directions treat initial TRL, planned TRL gain, taxonomy level, taxonomy rarity and other-organisation count as benefits and planned duration as a cost. These directions are modelling assumptions, not learned facts or NASA policy.

WASPAS uses an equal mixture of weighted sum and weighted product terms after ratio normalisation. MARCOS uses ideal and anti-ideal utility degrees and the standard utility function. Conventional comparisons include equal weights, standard-deviation weights, MEREC and a fully solved FUCOM profile based on declared synthetic priority ratios. The FUCOM profile is not presented as stakeholder elicitation.

## Robustness

Robustness outputs cover learner-specific profile agreement, learner-specific ranking agreement, outcome-specific profile and ranking agreement, WASPAS-MARCOS agreement, programme-separated validation, chronological transport and 5,000 independent perturbations. Each perturbation multiplies every element of the cross-learner median profile by a uniform factor in [0.8, 1.2] before renormalisation.

## Reproducibility boundary

TechPort records are current snapshots. The repository does not contain historical field versions proving that the date, target and organisation variables were unchanged after project start. For this reason, the analysis is explicitly retrospective. It demonstrates a transparent audit and benchmarking protocol, not a validated prospective NASA selection system.
