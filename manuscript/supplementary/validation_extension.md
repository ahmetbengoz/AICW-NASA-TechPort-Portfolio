# Independent shortlist evaluation supplement

## Question and estimand

The extension tests whether the conversion from outcome-predictive reliance to
an unsigned weighted score preserves useful outcome ordering among untouched
candidate projects. Observed shortlist lift is the selected outcome mean minus
the candidate-cohort mean. It is not financial return, welfare or a causal
effect of project funding.

## Nested construction

Each of the 25 random, five programme-separated and one strict-start-date outer
splits contains a three-fold inner importance experiment. Random Forest has
200 trees, minimum leaf size two and fixed seeds. Ten permutations per feature
are evaluated on each inner validation fold. The three signed loss-increase
vectors are averaged before clipping and normalisation. An equal fallback is
flagged if every average is nonpositive.

Rarity counts are estimated only from the relevant training subset. An unseen
category receives reciprocal count one. Criterion benefit maxima and positive
cost minima are frozen from outer training data; candidates are not used to
rescale one another. Outer outcomes do not enter fitting or coefficient
extraction. Programme-separated inner folds also exclude overlapping programmes.

The chronological cutoff is 1 October 2021. All records starting on the cutoff
are held out together, producing 157/44 progress and 153/43 attainment
training/test cases. Earlier-starting training projects are not required to
have completed by the cutoff, and historical field versions are absent.
Consequently this is transport, not a historical investment-policy backtest.

## Comparators and ties

Coefficients are Equal, Entropy, CRITIC, SD and MEREC, all trained without
candidate outcomes. Additional policies are planned gain, initial TRL, direct
Random Forest prediction, and the learned WASPAS and MARCOS scores. Entropy,
CRITIC and SD use direction-adjusted training min–max values. Constant columns
receive zero dispersion/information; an all-zero profile falls back to equal.
MEREC retains the disclosed 1e-12 floor rather than claiming zero-floor invariance.

At a 10%, 20% or 30% boundary, the requested count is rounded upward. If scores
tie, remaining places are allocated fractionally across every tied candidate.
Thus tie results are expectations and cannot exploit record order, project ID
or observed outcomes. Cohort means are equally weighted, not population
estimates; programme folds have unequal sizes.

## Prediction and falsification

Prediction comparators are the training mean, a planned-gain-only linear model,
an initial-plus-gain linear model, a Random Forest without gain and the full
forest. Repeat metrics assemble complete out-of-fold predictions. A separate
prespecified five-fold Random Forest experiment refits the model for 100
label permutations. The corrected tail proportion is `(1 + exceedances)/101`.
It tests model signal, not individual-feature significance or preference validity.

The frozen-universe standard MARCOS utility is a positive constant times the
weighted normalised row sum. The corresponding test verifies identical order;
its agreement with another partly additive method is not independent validation.

## Files and interpretation

`outputs/validation_extension/shortlist_summary.csv` contains cohort means.
`shortlist_fold_results.csv` retains fraction, lift, oracle gap and discounted
ranking gain. Detailed predictions and split memberships are regenerated and
included in `outputs/complete_results.zip`. `nested_proxy_weights.csv` records
each inner-trained outer coefficient vector and fallback flag.

The current article keeps negative findings and positive chronological exceptions
visible. No single method is declared a universally better screening policy.
Run `python scripts/reproduce_all.py` for every analytical result and invariant.
