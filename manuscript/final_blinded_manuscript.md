# Learning outcome-conditioned criterion importance for technology portfolio screening

## Abstract

Criterion weights learned from historical outcomes can appear empirical while remaining sensitive to leakage, learner choice and outcome construction. This study develops a leakage-aware protocol for converting held-out permutation importance into non-negative proxy-weight profiles and tracing them through multi-criteria rankings. A frozen NASA TechPort snapshot yields 201 completed projects. Elastic Net, Random Forest, Extra Trees and histogram gradient boosting are compared using five-times-repeated five-fold cross-validation. Raw technology-readiness progress is predicted moderately well (mean R-squared 0.553-0.577), but planned readiness gain receives 0.746-0.961 of the learned weight. Under target-normalised attainment, nonlinear R-squared falls to 0.185-0.231 and outcome-to-outcome weight agreement is weak. Corresponding project rankings also change materially, whereas rankings are stable under bounded perturbations within a fixed outcome. The findings show that supervised importance can support transparent MCDM benchmarking, but cannot be interpreted as preference, causal importance or a stable decision rule without explicit outcome and timing audits.

Keywords: multi-criteria decision-making; criterion weighting; supervised learning; permutation importance; technology portfolio; NASA TechPort; data leakage; outcome sensitivity; ranking robustness; WASPAS; MARCOS; decision support

## 1 Introduction

Technology portfolios require project comparisons across maturity, ambition, duration, technological domain and organisational participation. Multi-criteria decision-making (MCDM) offers a transparent way to represent these dimensions, yet its outputs depend strongly on the criterion weights supplied to the aggregation model. Weighting is therefore not a preliminary detail. It defines which trade-offs can influence a recommendation and how strongly they do so (Cinelli et al., 2020; Greco et al., 2025).

Most weighting methods obtain information from either people or the decision matrix. Expert-oriented methods structure pairwise, ordinal or level-based judgements. Objective methods use variation, contrast, entropy or removal effects in the observed alternatives. Both families are useful, but neither automatically exploits the historical outcomes that accumulate in repeated project environments. This has encouraged integrations of machine learning and MCDM in which a learner is fitted to a recorded outcome and a feature-importance vector is normalised for later use as MCDM weights (Fu et al., 2021; Maghsoodi et al., 2023).

The apparent simplicity of this integration hides four distinct questions. First, is the learner evaluated out of sample, or is an in-sample importance vector treated as evidence? Second, were the predictors available at the decision point, or do they contain information recorded after the outcome began to develop? Third, does an importance magnitude identify a preference direction? Fourth, would a different defensible outcome or learner produce similar weights and downstream rankings? Unless these questions are answered, data-derived weights can give an impression of objectivity while embedding circularity, model dependence and unreported normative assumptions.

These risks are especially visible in technology-readiness data. Suppose realised progress is defined as current technology readiness level (TRL) minus initial TRL. Including current TRL as a predictor places part of the outcome algebra inside the feature set. Removing current TRL alone is also insufficient when another predictor is target TRL minus current TRL. Even a genuinely baseline-oriented specification can remain outcome-dependent: planned TRL gain is an informative bound on raw realised progress, whereas a target-normalised attainment outcome asks whether the intended gain was achieved. Those outcomes answer different questions and need not produce the same importance profile.

This study reconstructs a NASA TechPort application around that problem. It does not propose Random Forest feature importance plus normalisation as a new artificial-intelligence algorithm. Instead, it develops and demonstrates a reproducible protocol for outcome-conditioned proxy importance. Four learners are assessed on identical repeated cross-validation splits. Importance is calculated by permuting features only in held-out folds, negative sampling estimates are handled explicitly, and fold-specific profiles are carried into a stability analysis. Two outcome definitions, two MCDM ranking models, conventional comparison profiles, grouped and chronological diagnostics, and bounded weight perturbations are then used to trace which conclusions are stable and which are specification-dependent.

The study makes three contributions. First, it provides a leakage-audit structure that distinguishes current-state, incompletely reduced and plausibly baseline-oriented predictor sets. Second, it supplies a learner-neutral conversion from held-out predictive reliance to non-negative unit-sum proxy weights, while formally separating magnitude from benefit/cost direction. Third, it quantifies the propagation of learner and outcome choices into WASPAS and MARCOS rankings rather than evaluating predictive performance and MCDM outputs as disconnected stages.

The empirical result is deliberately qualified. Raw TRL progress is predictable in this snapshot, but much of that predictability is associated with planned TRL gain. When the outcome is changed to the fraction of planned gain attained, predictive performance weakens and the importance structure changes. The resulting evidence supports a retrospective benchmarking contribution. It does not establish NASA preferences, a causal technology-maturation model or a prospective project-selection system.

The remainder of the article reviews the methodological background, documents the forensic data audit, defines the comparative protocol, reports predictive and MCDM results, and discusses managerial use and limitations.

## 2 Background and research gap

### 2.1 Criterion weights and what they represent

Criterion weights can represent several different constructs. In multi-attribute value models they commonly express trade-off or scaling information under stated value functions. In outranking or distance-based procedures they can affect concordance, separation or contribution to an aggregate index. Expert-derived methods such as the Full Consistency Method (FUCOM) seek internally coherent preference information from ordered comparative priorities (Pamučar et al., 2018). Objective methods instead derive coefficients from observed data geometry. The Method based on the Removal Effects of Criteria (MEREC), for example, assigns greater weight when removing a criterion changes alternative-level performance more strongly (Keshavarz-Ghorabaee et al., 2021).

These interpretations are not interchangeable. Dispersion is not preference, predictive reliance is not marginal utility, and a coefficient that improves prediction is not automatically a statement that larger criterion values are desirable. Robust and stochastic MCDM research has long shown that rankings should be examined across plausible weight spaces rather than treated as mechanically certain (Tervonen et al., 2013; Corrente et al., 2017). The same discipline is needed when weights originate from a learner.

### 2.2 Supervised learning as an empirical weighting layer

Supervised learning offers a way to ask which recorded attributes help predict a defined historical outcome. Elastic Net provides a regularised linear reference that can handle correlated predictors and shrink weak effects (Zou and Hastie, 2005). Random Forest averages decorrelated trees and captures nonlinearities and interactions (Breiman, 2001). Extra Trees adds stronger randomisation to tree construction (Geurts et al., 2006). Gradient boosting builds an additive nonlinear predictor by sequentially improving residual fit (Friedman, 2001). Comparing these learners is important because a single model cannot reveal whether an importance profile is data-driven or architecture-driven.

Importance extraction adds another choice. Tree impurity importance is convenient but is calculated from the fitted training structure and can favour variables with more potential split points. Permutation importance instead measures the loss in predictive performance when a feature is disrupted (Breiman, 2001; Altmann et al., 2010). When calculated on held-out observations, it directly addresses whether a fitted model relied on the feature for unseen cases. It remains conditional on the learner, loss function, correlated-feature structure and observed distribution. Accordingly, the present study calls its normalised output a proxy-importance profile rather than a preference-weight estimate.

### 2.3 Leakage, validation and outcome dependence

Data leakage occurs when a model uses information unavailable at the intended prediction time or when preprocessing allows test information to influence training. In small datasets, a single train-test split also produces unstable performance estimates and can encourage selective interpretation. Nested or repeated validation helps separate model selection from performance assessment and exposes sampling variation (Varma and Simon, 2006). Here, identical repeated outer splits are used for every learner, and Elastic Net tuning occurs only within each outer training set.

Outcome definition is equally consequential. Raw progress rewards the absolute number of readiness levels gained. Target-normalised attainment measures the fraction of intended gain realised. A project planned to advance four levels and achieving three has greater raw progress but lower attainment than a project planned to advance one level and achieving it. Neither outcome is universally correct. The correct choice depends on the managerial question. Comparing them is therefore not an optional sensitivity exercise; it is part of construct validity.

The research gap lies at the intersection of these issues. Existing machine-learning-MCDM applications demonstrate useful integrations, but often report one learner, one feature-importance mechanism and one outcome. The downstream ranking is then presented without testing whether another defensible learning specification changes the inferred weights. A publishable empirical weighting study should evaluate predictive validity, importance stability, outcome dependence and ranking propagation in a single auditable design.

## 3 Data and forensic audit

### 3.1 Frozen TechPort snapshot

NASA TechPort is a public platform containing project descriptions, programme information, technology taxonomy fields, dates and TRL fields. The repository snapshot contains 299 released project records obtained through the TechPort application programming interface. The raw record is frozen for reproduction; the analysis does not query the live service. The public API is the data source, but NASA did not participate in the study and the resulting profiles are not NASA recommendations.

[[PAGE_BREAK]]

An audit of the legacy workflow identified material inconsistencies summarised in Table 1. The stored decision matrix had 299 rows and omitted taxonomy rarity, whereas the legacy model outputs corresponded to 214 complete cases. A stored summary also reported 299 complete cases. The notebook required files in an undocumented current working directory, and the API URL file contained thousands of blank, semicolon-padded rows. More importantly, the legacy analysis fitted one Random Forest on all observations, used impurity importance, and described a robustness model that removed current TRL while retaining a remaining-gap feature that still contained current TRL. The prior MARCOS implementation was also numerically unstable because near-zero anti-ideal sums generated extremely large scores.

[[TABLE_1]]

### 3.2 Sample construction

The revised primary sample retains projects recorded as completed and requires non-missing initial, current and target TRL; positive planned duration; technology taxonomy code and level; programme title; and other-organisation count. Projects with negative realised progress or negative planned gain would have been excluded, although none remained after the complete-case restriction. The resulting primary sample contains 201 projects. Realised progress ranges from zero to five TRL levels, with a mean of 1.672 and standard deviation of 1.132.

The secondary sample excludes five projects with zero planned TRL gain, leaving 196 observations. Its outcome is the realised progress divided by planned gain, restricted to the interval from zero to one. The mean target attainment is 0.815. This cap prevents over-attainment from dominating the comparison, although no retained current TRL exceeds target TRL in the frozen snapshot.

Complete-case analysis removes 98 of the 299 frozen records before the primary model. Missingness is therefore a limitation rather than a benign processing step. The package reports field-level missingness and included project identifiers so that readers can inspect selection effects.

### 3.3 Outcomes, predictors and timing classes

The primary outcome for project i is:

EQ: y_i^(P) = TRL_current,i - TRL_initial,i.

For projects with positive planned gain, the secondary outcome is:

EQ: y_i^(A) = min{1, (TRL_current,i - TRL_initial,i) / (TRL_target,i - TRL_initial,i)}.

Six predictors are used: initial TRL, planned duration in months, planned TRL gain, primary taxonomy level, taxonomy rarity and number of other organisations. Planned duration equals the recorded end date minus start date divided by 30.4375. Planned TRL gain equals target TRL minus initial TRL. Taxonomy rarity is the reciprocal frequency of the primary technology code in the frozen 299-project portfolio.

These predictors are classified as plausibly baseline-available rather than proven baseline measurements. TechPort supplies a current snapshot, not a historical version of every field. The repository cannot establish that target dates, target TRL or organisation counts were never revised after project initiation. This is why the study is described as retrospective. A chronological holdout tests transport across project start dates, but it cannot repair unavailable field histories.

[[PAGE_BREAK]]

Table 2 distinguishes outcome timing, predictor timing and illustrative MCDM direction. The latter is deliberately separate. Initial TRL, planned gain, taxonomy level, taxonomy rarity and other-organisation count are treated as benefits, while duration is treated as a cost. These directions demonstrate propagation under a declared screening logic; they are not learned from the outcome and should not be read as NASA value judgements. In particular, greater taxonomy depth or rarity has no universal normative meaning.

[[TABLE_2]]

## 4 Method

### 4.1 Comparative learning protocol

Figure 1 summarises the full audit, learning, importance and ranking-propagation workflow.

[[FIGURE_1]]

Let X be an n by m matrix of n projects and m = 6 predictors, and let y be one of the two outcomes. Four learners are compared: Elastic Net, Random Forest, Extra Trees and histogram gradient boosting. A mean-prediction model supplies a performance floor but does not generate importance weights. Random Forest and Extra Trees each use 200 trees and a minimum leaf size of two. Histogram gradient boosting uses a learning rate of 0.05, 150 iterations, at most 15 leaf nodes and L2 regularisation of 0.1. Elastic Net examines 60 logarithmically spaced penalty values and four mixing values using five-fold validation within each outer training set.

The outer design uses five repetitions of five-fold cross-validation with fixed seed 20260811. Every learner receives exactly the same 25 train-test partitions for an outcome. For each repetition, predictions from its five held-out folds are assembled into one complete out-of-fold vector. R-squared, mean absolute error (MAE) and root mean squared error (RMSE) are then calculated across all observations in that repetition. Reported means, standard deviations and t-based 95% intervals refer to five complete out-of-fold repetitions, not averages of fold-level R-squared values.

Paired Wilcoxon signed-rank tests compare each learner with the learner having the lowest mean primary-outcome MAE. Tests use the five repeat-level observations and Holm adjustment. With only five pairs, inferential resolution is limited; the tests are descriptive checks against strong claims, not a basis for declaring a universally superior learner.

### 4.2 Held-out permutation importance and proxy weights

For learner l, outer test fold r and feature j, the raw permutation importance is the increase in test MAE after independently permuting that feature ten times:

EQ: I_jlr = mean_b [ MAE(y_r, f_lr(X_r,j^(perm,b))) - MAE(y_r, f_lr(X_r)) ].

Sampling noise or correlated substitutes can produce negative values. To obtain a non-negative profile compatible with standard weighted MCDM models, negative values are clipped:

EQ: I_jlr^+ = max(0, I_jlr).

The fold-specific proxy weight is:

EQ: w_jlr = I_jlr^+ / sum_h I_hlr^+.

If all clipped values equal zero, an equal profile is used and the event is flagged. No such fallback is hidden. The output reports raw importance, negative-value frequency, fold-specific weights, means, standard deviations and empirical 2.5th and 97.5th percentiles. Clipping and normalisation make the vector usable in a weighted score, but do not change its interpretation: w is a relative predictive-reliance profile under a learner, outcome and sample.

Learner agreement is assessed with Spearman correlation and Kendall tau between the six-feature weight orders, together with L1 distance between weight magnitudes. Downstream project ranks are compared using the same correlations and top-10 overlap.

### 4.3 Separation of empirical relevance and value assumptions

The analytical pipeline treats three objects separately. The first is the outcome definition, which states the historical behaviour to be explained. The second is predictive reliance, estimated from held-out observations. The third is the value model used to rank projects. This separation prevents a common category error. Permutation importance is unsigned: disrupting a feature can damage prediction whether its association with the outcome is positive, negative or non-monotonic. A large importance therefore cannot determine whether the feature should be maximised, minimised or transformed around a target. The illustrative MCDM directions are declared independently and can be replaced without refitting the predictive models.

The study also avoids selecting a learner after inspecting the most convenient ranking. All four prespecified learners remain in the results, even when one performs poorly for the secondary outcome. The cross-learner median profile is used only for the bounded perturbation experiment; learner-specific profiles remain visible. In this way, predictive accuracy, weight stability and ranking behaviour are reported as different evidential layers. A learner can predict well but yield unstable weights, or several weight profiles can differ while producing similar rankings because the decision matrix compresses those differences. Neither case is captured by a single performance or rank-correlation statistic.

No threshold is used to label a project acceptable. Ranks are relative to the frozen comparison set, and a high rank is not evidence that a project meets strategic, budgetary, safety or feasibility requirements. Operational use would require those constraints before an MCDM score is interpreted. The present propagation is therefore a stress test of analytical consequences, not a recommendation list.

### 4.4 MCDM propagation

The raw decision matrix is ratio-normalised. For a benefit criterion j, n_ij = x_ij / max_i x_ij. For the duration cost, n_ij = min_i x_ij / x_ij. Values are bounded away from zero only for numerical evaluation of logarithms and products.

WASPAS combines a weighted sum and weighted product with equal mixing coefficient (Zavadskas et al., 2012):

EQ: Q_i = 0.5 sum_j w_j n_ij + 0.5 product_j n_ij^(w_j).

MARCOS extends the normalised matrix with ideal and anti-ideal alternatives, calculates weighted sums S_i, and forms utility degrees K_i^- = S_i/S_AAI and K_i^+ = S_i/S_AI (Stević et al., 2020). The standard utility functions and final MARCOS utility are used rather than the legacy average of K values.

Each learner-specific mean profile and a cross-learner median profile are propagated through both ranking models. Four comparisons are retained: equal weights, standard-deviation weights, MEREC, and a solved FUCOM profile based on disclosed synthetic priority ratios. The FUCOM comparison is not elicited from stakeholders and is labelled synthetic throughout. It tests contrast with a coherent ordinal profile rather than claiming empirical preferences.

### 4.5 Robustness and generalisation diagnostics

Five diagnostics are applied. First, the leakage comparison fits the same Random Forest under (a) the legacy current-state predictors, (b) a partial reduction that removes current TRL but retains remaining TRL gap, and (c) the baseline-oriented predictor set. Second, programme-grouped five-fold validation is repeated five times so that projects from one programme do not appear in both training and test portions of a fold. Third, a chronological diagnostic trains on the earliest 80% of projects by recorded start date and tests on the latest 20%.

Fourth, the secondary-outcome weights are propagated over the same 196-project matrix as the primary weights, isolating the effect of outcome definition on the common alternatives. Fifth, every element of the primary cross-learner median profile is multiplied by an independent draw from a uniform distribution on [0.8, 1.2] and renormalised. Five thousand draws are evaluated for WASPAS and MARCOS. Spearman correlation, Kendall tau, top-10 overlap and project-level top-rank and top-10 acceptability are retained.

All code, fixed seeds, processed samples, fold-level results, figures and file hashes are generated by one command. The software versions are Python 3.12, NumPy 2.3.5, pandas 2.2.3, SciPy 1.17.0 and scikit-learn 1.8.0 (Pedregosa et al., 2011).

## 5 Results

### 5.1 Predictive performance

Table 3 reports repeated out-of-fold performance. For raw TRL progress, all four learners materially outperform mean prediction. Mean R-squared ranges from 0.553 for Extra Trees to 0.577 for histogram gradient boosting. Elastic Net and histogram gradient boosting have the lowest RMSE values, both approximately 0.735, whereas Extra Trees has the lowest MAE at 0.470. The metrics therefore do not identify one learner as uniformly best. Holm-adjusted paired tests are non-significant, which is unsurprising with five repeats and small differences among nonlinear models.

Figure 2 visualises the cross-validated R-squared results, and Table 3 retains all three metrics and their uncertainty.

[[FIGURE_2]]

[[TABLE_3]]

Target-normalised attainment is harder to predict. Elastic Net has mean R-squared -0.059, close to or worse than the mean baseline, while nonlinear learners achieve 0.185-0.231. Extra Trees again has the lowest MAE, 0.209, and Random Forest has the highest R-squared, 0.231. This decline is substantively important. Much of the predictability of raw progress disappears once planned ambition is placed in the outcome denominator.

Programme separation also lowers primary-outcome performance. Mean grouped R-squared ranges from 0.364 to 0.431, with Random Forest highest. In the chronological earliest-80%/latest-20% diagnostic, R-squared ranges from 0.253 to 0.463, with Elastic Net highest. Positive values indicate some transport, but the decline relative to random repeated folds confirms that within-portfolio performance should not be presented as a ready prospective system.

### 5.2 Leakage audit

The leakage diagnostic provides an initially counter-intuitive result. The legacy full current-state Random Forest achieves mean R-squared 0.469. Removing current TRL while retaining remaining gap reduces R-squared to 0.144. The leakage-aware baseline-oriented specification achieves 0.556. This does not validate a prospective interpretation. It shows that planned TRL gain is a strong predictor of absolute realised progress and that the incomplete deletion destroys information without constructing a coherent baseline question.

The audit therefore changes the scientific conclusion. The main risk is not simply optimistic performance caused by current TRL. It is construct dependence: raw progress is strongly structured by the amount of progress planned. A model can be free of current-state algebra and still yield a profile dominated by the target scale embedded in the outcome.

Figure 5 compares the three leakage-audit specifications.

[[FIGURE_5]]

### 5.3 Learner-specific proxy weights

Table 4 reports primary-outcome profiles. Planned TRL gain receives 0.961 under Elastic Net, 0.799 under Random Forest, 0.788 under Extra Trees and 0.746 under histogram gradient boosting. All other mean weights are below 0.102. The nonlinear learners allocate more importance to duration, taxonomy level and taxonomy rarity than Elastic Net, but the target-gain dominance is shared.

Figure 3 shows the same primary profiles with their outer-fold variation.

[[FIGURE_3]]

[[TABLE_4]]

Weight-order agreement is positive but not perfect. Pairwise Spearman correlations range from 0.600 to 0.943, Kendall tau from 0.467 to 0.867, and L1 distance from 0.083 to 0.431. Random Forest and histogram gradient boosting agree most closely in criterion order; Elastic Net and histogram gradient boosting differ most in magnitude. Wide fold-level intervals and non-zero negative-importance frequencies for weak predictors show that small residual weights should not be over-interpreted.

The secondary profiles are qualitatively different. Nonlinear learners place the largest weights on taxonomy level, taxonomy rarity and planned duration, while planned gain receives only 0.009-0.029. Elastic Net performs poorly for this outcome and concentrates on taxonomy level; its secondary importance profile is therefore descriptive of a weak model, not evidence that the feature is decision-dominant. Across outcomes, learner-specific weight-order correlations range only from 0.029 to 0.371 and L1 distances from 1.435 to 1.787. Outcome choice has a larger effect than learner choice within the primary specification.

### 5.4 Downstream ranking behaviour

Within the primary outcome, learner-specific ranking profiles are similar despite visible differences in minor weights. For WASPAS, pairwise rank correlations range from 0.972 to 0.997 and top-10 overlaps from nine to ten. For MARCOS, correlations range from 0.983 to 0.999 with the same top-10 range. Planned-gain dominance explains this stability: all learners place most mass on the same criterion.

Figure 4 contrasts learner agreement in proxy-weight order and downstream project ranking.

[[FIGURE_4]]

WASPAS and MARCOS also agree strongly within a fixed learned profile. Primary cross-method correlations are at least 0.983 for the learner profiles, with nine or ten common top-10 projects. Equal, synthetic FUCOM and standard-deviation profiles show lower but still substantial cross-method agreement. This evidence does not mean that the methods are interchangeable; it means that the present ratio-normalised matrix and weight structures generate similar orders.

Outcome changes are much more disruptive. When primary and secondary weights are applied to the same 196 projects, rank correlations range from 0.298 to 0.364 and top-10 overlaps from zero to two (Table 5). A project presented as robust under raw progress can therefore move substantially when the decision question changes from absolute advancement to target attainment. The outcome label is not a technical detail; it defines the empirical meaning of the weights.

[[TABLE_5]]

### 5.5 Weight perturbation and conventional comparisons

Bounded perturbations around the primary cross-learner median profile produce high local stability. Across 5,000 draws, the minimum correlation with the baseline is 0.998 for WASPAS and 0.999 for MARCOS after rounding to three decimals. WASPAS retains all ten baseline top projects in every draw. MARCOS has mean top-10 overlap 9.996 and minimum overlap nine. Figure 6 shows the resulting narrow distributions.

[[FIGURE_6]]

This robustness is local to a fixed outcome and neighbourhood. It cannot offset the outcome sensitivity reported above. Monte Carlo perturbation answers whether modest numerical changes around one profile alter the ranking. It does not answer whether the outcome construct or learner family is appropriate. The conventional profiles further show that wider modelling choices produce greater differences than bounded perturbations, so local stability should never be reported alone.

[[PAGE_BREAK]]

Table 6 consolidates the leakage, transport and perturbation diagnostics that bound interpretation of the main ranking results.

[[TABLE_6]]

## 6 Discussion

### 6.1 What can be learned from historical outcomes

Historical outcomes can provide an empirical lens on criterion relevance. In this study, held-out predictions show that baseline-oriented TechPort fields contain information about realised TRL progress. The approach is more defensible than fitting one ensemble to all data and reporting impurity importance because it exposes predictive error, fold-to-fold variability and learner dependence. It also preserves compatibility with standard MCDM models without proposing another ranking algorithm.

The strongest finding, however, is a warning. For raw progress, planned gain dominates every learner. That result is statistically coherent and managerially unsurprising: projects designed for more maturation have more opportunity to record absolute progress. It is not evidence that planned ambition deserves almost all normative decision weight. A portfolio manager may instead care about efficiency, strategic fit, target attainment, risk or knowledge spillovers. Predictive reliance on the selected outcome cannot decide among those values.

Normalising by planned gain changes the question and changes the learned profiles. The nonlinear models retain modest predictive skill, but the features they use are no longer the same. This demonstrates why learned weighting needs an outcome-design stage before model fitting. A technically accurate learner can answer the wrong managerial question precisely.

### 6.2 Implications for MCDM practice

Three practical safeguards follow. First, analysts should label learned profiles according to their outcome, learner and time frame. A generic name such as “AI weight” hides the source of the coefficients. Second, predictive importance and value direction should be stored separately. The former is estimated; the latter requires a stated policy, stakeholder process or value model. Third, robustness should span constructs, models and MCDM propagation. Perturbing one final vector by ten or twenty per cent is insufficient when alternative outcomes generate nearly disjoint top-10 sets.

The cross-learner median profile used here is a descriptive consensus, not a new estimator claimed to be optimal. It is useful for bounded perturbation because it avoids privileging one of several comparably performing learners. In an operational application, model selection or averaging should be defined before outcome inspection and should reflect the cost of prediction errors.

The corrected conventional benchmarks also have a limited but useful role. MEREC and standard-deviation weights describe decision-matrix structure. Synthetic FUCOM describes a coherent hypothetical priority order. None is a gold standard for the learned profiles because no stakeholder preference study is available. Their purpose is to reveal how different information sources propagate, not to validate one another by correlation.

### 6.3 Governance and reporting implications

For an organisation considering this workflow, the minimum governance artefact is a model card linked to a versioned decision dataset. It should record the decision date, intended users, outcome window, predictors available at that date, missing-data rules, learning algorithm, validation partitions, importance transformation, value directions and known prohibited uses. Each released ranking should also retain the project universe and weight vector from which it was calculated. Without these records, a later change in the live portfolio can make an apparently reproducible score impossible to reconstruct.

The results suggest a practical approval sequence. A domain owner should first approve the outcome as a meaningful historical proxy. A data steward should then verify time availability and revision histories for every predictor. A modelling reviewer should check out-of-sample performance and importance uncertainty. Finally, a decision owner should approve directions, value functions and constraints. The stages should not be collapsed into model fitting because they answer different accountability questions. In particular, decision owners should not be asked to endorse a weight vector before seeing how alternative defensible outcomes alter the ranking.

Monitoring would also be necessary in a continuing system. Changes in programme composition, TRL recording practice or project selection can alter both performance and importance. Predefined triggers could include a fall in grouped or chronological performance, an increase in clipping frequency, a large distance between successive importance profiles, or systematic rank changes for protected project classes. Such monitoring is outside this retrospective study, but stating it clarifies the gap between a research demonstration and an operational decision-support service.

### 6.4 Technology-portfolio interpretation

TechPort provides a credible public environment for demonstrating the protocol, but the project ranks should not be used as NASA recommendations. The snapshot mixes programmes, technology domains and project vintages. Higher initial TRL, taxonomy depth, rarity or collaboration count may be desirable under one screening objective and undesirable or irrelevant under another. The rankings are therefore methodological outputs under declared assumptions.

For technology-management practice, the main transferable idea is the audit sequence: define the decision time, classify variables by availability, define outcomes that match the managerial objective, compare credible learners out of sample, examine stability of importance, declare direction assumptions and trace results through more than one aggregation model. This sequence can be applied to defence acquisition, research funding, innovation portfolios and stage-gate systems when versioned historical records exist.

## 7 Limitations

The first limitation is temporal provenance. The source is a frozen current snapshot, not a panel of field values as originally recorded at each project start. Dates, target TRL and organisation counts are plausibly baseline-available but may have been revised. The analysis cannot claim a deployable ex-ante forecast. Future research should use versioned records or prospectively frozen decision packets.

Second, the complete-case primary sample contains 201 projects from one public portfolio. Programme-grouped and chronological results remain positive, but external validity is untested. Replication in another technology organisation is necessary before generalising the importance patterns.

Third, both outcomes are based on coarse TRL integers. Raw progress is bounded by planned gain, while the attainment ratio compresses over-achievement and is undefined for zero planned gain. Other outcomes, such as time-to-milestone, budget-adjusted progress, transition success or expert-rated value, would answer different questions and may require classification, survival or ordinal models.

Fourth, permutation importance can divide relevance among correlated features and can be negative by sampling variation. Clipping is transparent and convenient for MCDM but creates mass at zero. The reported clipping frequencies and intervals mitigate, but do not eliminate, this limitation. Conditional permutation or grouped importance could be explored with larger data.

Fifth, benefit/cost directions are illustrative. Neither the source data nor predictive models identify stakeholder utility. Actual portfolio use would require value-focused stakeholder work, explicit scale construction and governance review. Similarly, the synthetic FUCOM profile is not real elicitation.

Finally, repeated random, grouped and chronological validation quantify internal and transport behaviour but do not establish causality. No result should be interpreted as the effect of changing a criterion or as advice to manipulate recorded project attributes.

## 8 Conclusion

This study shows how supervised learning can be integrated with MCDM without disguising predictive reliance as preference. A leakage-aware, repeated-cross-validation protocol derives held-out permutation proxy weights from four learners and traces them through corrected WASPAS and MARCOS models. In the NASA TechPort snapshot, raw TRL progress is moderately predictable, but the resulting profiles are dominated by planned TRL gain. A target-normalised outcome is harder to predict and produces substantially different weights and rankings.

The evidence supports three conclusions. Learner comparison is necessary but, in this application, outcome construction matters more than learner choice. Local Monte Carlo stability does not establish construct robustness. Finally, learned importance magnitudes require separate normative directions and cannot support causal or stakeholder-preference claims. Used within these boundaries, outcome-conditioned proxy profiles can make historical-data-informed MCDM more transparent and auditable. Used without them, they risk replacing explicit judgement with hidden specification choices.

## Conflict of interest

All authors declare that they have no conflicts of interest.

## Data availability

An anonymised replication package containing the frozen data snapshot, scripts, generated tables, figures and hash manifest is supplied for review. Public repository and versioned archive links are provided in the separate title-page file and will be inserted into the article after double-blind review.

## References

Altmann, A., Toloşi, L., Sander, O. and Lengauer, T. (2010) ‘Permutation importance: a corrected feature importance measure’, Bioinformatics, Vol. 26, No. 10, pp.1340-1347, doi: 10.1093/bioinformatics/btq134.

Breiman, L. (2001) ‘Random forests’, Machine Learning, Vol. 45, No. 1, pp.5-32, doi: 10.1023/A:1010933404324.

Cinelli, M., Kadziński, M., Gonzalez, M. and Słowiński, R. (2020) ‘How to support the application of multiple criteria decision analysis? Let us start with a comprehensive taxonomy’, Omega, Vol. 96, 102261, doi: 10.1016/j.omega.2020.102261.

Corrente, S., Figueira, J.R., Greco, S. and Słowiński, R. (2017) ‘A robust ranking method extending ELECTRE III to hierarchy of interacting criteria, imprecise weights and stochastic analysis’, Omega, Vol. 73, pp.1-17, doi: 10.1016/j.omega.2016.11.008.

Fisher, A., Rudin, C. and Dominici, F. (2019) ‘All models are wrong, but many are useful: learning a variable’s importance by studying an entire class of prediction models simultaneously’, Journal of Machine Learning Research, Vol. 20, No. 177, pp.1-81.

Friedman, J.H. (2001) ‘Greedy function approximation: a gradient boosting machine’, The Annals of Statistics, Vol. 29, No. 5, pp.1189-1232, doi: 10.1214/aos/1013203451.

Fu, C., Chang, W., Xue, M. and Yang, S. (2021) ‘Data-driven decision making based on evidential reasoning approach and machine learning algorithms’, Applied Soft Computing, Vol. 110, 107622, doi: 10.1016/j.asoc.2021.107622.

Geurts, P., Ernst, D. and Wehenkel, L. (2006) ‘Extremely randomized trees’, Machine Learning, Vol. 63, No. 1, pp.3-42, doi: 10.1007/s10994-006-6226-1.

Greco, S., Słowiński, R. and Wallenius, J. (2025) ‘Fifty years of multiple criteria decision analysis: from classical methods to robust ordinal regression’, European Journal of Operational Research, Vol. 323, No. 2, pp.351-377, doi: 10.1016/j.ejor.2024.07.038.

Holm, S. (1979) ‘A simple sequentially rejective multiple test procedure’, Scandinavian Journal of Statistics, Vol. 6, No. 2, pp.65-70.

Keshavarz-Ghorabaee, M., Amiri, M., Zavadskas, E.K., Turskis, Z. and Antucheviciene, J. (2021) ‘Determination of objective weights using a new method based on the removal effects of criteria (MEREC)’, Symmetry, Vol. 13, No. 4, 525, doi: 10.3390/sym13040525.

Kizielewicz, B., Tomczyk, T., Gandor, M. and Sałabun, W. (2024) ‘Subjective weight determination methods in multi-criteria decision-making: a systematic review’, Procedia Computer Science, Vol. 246, pp.5396-5407, doi: 10.1016/j.procs.2024.09.673.

Maghsoodi, A.I., Khalilzadeh, M., Poursoltan, P., Antucheviciene, J. and Turskis, Z. (2023) ‘A machine learning driven multiple criteria decision analysis using LS-SVM feature elimination: sustainability performance assessment with incomplete data’, Engineering Applications of Artificial Intelligence, Vol. 119, 105785, doi: 10.1016/j.engappai.2022.105785.

NASA (2017) NASA Systems Engineering Handbook, NASA/SP-2016-6105 Rev. 2, National Aeronautics and Space Administration, Washington, DC.

NASA (2026) ‘TechPort API’, available at https://techport.nasa.gov/help/api (accessed 11 August 2026).

Pamučar, D., Stević, Ž. and Sremac, S. (2018) ‘A new model for determining weight coefficients of criteria in MCDM models: Full Consistency Method (FUCOM)’, Symmetry, Vol. 10, No. 9, 393, doi: 10.3390/sym10090393.

Pedregosa, F. et al. (2011) ‘Scikit-learn: machine learning in Python’, Journal of Machine Learning Research, Vol. 12, pp.2825-2830.

Stević, Ž., Pamučar, D., Puška, A. and Chatterjee, P. (2020) ‘Sustainable supplier selection in healthcare industries using a new MCDM method: Measurement of Alternatives and Ranking according to Compromise Solution (MARCOS)’, Computers & Industrial Engineering, Vol. 140, 106231, doi: 10.1016/j.cie.2019.106231.

Tervonen, T., van Valkenhoef, G., Baştürk, N. and Postmus, D. (2013) ‘Hit-And-Run enables efficient weight generation for simulation-based multiple criteria decision analysis’, European Journal of Operational Research, Vol. 224, No. 3, pp.552-559, doi: 10.1016/j.ejor.2012.08.026.

Varma, S. and Simon, R. (2006) ‘Bias in error estimation when using cross-validation for model selection’, BMC Bioinformatics, Vol. 7, 91, doi: 10.1186/1471-2105-7-91.

Zavadskas, E.K., Turskis, Z., Antucheviciene, J. and Zakarevicius, A. (2012) ‘Optimization of weighted aggregated sum product assessment’, Elektronika ir Elektrotechnika, Vol. 122, No. 6, pp.3-6, doi: 10.5755/j01.eee.122.6.1810.

Zou, H. and Hastie, T. (2005) ‘Regularization and variable selection via the elastic net’, Journal of the Royal Statistical Society: Series B, Vol. 67, No. 2, pp.301-320, doi: 10.1111/j.1467-9868.2005.00503.x.
