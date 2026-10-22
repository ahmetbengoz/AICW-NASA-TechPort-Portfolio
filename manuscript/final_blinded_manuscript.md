# Auditing learned criterion weights in technology portfolio screening

## Abstract

Learning criterion weights from historical outcomes does not establish that the resulting scores improve project screening. This study audits that inference using a frozen NASA TechPort batch containing 201 completed projects. Four learners are compared under repeated cross-validation, and permutation importance is propagated through two aggregation models. A nested evaluation then learns weights exclusively within training projects and evaluates shortlists on untouched projects, alongside equal, entropy, CRITIC, MEREC, dispersion, single-criterion and direct-prediction comparators. A planned-gain-only linear model achieves R² of 0.576 for raw readiness progress, compared with 0.543 for the six-feature Random Forest under training-only preprocessing. In random test cohorts, selecting the top fifth by planned gain produces mean progress lift of 1.259 readiness levels; learned WASPAS and MARCOS profiles produce 1.234 and 1.254. For target attainment, learned-weight lift is negative, at −0.032 and −0.029, whereas direct prediction yields 0.118. Programme-separated and chronological results vary substantially. Predictive signal and local rank stability therefore do not establish reliable shortlisting benefit. The contribution is a reproducible empirical audit of the importance-to-ranking conversion, not a new learning algorithm or an estimate of stakeholder preferences.

Keywords: technology portfolio screening; criterion weighting; permutation importance; nested validation; multi-criteria decision analysis; NASA TechPort; shortlist evaluation; outcome sensitivity

## 1 Introduction

Technology portfolio screening compares projects that differ in maturity, intended advancement, duration and organisational setting. Multi-criteria decision analysis (MCDA) makes such comparisons explicit, but the meaning of its coefficients depends on the evaluation model and the decision owner's objectives [1, 2]. Historical project data offer another source of information: a learner can predict a recorded outcome, and feature-importance magnitudes can be normalised into criterion weights. This produces a reproducible vector, but reproducibility alone does not show that its induced ranking is useful.

The integration itself is established. Fu et al. linked learned weights to evidential reasoning [3], while Ijadi Maghsoodi et al. combined feature elimination, missing-data treatment and target-based MCDA [4]. More directly, Zindani and Singh used Random Forest weighting with TODIM and 5,000 weight-perturbation runs in materials selection [5]. Wang et al. subsequently proposed peer-prediction trees and endogenous uniqueness weighting [6]. A 2026 review also documents a substantial ML–MCDA integration literature [7]. Consequently, adding a learner to a familiar aggregation model, or demonstrating local perturbation stability, is insufficient as a contribution in its own right.

This article asks a narrower, testable question: does translating predictive reliance into unsigned criterion weights preserve useful information for screening unseen projects? Three distinctions matter. Predictive validity concerns an estimated outcome. Importance validity concerns how a fitted model relies on its inputs. Shortlist validity concerns the outcomes represented among the projects selected by a scoring rule. None of these quantities identifies stakeholder utility, and none is guaranteed by another. Recent work on feature-rank uncertainty and auditing reinforces the need to evaluate more than numerical stability [8, 9].

The question is examined using a frozen public NASA TechPort batch. Raw readiness progress and target-normalised attainment provide two defensible but different historical proxies. The descriptive analysis compares four learners, evaluates importance on held-out folds and propagates their profiles through WASPAS and MARCOS. The decisive extension places weight learning inside each outer training set and evaluates the resulting shortlist on projects whose outcomes were unavailable to every fitting stage. It also includes structural baselines, so that performance due mainly to planned maturation headroom is not attributed to a multivariate weighting method.

The contribution is the empirical separation of prediction, importance and shortlisting in a reproducible technology-portfolio audit. It is not a claim to have invented permutation importance, nested validation, criterion normalisation or either aggregation model. The analysis identifies when apparently persuasive evidence for learned weighting fails a more relevant comparison. Its scope is retrospective engineering decision support: no NASA decision, funding allocation, causal intervention or stakeholder-preference model is observed.

## 2 Related work and evaluation requirements

### 2.1 Weight meaning and learning targets

Preference-oriented weights, data-geometry coefficients and predictive importance are different objects. FUCOM structures ordered comparative judgements [10]; CRITIC combines dispersion and intercriterion conflict [11]; MEREC measures changes caused by removing criteria from a score construction [12]. These procedures need not agree because they extract different information. A review of subjective weighting likewise treats elicitation procedures as part of the decision model rather than interchangeable estimators of one universal parameter [13].

The learned coefficients studied here measure relative reliance on a specified historical outcome. They do not estimate the trade-offs that a portfolio owner would endorse. Robust MCDA methods examine uncertainty in preferences and model structure [14, 15], whereas a learner's importance variability also reflects sampling, predictive loss and feature dependence. Treating these sources as equivalent would obscure which question a robustness calculation answers.

### 2.2 Feature importance is not a value function

Permutation importance measures the deterioration in predictive performance when an input is disrupted. It remains model- and distribution-dependent; correlated predictors can make unrestricted permutations unrealistic [16]. Model-reliance research also shows why one importance vector should not stand for every predictively adequate model [17]. Recent work provides explicit guarantees for selected importance rankings [8] and examines causal-context problems in multivariate attribution [18]. The empirical fold percentiles reported here provide neither those guarantees nor causal identification.

A simple counterexample establishes the relevant limitation without requiring a new theorem. Consider one criterion x between zero and one and a perfectly known outcome y = 1 − x. Normalising its positive permutation importance yields weight one. If x is then declared a benefit, the induced ranking is the reverse of the outcome ranking. Perfect prediction therefore cannot make an unsigned importance coefficient identify direction or preserve the predictive decision rule. Nonlinearities and interactions create further opportunities for information loss when a fitted function is replaced by a weighted aggregate.

### 2.3 Position relative to current integrations

The closest weighting applications motivate a validation requirement, not a claim that their results are generally wrong. Zindani and Singh's application illustrates the established Random-Forest-to-MCDA pathway [5]. Wang et al.'s peer-prediction method instead derives expectations from relationships among criteria, rather than an external maturation outcome [6]. It addresses a different estimand and is not a forecasting comparator here. The present evaluation uses outcome-matched baselines: simple headroom models, conventional coefficient profiles and direct predictions from the same Random Forest used to learn weights.

Daneshvar et al. combine label randomisation, deletion and sufficiency checks with temporal explanation audits in data streams [9]. This article does not claim the first general importance-auditing protocol. It adds an engineering-portfolio demonstration in which the downstream conversion itself is tested on held-out alternatives, together with sensitivity to what counts as a successful historical outcome. Table 1 defines the dataset to which those findings apply.

## 3 Data and outcomes

### 3.1 Frozen project batch

The source is the public TechPort API [19]. The repository contains a frozen batch of 299 released project records, not a probability sample or a census of NASA's technology portfolio. No live API call is required for reproduction. Project titles, identifiers and programme names are retained for traceability, but titles, identifiers and current TRL are excluded from the baseline-oriented predictor set.

The six-feature analysis requires initial, current and target TRL; positive recorded duration; technology taxonomy information; and organisation count. There are 214 complete records under the current-state audit fields, of which 13 are not completed. The primary analysis therefore contains 201 completed projects. Five have zero planned gain and cannot enter the attainment-ratio analysis, leaving 196. Complete-case restriction and batch selection limit representativeness; the reported effects should not be extrapolated to all NASA projects.

[[TABLE_1]]

### 3.2 Outcomes and predictor timing

For project i, raw progress is current TRL minus initial TRL. Planned gain is target TRL minus initial TRL. For the 196 projects with positive planned gain, attainment is realised progress divided by planned gain, capped at one. No retained project exceeds its target, so the cap does not change the observed values. These definitions are:

EQ: y_i^P = TRL_current,i − TRL_initial,i

EQ: g_i = TRL_target,i − TRL_initial,i

EQ: y_i^A = min(1, y_i^P / g_i),  g_i > 0

Raw progress averages 1.672 TRL levels with standard deviation 1.132; mean attainment is 0.815. Absolute progress rewards the number of levels gained, while attainment rewards completing the intended gain. A project advancing three of four planned levels outranks one advancing one of one under progress, but not under attainment. Neither proxy measures strategic value, transition to use, cost effectiveness or the desirability of the project.

Six predictors are retained: initial TRL, duration, planned gain, taxonomy depth, taxonomy rarity and number of other organisations. Duration is the difference between recorded end and start dates divided by 30.4375. In the descriptive analysis, rarity is reciprocal category frequency in the fixed 299-record batch. In the nested evaluation it is re-estimated from training projects only; a category absent from training receives rarity one. These are deliberately distinct descriptive and inductive preprocessing regimes.

TechPort provides a current snapshot rather than the field values originally recorded at project initiation. Target TRL, dates and partners may have been revised. They are therefore plausibly baseline-available fields, not verified baseline measurements. Even a strict chronological separation of recorded start dates cannot establish that training outcomes or field revisions were observable at a historical funding decision. The analysis tests transport within a snapshot, not deployability.

[[TABLE_2]]

Initial TRL, planned gain, taxonomy depth, rarity and organisation count are treated as benefits; duration is treated as a cost. These directions are disclosed screening assumptions. In particular, neither deeper taxonomy nor rarity is universally desirable. Their role is to expose the consequences of translating unsigned importance into a declared value model, not to reconstruct NASA preferences.

## 4 Methods

### 4.1 Descriptive prediction and importance comparison

Figure 1 separates the descriptive analysis from the outer-test shortlist evaluation. Elastic Net, Random Forest, Extra Trees and histogram gradient boosting provide linear and nonlinear references [20, 21, 22, 23]. They share five repetitions of five-fold cross-validation with seed 20260811. A mean-prediction baseline is included. Random Forest and Extra Trees use 200 trees and minimum leaf size two. Gradient boosting uses 150 iterations, learning rate 0.05, at most 15 leaves and L2 regularisation 0.1. Full settings are supplied in the supplement.

[[FIGURE_1]]

Elastic Net searches 60 penalty values and four mixing values by five-fold inner validation. Scaling is fitted inside each inner training split through a pipeline-based grid search, avoiding preprocessing of inner validation cases [24]. Each outer repetition assembles a complete out-of-fold prediction vector. R², mean absolute error (MAE) and root mean squared error (RMSE) are then calculated across the complete vector rather than averaged from fold R² values.

For each outer held-out fold, ten permutations per feature estimate the increase in MAE [25]. Negative values are retained in detailed results, counted and then clipped at zero for coefficient construction. If every clipped value is zero, an equal profile is substituted and flagged. Fold profiles are averaged for the descriptive ranking comparison. Their empirical 2.5th and 97.5th percentiles describe partition variation, not population confidence limits. Repetitions reuse the same projects, so neither repeat-level intervals nor paired tests are interpreted as independent-sample inference.

EQ: I_jlr = mean_b [MAE(y_r, f_lr(X_r,j^(perm,b))) − MAE(y_r, f_lr(X_r))]

EQ: w_jlr = max(0, I_jlr) / Σ_h max(0, I_hlr)

These profiles are descriptive outputs of held-out importance estimation. They are not used as prevalidated weights for the same projects in the nested shortlist test described below.

### 4.2 Aggregation and descriptive robustness

For the descriptive choice universe, each benefit criterion is divided by its maximum and each cost criterion is transformed by its positive minimum divided by its value. Values entering logarithms have a disclosed 10⁻¹² floor. WASPAS averages weighted additive and weighted multiplicative scores with mixing coefficient 0.5 [26]. MARCOS uses the standard ideal and anti-ideal utility construction [27].

EQ: Q_i^W = 0.5 Σ_j w_j z_ij + 0.5 exp(Σ_j w_j log z_ij)

For fixed weights and comparison universe, standard MARCOS utility is a positive multiple of the weighted normalised row sum. Its rank agreement with WASPAS is therefore not independent evidence that a recommendation is valid. The supplement gives the algebra and a numerical invariant test. The models are used to inspect propagation under additive and additive–multiplicative aggregation.

Spearman correlation, Kendall correlation and top-ten overlap describe ranking differences. Primary and attainment weights are also applied to the same 196 projects to isolate outcome construction. Local robustness multiplies each primary cross-learner median weight by an independent uniform draw on [0.8, 1.2], renormalises and repeats 5,000 times. This quantifies stability near one profile, not uncertainty over stakeholder values or outcome choice.

### 4.3 Leakage and transport diagnostics

Three Random Forest specifications distinguish current-state algebra from baseline-oriented prediction. The first includes current TRL and remaining gap; the second removes current TRL but retains target minus current TRL; the third excludes both current-state quantities and uses planned gain. Since the outcome contains current TRL, removing one explicit column is not enough to remove algebraic dependence.

Programme-separated validation prevents the same programme from appearing in both outer training and test sets. A start-date transport check trains on earlier-starting projects and evaluates later-starting projects. The strictly separated version in the nested evaluation uses 1 October 2021 as the boundary and keeps every project with that date on the test side: 157/44 primary training/test projects and 153/43 attainment projects. This avoids splitting a tied start date across partitions. It does not supply missing historical field versions or guarantee training-label availability at that date.

### 4.4 Nested shortlist evaluation

The nested experiment evaluates the conversion from a predictor to an MCDA score rather than treating rank agreement as a performance metric. Random Forest is fixed as the reference importance learner because it is the model commonly used in the integration being audited, not because it wins every descriptive metric. Both its direct predictions and its normalised importance profiles are retained. This controlled comparison isolates information lost in the conversion while leaving the four-learner descriptive analysis visible.

For each outer split, three inner folds use only the outer training projects. An inner model is fitted, and its feature importance is evaluated on the corresponding inner held-out projects using ten permutations. The three signed importance vectors are averaged before clipping and unit-sum normalisation. Unlike the descriptive fold-average convention, this construction avoids giving an independently normalised weak inner fold the same total mass as an informative one. Every preprocessing step is fitted within the relevant training set. Programme-separated outer splits also use programme-separated inner folds.

The resulting weights are frozen before outer candidates are scored. Rarity counts, criterion scales, objective comparator weights and the direct Random Forest predictor are learned from outer training projects alone. Benefit scales use training maxima; cost scales use positive training minima. A candidate beyond the training range may have a ratio above one and is not used to rescale other candidates. No candidate outcome enters fitting, weight extraction or scale estimation.

There are 25 random outer splits, five programme-separated splits and one strictly start-date-separated split for each outcome. Random and programme splits are different transport questions, not independent datasets. The chronological cohort is one diagnostic and does not support a distribution-free claim about future performance.

### 4.5 Comparators and screening metrics

Comparator coefficients are equal, Shannon entropy, CRITIC, standard deviation and MEREC [28, 11, 12]. They are estimated from training data only and propagated through WASPAS. The full comparison also includes ranking by planned gain alone, ranking by initial TRL alone, direct Random Forest prediction, and the learned WASPAS and MARCOS profiles. Entropy and CRITIC use training min–max-scaled, direction-adjusted inputs; their full equations and constant-column handling are in the supplement. These profiles are not stakeholder-preference gold standards.

Predictive ablations use an intercept-only mean, a linear planned-gain model, a linear initial-TRL-plus-gain model, a Random Forest without gain and the six-feature Random Forest. All are evaluated on identical outer cases with the same training-only rarity regime. This tests how much apparent learning is explained by the mechanical headroom associated with intended progress. It also prevents comparisons across different sample definitions from being mistaken for algorithmic gains.

The primary shortlist fraction is 20%, rounded upward within each outer test cohort; 10% and 30% are sensitivity checks. The main endpoint is mean observed outcome among selected candidates minus mean outcome of the candidate cohort. It is called outcome lift, not financial return or stakeholder utility. An oracle gap compares the shortlist mean with the best observed equal-size shortlist, and discounted ranking gain is retained as a supplementary metric.

EQ: L_q = mean(y in shortlist_q) − mean(y in candidate cohort)

When a tie crosses the selection boundary, remaining shortlist places are allocated fractionally and equally among tied candidates. This reports expected selection performance without breaking ties using project identifiers, record order or outcomes. Results are summarised as mean cohort-level lift, with all split-level values available. Programme folds have unequal sizes; cohort means are not population estimates. No significance claim is made from the overlapping random folds, and no policy is retrospectively selected as universally best.

### 4.6 Falsification and implementation checks

A label-randomisation test compares a prespecified five-fold Random Forest score with 100 permutations of the outcome. The model is refitted on every permutation with the same folds and training-only rarity construction. The corrected tail proportion is (1 + number of null scores at least as high as observed)/(101). This assesses overall predictive signal, not the significance of each feature or the validity of any preference direction. It cannot validate the subsequent scoring rule.

The implementation checks partition disjointness, programme separation, chronological date separation, training-only rarity and normalisation, coefficient simplex constraints, fractional ties, oracle bounds and the unsigned-direction counterexample. The pipeline also rejects truncated image files and exports opaque, white-background publication figures. Code uses the pinned numerical environment and scikit-learn [29].

### 4.7 Use of generative AI

OpenAI ChatGPT assisted literature-search organisation, code revision, analysis checking, drafting and document formatting. Numerical findings were calculated by the executable analysis scripts, not supplied as generated estimates. The authors remain responsible for verifying citations, analytical decisions and the final text. The tool is not an author and did not provide stakeholder judgements or new observational data.

## 5 Results

### 5.1 Descriptive predictive performance

Raw progress is predictable under random repeated folds: mean R² across the four learners ranges from 0.553–0.577. Target attainment is more difficult: the nonlinear range is 0.185–0.231, while Elastic Net has mean R² −0.067. Figure 2 and Table 3 retain the predictive comparison rather than assigning one winner across incompatible loss metrics. These results use the fixed-batch rarity variable and must not be conflated with the stricter training-only preprocessing in the nested experiment.

[[FIGURE_2]]

[[TABLE_3]]

### 5.2 Outcome-conditioned importance

Planned gain dominates every primary profile, receiving 0.746–0.963 of the total mean importance. Figure 3 and Table 4 show the remaining coefficients and their resampling variation. Differences among the small residual weights should not be interpreted as stable preference orderings. The secondary profiles shift towards taxonomy depth, rarity and duration, while the predictive performance of the associated models is weaker.

[[FIGURE_3]]

[[TABLE_4]]

The result is consistent with headroom dependence: a project intended to advance more TRL levels can record more absolute progress. It does not establish that greater ambition deserves a correspondingly large normative weight. The outcome change alters the importance construct rather than merely perturbing its numerical estimate.

### 5.3 Ranking propagation

Figure 4 compares primary weight-order and WASPAS project-rank agreement across learners. High project-rank agreement coexists with visible differences in small coefficients because all learners emphasise planned gain. When the outcome changes but the 196-project universe is held fixed, correlations range from 0.298–0.372, with 0–3 common top-ten projects (Table 5). Outcome construction is therefore consequential even where within-outcome learner agreement is high.

[[FIGURE_4]]

[[TABLE_5]]

### 5.4 Current-state leakage diagnostic

Mean Random Forest R² is 0.469 with legacy current-state inputs, 0.144 after removing current TRL but retaining remaining gap, and 0.556 under the baseline-oriented specification (Figure 5). The increase in the last model is not evidence that leakage was helpful or that future deployment is valid. Planned gain changes the information represented by the predictors and strongly structures raw progress. Removing current-state algebra and checking a meaningful structural baseline are separate requirements.

[[FIGURE_5]]

### 5.5 Local perturbation stability

Across 5,000 bounded perturbations of the primary median profile, minimum rank correlation is 0.998–0.999 across the two aggregation models. This narrow neighbourhood yields highly stable rankings (Figure 6), but it does not address the outcome sensitivity in Table 5 or independent shortlist performance. MARCOS and WASPAS agreement also partly reflects their shared additive component and common dominant criterion.

[[FIGURE_6]]

### 5.6 Structural baselines under training-only preprocessing

Table 6 provides the outcome-matched ablation. For progress, the gain-only linear model reaches mean random-fold R² 0.576, versus 0.543 for the full Random Forest; excluding gain from the forest yields −0.050. Random Forest has lower MAE than the gain-only model, at 0.521 versus 0.553, so the baseline is not uniformly superior across losses. Nevertheless, a six-feature algorithm is unnecessary to explain most of the R² result. The initial-TRL-plus-gain model yields 0.576, offering almost no additional explanatory performance over gain alone in random folds.

Under programme separation, progress R² is 0.558 for gain alone and 0.442 for the full forest. The corresponding chronological values are 0.490 and 0.449. For attainment, full-forest R² is 0.147 under random splits but falls to −0.694 under programme separation and −0.328 chronologically. Negative values are relative to the mean of the corresponding held-out outcomes. These results limit claims that the attainment importance structure transfers reliably across programmes or vintages.

[[TABLE_6]]

### 5.7 Independent shortlist outcomes

The nested shortlist test does not show a consistent benefit from replacing simple rules with learned weights. In random progress cohorts, gain-only ranking produces lift 1.259, while learned WASPAS and MARCOS produce 1.234 and 1.254. Their paired mean differences from gain-only ranking are −0.024 and −0.005. WASPAS improves on gain alone in 10 of 25 cohorts and MARCOS in 12; neither pattern supports a general advantage.

The progress result varies under transport. Programme-separated lift is 0.876 for gain alone, 0.828 for learned WASPAS and 0.844 for learned MARCOS. In the single chronological cohort, both learned aggregates yield 1.611 versus 1.367 for gain alone. That local improvement is retained, but it does not reverse the absence of a consistent advantage across validation settings.

For attainment, random-cohort lift is −0.032 for learned WASPAS and −0.029 for learned MARCOS. Direct Random Forest prediction yields 0.118. Thus the same model can provide useful outcome ordering on these cohorts while its importance-to-aggregate conversion loses that information. The learned aggregates improve in the chronological cohort, whereas direct-prediction lift becomes negative. These reversals rule out a universal replacement rule in either direction. Table 7 and Figure 7 show the full comparison, including conventional weighting baselines.

[[TABLE_7]]

[[FIGURE_7]]

The 10% and 30% results, oracle gaps, discounted ranking gain and every fold-level selection result are included in the supplement and machine-readable outputs. No superior policy is inferred simply because it wins one chronological or programme test.

### 5.8 Label randomisation

The prespecified full-forest five-fold R² values are 0.193–0.523 across the two outcomes. The largest null values from 100 label permutations are −0.031–−0.013, with corrected tail proportions 0.0099 for both outcomes. This supports predictive signal under the stated snapshot and split design. It does not contradict the poor attainment performance of the converted scores: signal detection concerns the fitted predictor, whereas shortlist lift concerns the rule actually used to select candidates.

## 6 Discussion

### 6.1 What the audit adds

The empirical finding is not that machine learning cannot support MCDA. It is that three persuasive observations can coexist without establishing the intended screening benefit: nonzero predictive skill, a reproducible importance profile and stability under modest weight perturbation. Here a simple planned-gain model explains most progress predictability, and importance-derived aggregates fail to improve random-cohort progress shortlists consistently. Under attainment, they can underperform random selection despite a useful direct predictor on the same outer cohorts.

This changes the interpretation of earlier integration patterns [3, 5]. A historical-outcome learner does not automatically supply a value model. Independent testing must evaluate the complete conversion rather than the predictor alone. A new attribution estimator might improve explanation reliability, but cannot by itself establish that maximising its unsigned coefficients is a defensible preference model. The direction counterexample and the observed direct-prediction-versus-weighted-score contrast make that distinction explicit.

### 6.2 Implications for technology screening

A portfolio analyst should first decide which historical proxy is relevant. Absolute maturation, target completion, transition to application and cost-adjusted progress are not interchangeable. An analyst should next compare the full method with a simple rule reflecting the proxy's construction. If planned gain alone gives similar predictive or shortlisting performance, complexity requires another justification, such as interpretability, operational constraints or stakeholder relevance that is measured rather than assumed.

The screening rule itself must then be evaluated on candidates excluded from weight learning. Outcomes used to estimate importance cannot subsequently be called independent evidence for the induced project rankings. Training-only scale and rarity construction are equally necessary for an inductive claim. Where candidate sets are intentionally known in advance, a descriptive transductive analysis can be legitimate, but should not be presented as an unseen-project forecast.

Benefit/cost directions require a domain decision. A criterion may be predictively relevant because it signals difficulty, delay or disadvantage. Giving it positive utility because its importance is large can reverse the intended recommendation. A practical model record should therefore preserve outcome definition, predictor timing, signed or nonlinear relationships, value assumptions, comparison universe, coefficient vector and an outer-test shortlist report. Without these records, an apparently objective learned weight can conceal a substantial modelling choice.

### 6.3 Robustness and practical boundaries

Local numerical stability is useful for detecting fragile recommendations, but is weak evidence when a profile is dominated by one structural variable. The appropriate robustness questions include outcome definition, train/test separation, programme and vintage transport, comparator simplicity and the translation from prediction to action. Modern explanation audits motivate a similar separation of signal, faithfulness and stability [9]. The present results add that a stable coefficient profile may be a poor scoring policy even when the predictor passes a signal falsification check.

The findings do not identify a preferred NASA project list. No budgets, strategic priorities, interactions among selected projects or constraints on programme balance are modelled. A fixed-fraction shortlist treats selection capacity as equal slots and reports outcomes of already-completed projects. It cannot estimate what would have happened if other projects had been funded, or the value of reallocating resources. An operational portfolio model would require prospective data provenance, stakeholder value functions, constraints and a policy-specific validation design.

## 7 Limitations

The study uses one nonrandom batch and 201 complete completed-project records. Missingness and completion conditioning can alter the observed associations. No independent agency or later frozen batch is available, and repeated folds are not external replications. The small chronological and programme cohorts also make performance sensitive to composition.

Snapshot provenance remains unresolved. Strictly separating recorded start dates does not prove that fields were frozen at initiation or that all earlier-starting projects had completed by the first test start. The chronological results are accordingly transport diagnostics, not historical investment backtests. The larger public TechPort inventory must not be confused with the small frozen analytical batch.

TRL values are coarse ordered levels whose numerical differences and ratios are treated as working summaries. Raw progress is bounded by planned gain in every retained record; this is an empirical property of this batch, not a universal law. Other outcomes could yield different importance and shortlist patterns. Correlated-feature permutation can also extrapolate [16], and clipping noisy negative loss increases can create mass at zero. No conditional-importance or causal guarantee is claimed.

The nested importance experiment fixes Random Forest and two aggregation functions rather than exhaustively searching learners, attribution methods and preference models. It tests a common conversion route under disclosed assumptions. Broader generalisation requires independent portfolios and prespecified follow-up comparisons. Label randomisation is limited to 100 draws and tests overall predictive signal only. Finally, outcome lift is an observational diagnostic with no causal or welfare interpretation.

## 8 Conclusions

In this NASA TechPort batch, predictive importance does not provide a consistently better technology-screening rule than simple structural comparators. A planned-gain-only model explains most raw-progress predictability, while nested importance-derived shortlists do not consistently outperform gain-only ranking. For attainment, direct predictions and converted coefficients can lead to opposite selection results on the same unseen cohorts. Local weight stability does not resolve these failures, and transport results vary.

The defensible use of learned coefficients is therefore conditional. They can describe predictive reliance and support transparent sensitivity analysis. To support screening, the complete scoring rule must additionally demonstrate relevant shortlist performance against simple baselines, with independent candidates, training-only preprocessing and separately justified value assumptions. This empirical audit supplies a reproducible way to expose that distinction without claiming a new AI algorithm or replacing stakeholder judgement.

## List of abbreviations

AI artificial intelligence; CRITIC Criteria Importance Through Intercriteria Correlation; FUCOM Full Consistency Method; MAE mean absolute error; MARCOS Measurement of Alternatives and Ranking according to Compromise Solution; MCDA multi-criteria decision analysis; MCDM multi-criteria decision-making; MEREC Method based on the Removal Effects of Criteria; RF Random Forest; RMSE root mean squared error; SD standard deviation; TRL technology readiness level; WASPAS Weighted Aggregated Sum Product Assessment.

## Declarations

### Availability of data and materials

The frozen source records, analysis scripts, processed samples, fold membership, importance profiles and shortlist results are available in the reproducibility repository [30]. A copy of the replication data and code accompanies this submission. Earlier Zenodo versions preserve superseded analyses and are not substitutes for the revision-specific files. The repository commit associated with this package is recorded in its reproducibility note.

### Competing interests

The authors declare that they have no competing interests.

### Funding

This research received no specific grant from any funding agency in the public, commercial or not-for-profit sectors.

### Authors contributions

AB contributed conceptualisation, data curation, formal analysis, investigation, methodology, software, validation, visualisation and drafting. CTG contributed conceptualisation, methodology, supervision, validation and critical review and editing. The authors are responsible for the submitted content.

### Acknowledgements

Not applicable.

### Ethics approval and consent to participate

Not applicable. The study analyses publicly released project-level administrative records and involves no recruited participants or intervention.

### Consent for publication

Not applicable.

### Institutional disclaimer

The views and conclusions are those of the authors and do not represent official positions or recommendations of NATO, NSPA or NASA. NASA did not participate in the study.

## References

1. Cinelli M, Kadziński M, Gonzalez M, Słowiński R (2020) How to support the application of multiple criteria decision analysis? Let us start with a comprehensive taxonomy. Omega 96:102261. https://doi.org/10.1016/j.omega.2020.102261

2. Greco S, Słowiński R, Wallenius J (2025) Fifty years of multiple criteria decision analysis: From classical methods to robust ordinal regression. European Journal of Operational Research 323(2):351-377. https://doi.org/10.1016/j.ejor.2024.07.038

3. Fu C, Xu C, Xue M, Liu W, Yang S (2021) Data-driven decision making based on evidential reasoning approach and machine learning algorithms. Applied Soft Computing 110:107622. https://doi.org/10.1016/j.asoc.2021.107622

4. Ijadi Maghsoodi A, Torkayesh AE, Wood LC, Herrera-Viedma E, Govindan K (2023) A machine learning driven multiple criteria decision analysis using LS-SVM feature elimination: Sustainability performance assessment with incomplete data. Engineering Applications of Artificial Intelligence 119:105785. https://doi.org/10.1016/j.engappai.2022.105785

5. Zindani D, Singh S (2025) Leveraging random forest regression for informed and risk-aware material selection in supercapacitor fabrication. Journal of Engineering and Applied Science 72(1):223. https://doi.org/10.1186/s44147-025-00789-4

6. Wang Z, Lim T, Baydaş M, Stević Ž, Yu Y, Liu V (2026) A Machine Learning Approach to Multi-Criteria Decision-Making via Peer-Prediction Trees and Excess Performance Evaluation. Algorithms 19(9):762. https://doi.org/10.3390/a19090762

7. Upadhyay D, Kumar A, Pant S, Kotecha K, Abraham A (2026) Machine learning assisted multi-criteria decision-making approaches for site selection: A systematic review. MethodsX 17:104007. https://doi.org/10.1016/j.mex.2026.104007

8. Goldwasser J, Hooker G (2025) Statistical significance of feature importance rankings. Proceedings of the Forty-first Conference on Uncertainty in Artificial Intelligence. PMLR 286:1476–1496. https://proceedings.mlr.press/v286/goldwasser25a.html

9. Daneshvar B, Fahy C, Yang S (2026) Auditing feature importance rankings in nonstationary data streams. Proceedings of the Fourth UK AI Conference 2026. PMLR 348:42–51. https://proceedings.mlr.press/v348/daneshvar26a.html

10. Pamučar D, Stević Ž, Sremac S (2018) A New Model for Determining Weight Coefficients of Criteria in MCDM Models: Full Consistency Method (FUCOM). Symmetry 10(9):393. https://doi.org/10.3390/sym10090393

11. Diakoulaki D, Mavrotas G, Papayannakis L (1995) Determining objective weights in multiple criteria problems: The critic method. Computers &amp; Operations Research 22(7):763-770. https://doi.org/10.1016/0305-0548(94)00059-H

12. Keshavarz-Ghorabaee M, Amiri M, Zavadskas EK, Turskis Z, Antucheviciene J (2021) Determination of Objective Weights Using a New Method Based on the Removal Effects of Criteria (MEREC). Symmetry 13(4):525. https://doi.org/10.3390/sym13040525

13. Kizielewicz B, Tomczyk T, Gandor M, Sałabun W (2024) Subjective weight determination methods in multi-criteria decision-making: a systematic review. Procedia Computer Science 246:5396-5407. https://doi.org/10.1016/j.procs.2024.09.673

14. Tervonen T, van Valkenhoef G, Baştürk N, Postmus D (2013) Hit-And-Run enables efficient weight generation for simulation-based multiple criteria decision analysis. European Journal of Operational Research 224(3):552-559. https://doi.org/10.1016/j.ejor.2012.08.026

15. Corrente S, Figueira JR, Greco S, Słowiński R (2017) A robust ranking method extending ELECTRE III to hierarchy of interacting criteria, imprecise weights and stochastic analysis. Omega 73:1-17. https://doi.org/10.1016/j.omega.2016.11.008

16. Hooker G, Mentch L, Zhou S (2021) Unrestricted permutation forces extrapolation: variable importance requires at least one more model, or there is no free variable importance. Statistics and Computing 31(6):82. https://doi.org/10.1007/s11222-021-10057-z

17. Fisher A, Rudin C, Dominici F (2019) All models are wrong, but many are useful: Learning a variable’s importance by studying an entire class of prediction models simultaneously. Journal of Machine Learning Research 20(177):1–81. https://jmlr.org/papers/v20/18-760.html

18. Martin J, Haufe S (2026) cc-Shapley: Measuring multivariate feature importance needs causal context. Proceedings of the 42nd Conference on Uncertainty in Artificial Intelligence. PMLR 337:4323–4352. https://proceedings.mlr.press/v337/martin26a.html

19. NASA (2026) TechPort API. https://techport.nasa.gov/help/api. Accessed 22 October 2026. The study uses the frozen repository batch, not a live API extraction on the access date.

20. Zou H, Hastie T (2005) Regularization and Variable Selection Via the Elastic Net. Journal of the Royal Statistical Society Series B: Statistical Methodology 67(2):301-320. https://doi.org/10.1111/j.1467-9868.2005.00503.x

21. Breiman L (2001) Random Forests. Machine Learning 45(1):5-32. https://doi.org/10.1023/A:1010933404324

22. Geurts P, Ernst D, Wehenkel L (2006) Extremely randomized trees. Machine Learning 63(1):3-42. https://doi.org/10.1007/s10994-006-6226-1

23. Friedman JH (2001) Greedy function approximation: A gradient boosting machine. The Annals of Statistics 29(5):1189-1232. https://doi.org/10.1214/aos/1013203451

24. Varma S, Simon R (2006) Bias in error estimation when using cross-validation for model selection. BMC Bioinformatics 7(1):91. https://doi.org/10.1186/1471-2105-7-91

25. Altmann A, Toloşi L, Sander O, Lengauer T (2010) Permutation importance: a corrected feature importance measure. Bioinformatics 26(10):1340-1347. https://doi.org/10.1093/bioinformatics/btq134

26. Zavadskas EK, Turskis Z, Antucheviciene J, Zakarevicius A (2012) Optimization of Weighted Aggregated Sum Product Assessment. Elektronika ir Elektrotechnika 122(6):3-6. https://doi.org/10.5755/j01.eee.122.6.1810

27. Stević Ž, Pamučar D, Puška A, Chatterjee P (2020) Sustainable supplier selection in healthcare industries using a new MCDM method: Measurement of alternatives and ranking according to COmpromise solution (MARCOS). Computers &amp; Industrial Engineering 140:106231. https://doi.org/10.1016/j.cie.2019.106231

28. Shannon CE (1948) A Mathematical Theory of Communication. Bell System Technical Journal 27(3):379-423. https://doi.org/10.1002/j.1538-7305.1948.tb01338.x

29. Pedregosa F et al (2011) Scikit-learn: Machine learning in Python. Journal of Machine Learning Research 12:2825–2830. https://jmlr.org/papers/v12/pedregosa11a.html

30. Bengöz A, Temel Gencer C (2026) Reproducibility package for auditing learned criterion weights in technology portfolio screening. Version 3.0.0. https://github.com/ahmetbengoz/AICW-NASA-TechPort-Portfolio. Accessed 22 October 2026.
