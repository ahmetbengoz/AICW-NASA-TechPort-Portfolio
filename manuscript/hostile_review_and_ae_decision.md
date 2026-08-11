# Hostile reviewer report and associate-editor decision

## Hostile reviewer report

### Recommendation

**Major revision if reviewed as a methods claim; minor-to-moderate revision if evaluated as a retrospective applied protocol.** The manuscript is now potentially publishable, but only under its deliberately narrow interpretation.

### Major concerns

1. **The paper does not introduce a new AI algorithm.** Normalising feature importance into a unit-sum vector is straightforward. The authors have largely corrected the novelty claim, but the editor should ensure that the final metadata and cover letter do not revive the “AICW” branding. The contribution is the leakage-aware comparative protocol, outcome-sensitivity evidence and reproducibility package.

2. **Prospective availability is not proved.** Target TRL, planned end date and organisation counts may be revised in a live TechPort record. The paper calls these predictors “plausibly baseline-available”, which is appropriately cautious. The study remains retrospective and cannot support deployment claims without versioned field histories.

3. **The primary target is structurally tied to planned gain.** Planned gain dominates the learned profile because raw progress has a natural ceiling determined by intended progress. This is not necessarily leakage, but it limits the normative meaning of the result. The secondary attainment outcome usefully exposes the dependence, and the discussion must keep this as the central empirical finding rather than treating the primary ranking as the answer.

4. **The secondary model is only modestly predictive.** Nonlinear R² values of 0.185–0.231 may support exploratory benchmarking, not strong project evaluation. The Elastic Net secondary profile should not be interpreted because its R² is negative. The manuscript now says this explicitly.

5. **Permutation importance is not preference and is unstable under correlation.** Clipping negative estimates and normalising them produces a usable mathematical vector but not a stakeholder weight. The paper is acceptable only because it reports clipping frequency, fold variation, learner dependence and a separate illustrative direction model.

6. **The statistical comparison is low-powered.** Five repeated out-of-fold observations cannot establish learner equivalence. Holm-adjusted non-significance is reported appropriately, but readers might still infer equality. A future expansion should use more independent portfolios or a richer hierarchical resampling design.

7. **External validity is absent.** Grouped and chronological diagnostics are valuable, but they do not replace validation on a different organisation or later snapshot. A second portfolio would materially strengthen the evidence.

8. **Synthetic FUCOM is not a stakeholder benchmark.** Its inclusion is acceptable as a disclosed coherent comparison profile. It must never be described as expert validation, and no conclusion should depend on its superiority.

9. **The ranks have no policy legitimacy.** Treating taxonomy depth and rarity as benefits is illustrative. Actual screening would need stakeholder-approved value functions, constraints and governance. The package correctly warns that high rank is not project acceptability.

10. **Managerial relevance must remain concrete.** The staged approval and monitoring guidance in the discussion improves fit with IJADS. The editor may still ask for a worked governance example or decision-maker commentary.

### Minor concerns

- The attainment cap at one should be justified as a construct choice and tested if future data include over-attainment.
- Complete-case exclusion of 98 records may create selection effects; a missingness model would be useful with richer data.
- Confidence intervals over 25 folds would be pseudo-replicated; the paper correctly uses five complete out-of-fold repetitions for performance and empirical fold distributions for importance.
- The cross-learner median is a descriptive consensus rather than an estimator with optimality guarantees; this is now stated.
- Project identifiers in review files should be checked against the journal's anonymity expectations, even though they are public administrative identifiers.

### Reviewer verdict after revision

**Publishable as a careful applied decision-science study, not as an algorithmic novelty paper.** I would recommend external review and expect one substantive revision round focused on construct validity and managerial interpretation.

## Associate-editor simulation

### Stage decisions

| Stage | Novelty | Rigour | Benchmarking | Results | Positioning | Decision |
|---|---|---|---|---|---|---|
| Forensic audit | Red | Red | Red | Yellow | Red | REVISION REQUIRED |
| Revised experimental design | Yellow | Green | Green | Yellow | Green | GO |
| Reconstructed manuscript | Yellow | Green | Green | Yellow | Green | GO |
| Submission package | Yellow | Green | Green | Yellow | Green | GO |

### Decision letter

**Decision: SEND FOR EXTERNAL REVIEW.**

The revised submission now addresses the main reasons for earlier rejection: it abandons the claim of a fundamentally new AI weighting method; adds four comparable learners, repeated out-of-sample validation, held-out permutation importance, paired checks, grouped and chronological diagnostics; corrects the MCDM implementations; tests outcome and weight sensitivity; and supplies a reproducible, professionally formatted package. The title and abstract accurately position the work as an outcome-conditioned screening study.

The principal residual concern is construct validity. The primary outcome is dominated by planned TRL gain, while the target-normalised outcome is less predictable and generates different rankings. The authors treat this as a finding rather than hiding it. Because the paper's contribution is now the audit and propagation protocol, that limitation does not justify desk rejection. External reviewers should focus on whether the retrospective evidence and managerial governance discussion are sufficient for IJADS.

### Acceptance-probability estimate

Estimated probability of eventual acceptance at IJADS: **35–45%**. Estimated desk-review survival: **65–75%** if submitted with the current title, cover letter and double-blind formatting. The interval is a judgement, not a statistical forecast. The largest downside risks are single-portfolio validation, unverifiable historical field timing and weak secondary-outcome performance.
