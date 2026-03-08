# Methodological Notes

## Data Source

All project data used in this study were obtained from the publicly available **NASA TechPort API**.

NASA TechPort is a project management database containing detailed information on NASA technology development projects, including technology readiness levels, project timelines, organizational participation, and technology taxonomy classifications.

The data extraction process used the official API endpoint:

https://techport.nasa.gov/api/projects/

Project-level JSON records were retrieved and parsed to construct the empirical dataset.

---

## Sample Construction

The initial dataset consisted of all projects retrieved through the API query process.

A **complete-case filtering strategy** was applied to ensure that all projects included in the final analysis contained valid values for the required decision criteria.

Projects with missing values in the following fields were excluded:

- TRL current
- TRL begin
- project duration
- taxonomy classification
- organizational participation

After filtering, the final empirical sample consisted of the projects included in the decision matrix.

---

## Decision Criteria

Six decision criteria were constructed for the technology portfolio evaluation.

1. **TRL current**  
   Current technology readiness level of the project.

2. **Duration (months)**  
   Project duration calculated from project start and end dates.

3. **TRL gap**  
   Difference between expected final TRL and current TRL.

4. **Primary taxonomy level**  
   Depth of the technology classification within the NASA technology taxonomy.

5. **Taxonomy rarity**  
   Inverse frequency of the technology taxonomy code in the dataset.

6. **Organizational participation**  
   Number of participating organizations involved in the project.

These criteria capture multiple dimensions of technology development including maturity, advancement potential, structural novelty, and collaboration intensity.

---

## AICW Target Definition

The Artificial Intelligence for Criterion Weighting (AICW) framework learns criterion importance from **observed technology maturation outcomes**.

The learning target is defined as:

TRL progress = TRL_current − TRL_begin

This variable measures realized technology advancement during the project lifecycle and serves as the prediction target for the machine learning model.

---

## Weighting Methods

Five weighting approaches were evaluated:

- AICW (Artificial Intelligence for Criterion Weighting)
- FUCOM
- LBWA
- MEREC
- Standard Deviation weighting

FUCOM and LBWA were implemented as **structured benchmark parameterizations**, rather than expert-elicited judgments, to allow consistent comparison with the data-driven AICW approach.

---

## Ranking Models

Two ranking models were used to evaluate project prioritization:

- **WASPAS** (Weighted Aggregated Sum Product Assessment)
- **MARCOS** (Measurement Alternatives and Ranking according to Compromise Solution)

Using two aggregation models allows evaluation of ranking robustness across different decision aggregation logics.

---

## Concordance Analysis

Ranking agreement between weighting methods was evaluated using:

- Kendall's tau rank correlation coefficient
- Spearman's rho correlation coefficient

In addition, overlap between the **top-10 ranked projects** across weighting methods was analyzed.

---

## Sensitivity Analysis

Sensitivity analysis was performed to evaluate the robustness of AICW rankings.

A **one-at-a-time perturbation approach** was applied where each criterion weight was increased by **10%**, while the remaining weights were proportionally normalized.

The resulting ranking changes were measured using:

- mean rank shift
- maximum rank shift

---

## Reproducibility

All steps required to reproduce the empirical analysis are provided in this repository, including:

- data extraction scripts
- decision matrix construction
- weighting method implementations
- ranking algorithms
- concordance analysis
- sensitivity analysis
- figure generation

The full analytical workflow is documented in the accompanying Jupyter/Colab notebook.
