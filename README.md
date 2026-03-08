# AICW-NASA-TechPort-Portfolio

This repository contains the data, analytical scripts, and reproducibility materials for the study:

**Learning Criterion Weights from Technology Maturation Data:  
AICW and an Empirical Benchmark on NASA TechPort Projects**

The repository implements the Artificial Intelligence for Criterion Weighting (AICW) framework and compares it with established multi-criteria decision-making (MCDM) weighting methods using NASA technology development project data.

---

# Data Source

All project data were obtained from the publicly available **NASA TechPort API**.

NASA TechPort provides detailed information on NASA technology development projects, including:

- technology readiness levels (TRL)
- project timelines
- technology taxonomy classifications
- participating organizations

API endpoint used:

https://techport.nasa.gov/api/projects/

---

# Repository Structure
# AICW-NASA-TechPort-Portfolio

This repository contains the data, analytical scripts, and reproducibility materials for the study:

**Learning Criterion Weights from Technology Maturation Data:  
AICW and an Empirical Benchmark on NASA TechPort Projects**

The repository implements the Artificial Intelligence for Criterion Weighting (AICW) framework and compares it with established multi-criteria decision-making (MCDM) weighting methods using NASA technology development project data.

---

# Data Source

All project data were obtained from the publicly available **NASA TechPort API**.

NASA TechPort provides detailed information on NASA technology development projects, including:

- technology readiness levels (TRL)
- project timelines
- technology taxonomy classifications
- participating organizations

API endpoint used:

https://techport.nasa.gov/api/projects/

---

# Repository Structure
data/
raw/ → raw TechPort project data
processed/ → decision matrix and analysis outputs

scripts/
→ Python scripts for data processing and analysis

figures/
→ figures used in the manuscript

notebooks/
→ full reproducible analysis notebook

manuscript/
supplementary/ → methodological documentation


---

# Decision Criteria

The empirical analysis uses six evaluation criteria:

1. TRL current  
2. Project duration (months)  
3. TRL gap  
4. Primary taxonomy level  
5. Taxonomy rarity  
6. Organizational participation

These criteria capture technology maturity, advancement potential, structural novelty, and collaboration intensity.

---

# Weighting Methods

The study evaluates five weighting approaches:

- **AICW** – Artificial Intelligence for Criterion Weighting
- **FUCOM**
- **LBWA**
- **MEREC**
- **Standard Deviation weighting**

AICW derives criterion importance using machine learning based on observed technology maturation outcomes.

---

# Ranking Models

Two multi-criteria ranking models are used:

- WASPAS
- MARCOS

These models evaluate technology portfolio prioritization under different aggregation structures.

---

# Robustness Analysis

The study evaluates ranking robustness using:

- Kendall's tau correlation
- Spearman correlation
- Top-10 project overlap
- Sensitivity analysis (+10% weight perturbation)

---

# Reproducibility

The complete analytical workflow is provided in the notebook:


---

# Decision Criteria

The empirical analysis uses six evaluation criteria:

1. TRL current  
2. Project duration (months)  
3. TRL gap  
4. Primary taxonomy level  
5. Taxonomy rarity  
6. Organizational participation

These criteria capture technology maturity, advancement potential, structural novelty, and collaboration intensity.

---

# Weighting Methods

The study evaluates five weighting approaches:

- **AICW** – Artificial Intelligence for Criterion Weighting
- **FUCOM**
- **LBWA**
- **MEREC**
- **Standard Deviation weighting**

AICW derives criterion importance using machine learning based on observed technology maturation outcomes.

---

# Ranking Models

Two multi-criteria ranking models are used:

- WASPAS
- MARCOS

These models evaluate technology portfolio prioritization under different aggregation structures.

---

# Robustness Analysis

The study evaluates ranking robustness using:

- Kendall's tau correlation
- Spearman correlation
- Top-10 project overlap
- Sensitivity analysis (+10% weight perturbation)

---

# Reproducibility

The complete analytical workflow is provided in the notebook:


Running this notebook reproduces:

- decision matrix construction
- criterion weighting
- ranking models
- concordance analysis
- sensitivity analysis
- manuscript figures

---

# License

This repository is released under the MIT License.

---

# Citation

If you use this repository, please cite the associated study.

How to reproduce the analysis

---
# Environment

The analysis requires Python 3 and the following packages:

- pandas
- numpy
- matplotlib
- scikit-learn
- scipy  

Install them using:

pip install -r requirements.txt

# Reproducing the Analysis

To reproduce the empirical analysis:

1. Download the repository
2. Open the notebook:

notebooks/AICW_NASA_TechPort_Analysis.ipynb

3. Run all cells sequentially

The notebook performs the complete workflow:

- NASA TechPort data processing
- decision matrix construction
- criterion weighting
- ranking models
- concordance analysis
- sensitivity analysis
- figure generation
