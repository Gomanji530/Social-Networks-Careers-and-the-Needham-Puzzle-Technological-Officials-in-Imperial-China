# Social Networks, Careers, and the Needham Puzzle: Technological Officials in Imperial China (480–1864)

[![License: MIT](https://img.shields.org/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.org/badge/python-3.9+-blue.svg)](pyproject.toml)
[![Data Source: CBDB](https://img.shields.org/badge/Data_Source-CBDB-red.svg)](https://cbdb.hsites.harvard.edu/)
[![Code Style: Black](https://img.shields.org/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This repository contains the complete quantitative data pipeline, econometrics models, network analysis, spatial diagnostics, and interactive visualization suite for the MSc dissertation: 

> **"Social Networks, Careers, and the Needham Puzzle: Technological Officials in Imperial China (480–1864)"**

Utilizing the **China Biographical Database (CBDB)** relational SQLite snapshot (`cbdb_20260516.sqlite3`), this project quantitatively re-evaluates the Needham Question by reconstructing the human and social infrastructure of technical officials across 1,400 years of Chinese history.

---

## ⚙️ Quick Start & Environment Setup

This project is packaged with standard PEP 621 metadata (`pyproject.toml`). You can install the environment and all required dependencies using `pip`:


# Clone the repository
git clone [https://github.com/Gomanji530/The-Social-Infrastructure-of-Technical-Knowledge-in-Imperial-China.git](https://github.com/Gomanji530/The-Social-Infrastructure-of-Technical-Knowledge-in-Imperial-China.git)
cd The-Social-Infrastructure-of-Technical-Knowledge-in-Imperial-China

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install project and dependencies
pip install -e .

## 📑 Mapping to Thesis Methodology & Structure

The repository structure directly mirrors the analytical workflow and empirical sections (**Section IV: Data Design** and **Section V: Research Design**) of the dissertation:

| Repository Folder / File | Dissertation Chapter / Section | Methodology & Description |
| :--- | :--- | :--- |
| `CBDB_Publications_List.md` | **IV. Data Design** | Curated bibliography of academic literature utilizing CBDB data to contextualize data validity. |
| `Scientists_network_visualization/` | **IV.1 Sample Extraction & Keyword Matching**<br>**IV.2 Network Visualization** | Pipeline extracting technical personnel via keyword regex matching, computing BFS shortest paths (up to 6 degrees) to political elites, and generating an interactive D3.js web visualizer. |
| `Structural_break/` | **IV.3 Model-Based Structural Breaks** | Dynamic programming algorithm minimizing segment-wise logistic deviance with BIC penalty to identify historical structural breakpoints in official selection. |
| `Analysis 1_.../` | **V.1 Analysis 1: Network Structure & Social Contagion Dynamics** | Autologistic Actor-Attribute Models (ALAAM) using Maximum Pseudo-Likelihood Estimation (MPLE) to assess social contagion and centrality effects. |
| `Analysis 2_.../` | **V.2 Analysis 2: Spatial Patterns & Knowledge Diffusion** | Spatial analysis including Haversine centroids, $k$-NN Moran's $I$ spatial autocorrelation, permutation tests, and multinomial logit modeling. |
| `Analysis 3_.../` | **V.3 Analysis 3: Career Trajectories & Bureaucratic Ceilings** | Sequence analysis linking postings to technical domains, computing Levenshtein edit distances, weighted $k$-medoids clustering, and high-status access modeling. |
| `Analysis 4_.../` | **V.4 Analysis 4: Intergenerational Transmission of Technical Skills** | Extraction of father-son kinship pairs from CBDB `KIN_DATA` to model status inheritance via binary logit, benchmarked against civil examination (Jinshi) rates. |

## 📁 Repository Structure
.
├── Analysis 1_Network_Structure_and_Social_Contagion_Dynamics/
│   ├── alaam_science_officials.py                    # ALAAM model driver script
│   ├── alaam_science_officials_break3_induced_* # Induced subgraph MPLE estimates & matrices
│   ├── alaam_science_officials_break3_projection_* # Projected network MPLE estimates & matrices
│   └── test_alaam_science_officials.py               # Unit tests for ALAAM module
│
├── Analysis 2_Spatial_Patterns_and_Knowledge_Diffusion/
│   ├── spatial_cluster_analysis.py                   # Centroids, Moran's I & Multinomial Logit
│   ├── spatial_cluster_centroid_tests.csv            # Haversine distance shift outputs
│   ├── spatial_cluster_morans_i.csv                  # Spatial autocorrelation stats
│   ├── spatial_cluster_multinomial_logit.csv         # Spatial multinomial parameter estimates
│   └── test_spatial_cluster_analysis.py              # Unit tests for Spatial module
│
├── Analysis 3_Career_Trajectories_and_Bureaucratic_Ceilings/
│   ├── career_trajectory_analysis.py                 # Sequence edit distance & k-medoids
│   ├── career_trajectory_high_status_logit.csv       # High-official promotion odds ratios
│   ├── career_trajectory_transition_matrix.csv       # Bureau transition matrices
│   ├── career_trajectory_type_medoids.csv            # Archetypal career path medoids
│   └── test_career_trajectory_analysis.py            # Unit tests for Career module
│
├── Analysis 4_Intergenerational_Transmission_of_Technical_Skills/
│   ├── intergenerational_transmission_analysis.py    # Father-son kinship logit estimation
│   ├── intergenerational_logit.csv                   # Status transmission odds ratios
│   ├── intergenerational_by_cluster.csv               # Dynastic cluster cross-tabulations
│   └── test_intergenerational_transmission_analysis.py
│
├── Scientists_network_visualization/
│   ├── generate_science_network_20260516.py          # Data extraction & graph construction
│   ├── science_network_data_20260516.json            # Graph serialization schema
│   └── network_20260516.html                         # Standalone interactive D3.js viewer
│
├── Structural_break/
│   ├── science_official_structural_breaks.py         # DP structural break search engine
│   ├── science_official_model_breaks.csv             # Endogenous break dates & BIC scores
│   └── science_official_model_summary.json
│
├── CBDB_Publications_List.md                         # Reference catalog for CBDB literature
├── CITATION.cff                                      # Citation metadata file
├── LICENSE                                           # MIT License
├── pyproject.toml                                    # Project metadata and dependencies configuration
└── README.md                                         # Main repository documentation
│   └── science_official_model_summary.json
│
├── CBDB_Publications_List.md
└── README.md

## 📊 Data Source & Replication
The empirical dataset is derived from the China Biographical Database (CBDB), created through a collaborative effort between Harvard University, Academia Sinica, and Peking University.

To replicate the analyses:

Download the latest SQLite release from the CBDB Official Website.

Place the database file in your local path or configure the connection string inside generate_science_network_20260516.py.

Execute scripts sequentially according to the analysis modules.

## 📖 Citation
If you use this repository, dataset pipelines, or analytical models in your research, please cite as follows:
@mastersthesis{gu2026social,
  author       = {Gu, Yuchen},
  title        = {Social Networks, Careers, and the Needham Puzzle: Technological Officials in Imperial China (480--1864)},
  school       = {University College London},
  department   = {Department of Physics and Astronomy},
  year         = {2026},
  type         = {MSc Dissertation},
  url          = {[https://github.com/Gomanji530/The-Social-Infrastructure-of-Technical-Knowledge-in-Imperial-China](https://github.com/Gomanji530/The-Social-Infrastructure-of-Technical-Knowledge-in-Imperial-China)}
}

## 📜 License
This project is licensed under the MIT License - see the LICENSE file for details.
