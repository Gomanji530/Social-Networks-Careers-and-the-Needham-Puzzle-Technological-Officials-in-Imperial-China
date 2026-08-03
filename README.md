# The Social Infrastructure of Technical Knowledge in Imperial China

[![License: MIT](https://img.shields.org/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.org/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Data Source: CBDB](https://img.shields.org/badge/Data_Source-CBDB-red.svg)](https://cbdb.hsites.harvard.edu/)

This repository contains the full data extraction pipelines, quantitative statistical models, and visualization tools for the MSc dissertation: **"The Social Infrastructure of Technical Knowledge in Imperial China"**.

Utilizing the **China Biographical Database (CBDB)** SQLite snapshot (`cbdb_20260516.sqlite3`), this study investigates how scientific and technical personnel in pre-modern China were socially networked, temporally restructured, spatially distributed, bureaucratically promoted, and intergenerationally transmitted across imperial dynasties.

---

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

---

## 📁 Detailed Repository Structure

```text
.
├── Analysis 1_Network_Structure_and_Social_Contagion_Dynamics/
│   ├── alaam_science_officials.py
│   ├── alaam_science_officials_break3_induced_diagnostics.csv
│   ├── alaam_science_officials_break3_induced_edges.csv
│   ├── alaam_science_officials_break3_induced_mple_results.csv
│   ├── alaam_science_officials_break3_induced_mpnet_attributes.csv
│   ├── alaam_science_officials_break3_induced_nodes.csv
│   ├── alaam_science_officials_break3_projection_diagnostics.csv
│   ├── alaam_science_officials_break3_projection_edges.csv
│   ├── alaam_science_officials_break3_projection_mple_results.csv
│   ├── alaam_science_officials_break3_projection_mpnet_attributes.csv
│   ├── alaam_science_officials_break3_projection_nodes.csv
│   └── test_alaam_science_officials.py
│
├── Analysis 2_Spatial_Patterns_and_Knowledge_Diffusion/
│   ├── spatial_cluster_analysis.py
│   ├── spatial_cluster_centroid_tests.csv
│   ├── spatial_cluster_model_diagnostics.csv
│   ├── spatial_cluster_morans_i.csv
│   ├── spatial_cluster_multinomial_logit.csv
│   ├── spatial_cluster_nodes.csv
│   ├── spatial_cluster_summary.csv
│   ├── spatial_cluster_top_locations.csv
│   └── test_spatial_cluster_analysis.py
│
├── Analysis 3_Career_Trajectories_and_Bureaucratic_Ceilings/
│   ├── career_trajectory_analysis.py
│   ├── career_trajectory_cluster_summary.csv
│   ├── career_trajectory_high_status_logit.csv
│   ├── career_trajectory_model_diagnostics.csv
│   ├── career_trajectory_nodes.csv
│   ├── career_trajectory_postings.csv
│   ├── career_trajectory_transition_matrix.csv
│   ├── career_trajectory_type_medoids.csv
│   ├── career_trajectory_type_summary.csv
│   └── test_career_trajectory_analysis.py
│
├── Analysis 4_Intergenerational_Transmission_of_Technical_Skills/
│   ├── intergenerational_by_cluster.csv
│   ├── intergenerational_logit.csv
│   ├── intergenerational_model_diagnostics.csv
│   ├── intergenerational_summary.csv
│   ├── intergenerational_transmission_analysis.py
│   └── test_intergenerational_transmission_analysis.py
│
├── Scientists_network_visualization/
│   ├── generate_science_network_20260516.py
│   ├── science_network_data_20260516.json
│   └── network_20260516.html
│
├── Structural_break/
│   ├── science_official_structural_breaks.py
│   ├── science_official_model_panel.csv
│   ├── science_official_model_breaks.csv
│   ├── science_official_model_segments.csv
│   └── science_official_model_summary.json
│
├── CBDB_Publications_List.md
└── README.md
