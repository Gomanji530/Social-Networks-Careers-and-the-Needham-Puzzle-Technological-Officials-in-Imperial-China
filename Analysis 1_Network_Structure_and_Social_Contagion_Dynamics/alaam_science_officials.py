#!/usr/bin/env python3
"""ALAAM-style MPLE models for science-official temporal clusters.

This script isolates scientific/technical official nodes from CBDB data and constructs 
a sub-network. It fits Autologistic Actor-Attribute Models (ALAAM) using Maximum 
Pseudo-Likelihood Estimation (MPLE) approximations.

The temporal-cluster membership serves as the outcome variable across defined historical
time breaks.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
import sys
import unittest
import warnings
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

# Configure path constants for database and data file dependencies
ROOT = Path(__file__).resolve().parents[1]
LOCAL_PACKAGES = ROOT / "analysis" / ".python_packages"
if LOCAL_PACKAGES.exists():
    sys.path.insert(0, str(LOCAL_PACKAGES))

import networkx as nx  # noqa: E402

NETWORK_JSON = ROOT / "network" / "science_network_data_20260516.json"
DB_PATH = ROOT / "latest" / "cbdb_20260516.sqlite3"
OUT_DIR = ROOT / "analysis" / "outputs"

# Define historical temporal cluster boundaries
CLUSTER_SCHEMES = {
    "break4": [
        ("pre_song", -10_000, 959, "Before Song"),
        ("song_to_yuan", 960, 1278, "Song to Yuan"),
        ("yuan_to_ming", 1279, 1367, "Yuan to Ming"),
        ("ming_to_qing", 1368, 1911, "Ming to Qing"),
    ],
    "break3": [
        ("cluster1_song_and_before", -10_000, 1278, "Song and before"),
        ("cluster2_yuan_ming", 1279, 1643, "Yuan and Ming"),
        ("cluster3_qing_after", 1644, 3000, "Qing and after"),
    ],
}

# CBDB entry code categorizations
CIVIL_EXAM_CODES = {26, 28, 29, 36, 37, 39, 42, 47, 48, 49, 50, 51, 52, 53, 54, 56, 57}
MEDICAL_EXAM_CODES = {43}
MILITARY_EXAM_CODES = {44, 46, 328}
RECOMMENDATION_CODES = {101, 225, 310, 319}
YIN_PRIVILEGE_CODES = {8, 59, 60, 62, 118, 138, 157, 163, 197, 198, 317, 318}

# Baseline feature variables for regression models
BASE_FEATURES = [
    "contagion",
    "indirect_contagion",
    "degree_z",
    "betweenness_z",
    "clustering_z",
    "eigenvector_z",
    "is_high_official",
    "has_civil_exam",
    "has_yin_privilege",
    "has_recommendation",
    "has_technical_exam",
    "has_geo",
    "x_coord_z",
    "y_coord_z",
]

BUREAU_DUMMIES = [
    "bureau_astronomical",
    "bureau_medical",
    "bureau_hydraulic",
    "bureau_military_industrial",
    "bureau_construction",
]


def endpoint_id(value: Any) -> int:
    """Extract integer ID from an edge endpoint, supporting dicts or primitives."""
    if isinstance(value, dict):
        return int(value["id"])
    return int(value)


def pct(part: int | float, total: int | float) -> float:
    """Calculate percentage safely avoiding division by zero."""
    return round(part / total * 100, 2) if total else 0.0


def cluster_for_year(year: Any, clusters: list[tuple[str, int, int, str]]) -> str:
    """Determine the temporal cluster key for a given index year."""
    if year is None or year == "":
        return "out_of_scope"
    year_i = int(year)
    for key, start, end, _label in clusters:
        if start <= year_i <= end:
            return key
    return "out_of_scope"


def load_network() -> dict[str, Any]:
    """Load the primary science network JSON dataset."""
    return json.loads(NETWORK_JSON.read_text(encoding="utf-8"))


def chunked(values: list[int], size: int = 800):
    """Yield chunks of a list to avoid SQLite parameter limit limits."""
    for start in range(0, len(values), size):
        yield values[start : start + size]


def fetch_covariates(person_ids: list[int]) -> dict[int, dict[str, Any]]:
    """Query CBDB database to extract entry route and geographic covariates."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows: dict[int, dict[str, Any]] = {
        pid: {
            "has_civil_exam": 0,
            "has_yin_privilege": 0,
            "has_recommendation": 0,
            "has_technical_exam": 0,
            "has_geo": 0,
            "x_coord": None,
            "y_coord": None,
            "addr_name_chn": "",
        }
        for pid in person_ids
    }

    # Fetch entry exam indicators
    for chunk in chunked(person_ids):
        placeholders = ",".join("?" for _ in chunk)
        sql = f"select c_personid, c_entry_code from ENTRY_DATA where c_personid in ({placeholders})"
        for row in conn.execute(sql, chunk):
            pid = int(row["c_personid"])
            code = int(row["c_entry_code"])
            if code in CIVIL_EXAM_CODES:
                rows[pid]["has_civil_exam"] = 1
            if code in YIN_PRIVILEGE_CODES:
                rows[pid]["has_yin_privilege"] = 1
            if code in RECOMMENDATION_CODES:
                rows[pid]["has_recommendation"] = 1
            if code in MEDICAL_EXAM_CODES or code in MILITARY_EXAM_CODES:
                rows[pid]["has_technical_exam"] = 1

    # Fetch primary geographic location coordinates
    for chunk in chunked(person_ids):
        placeholders = ",".join("?" for _ in chunk)
        sql = f"""
            select b.c_personid, a.c_name_chn, a.x_coord, a.y_coord, b.c_addr_type, b.c_sequence
            from BIOG_ADDR_DATA b
            join ADDR_CODES a on b.c_addr_id = a.c_addr_id
            where b.c_personid in ({placeholders})
              and a.x_coord is not null
              and a.y_coord is not null
            order by b.c_personid, case when b.c_addr_type = 1 then 0 else 1 end, b.c_sequence
        """
        for row in conn.execute(sql, chunk):
            pid = int(row["c_personid"])
            if not rows[pid]["has_geo"]:
                rows[pid]["has_geo"] = 1
                rows[pid]["x_coord"] = float(row["x_coord"])
                rows[pid]["y_coord"] = float(row["y_coord"])
                rows[pid]["addr_name_chn"] = row["c_name_chn"] or ""

    conn.close()
    return rows


def build_full_adjacency(data: dict[str, Any]) -> dict[int, set[int]]:
    """Build complete graph adjacency mapping from raw network edges."""
    adjacency: dict[int, set[int]] = defaultdict(set)
    for edge in data["edges"]:
        source = endpoint_id(edge["source"])
        target = endpoint_id(edge["target"])
        if source != target:
            adjacency[source].add(target)
            adjacency[target].add(source)
    return adjacency


def build_science_graph(data: dict[str, Any], node_ids: set[int], graph_mode: str) -> nx.Graph:
    """Construct either an induced or co-neighbor projection NetworkX graph."""
    graph = nx.Graph()
    graph.add_nodes_from(sorted(node_ids))

    if graph_mode == "induced":
        for edge in data["edges"]:
            source = endpoint_id(edge["source"])
            target = endpoint_id(edge["target"])
            if source in node_ids and target in node_ids and source != target:
                graph.add_edge(source, target, weight=1)
        return graph

    if graph_mode != "projection":
        raise ValueError(f"Unknown graph mode: {graph_mode}")

    # Build 2-hop projection graph through shared non-science intermediate nodes
    full_adjacency = build_full_adjacency(data)
    pair_weights: Counter[tuple[int, int]] = Counter()
    for _hub, neighbors in full_adjacency.items():
        science_neighbors = sorted(n for n in neighbors if n in node_ids)
        if len(science_neighbors) < 2:
            continue
        for source, target in combinations(science_neighbors, 2):
            pair_weights[(source, target)] += 1
    for (source, target), weight in pair_weights.items():
        graph.add_edge(source, target, weight=weight)
    return graph


def safe_z(values: pd.Series) -> pd.Series:
    """Compute Z-score normalization safely handling zero variance or NaN values."""
    values = pd.to_numeric(values, errors="coerce")
    values = values.fillna(values.mean())
    std = values.std(ddof=0)
    if not std or np.isnan(std):
        return pd.Series(np.zeros(len(values)), index=values.index)
    return (values - values.mean()) / std


def compute_centralities(graph: nx.Graph, seed: int) -> pd.DataFrame:
    """Calculate network centrality metrics (degree, clustering, eigenvector, betweenness)."""
    degree = dict(graph.degree())
    clustering = nx.clustering(graph)
    if graph.number_of_edges():
        eigenvector = nx.eigenvector_centrality(graph, max_iter=2000, tol=1e-8)
        k = min(512, graph.number_of_nodes())
        betweenness = nx.betweenness_centrality(graph, k=k, normalized=True, seed=seed)
    else:
        eigenvector = {node: 0.0 for node in graph.nodes}
        betweenness = {node: 0.0 for node in graph.nodes}

    rows = []
    for node in sorted(graph.nodes):
        rows.append(
            {
                "id": node,
                "degree": degree.get(node, 0),
                "clustering": clustering.get(node, 0.0),
                "eigenvector": eigenvector.get(node, 0.0),
                "betweenness": betweenness.get(node, 0.0),
            }
        )
    df = pd.DataFrame(rows)
    for raw, zed in [
        ("degree", "degree_z"),
        ("clustering", "clustering_z"),
        ("eigenvector", "eigenvector_z"),
        ("betweenness", "betweenness_z"),
    ]:
        df[zed] = safe_z(df[raw])
    return df


def node_table(
    data: dict[str, Any],
    graph: nx.Graph,
    clusters: list[tuple[str, int, int, str]],
) -> pd.DataFrame:
    """Build standardized pandas DataFrame containing node attributes and centralities."""
    nodes = []
    covariates = fetch_covariates(sorted(graph.nodes))
    for node in data["nodes"]:
        pid = int(node["id"])
        if pid not in graph:
            continue
        cluster = cluster_for_year(node.get("indexYear"), clusters)
        cov = covariates[pid]
        bureau_type = node.get("bureauType") or ""
        row = {
            "id": pid,
            "name_zh": node.get("nameZh", ""),
            "name_en": node.get("nameEn", ""),
            "index_year": node.get("indexYear"),
            "cluster": cluster,
            "dynasty_zh": node.get("dynastyZh", ""),
            "bureau_type": bureau_type,
            "bureau_zh": node.get("bureauZh", ""),
            "is_high_official": int(bool(node.get("isHighOfficial"))),
            **cov,
        }
        for dummy in BUREAU_DUMMIES:
            row[dummy] = 0
        if f"bureau_{bureau_type}" in BUREAU_DUMMIES:
            row[f"bureau_{bureau_type}"] = 1
        nodes.append(row)

    df = pd.DataFrame(nodes)
    centrality = compute_centralities(graph, seed=20260516)
    df = df.merge(centrality, on="id", how="left")
    df["x_coord_z"] = safe_z(df["x_coord"])
    df["y_coord_z"] = safe_z(df["y_coord"])
    for col in ["x_coord_z", "y_coord_z"]:
        df[col] = df[col].fillna(0.0)
    return df


def distance_two_sets(graph: nx.Graph) -> dict[int, set[int]]:
    """Compute 2-hop neighborhood node sets for indirect contagion metrics."""
    output: dict[int, set[int]] = {}
    for node in graph.nodes:
        direct = set(graph.neighbors(node))
        second: set[int] = set()
        for neighbor in direct:
            second.update(graph.neighbors(neighbor))
        second.discard(node)
        second.difference_update(direct)
        output[node] = second
    return output


def fit_cluster_mple(
    df: pd.DataFrame,
    graph: nx.Graph,
    cluster: str,
    feature_columns: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Fit ALAAM MPLE logistic regression model for a single temporal cluster."""
    y_by_id = {int(row.id): int(row.cluster == cluster) for row in df.itertuples(index=False)}
    distance_two = distance_two_sets(graph)
    model_df = df.copy()
    model_df["outcome"] = model_df["id"].map(y_by_id)
    model_df["contagion"] = model_df["id"].map(
        lambda pid: sum(y_by_id.get(neighbor, 0) for neighbor in graph.neighbors(int(pid)))
    )
    model_df["indirect_contagion"] = model_df["id"].map(
        lambda pid: sum(y_by_id.get(neighbor, 0) for neighbor in distance_two[int(pid)])
    )
    model_df["contagion"] = safe_z(model_df["contagion"])
    model_df["indirect_contagion"] = safe_z(model_df["indirect_contagion"])

    y = model_df["outcome"].to_numpy(dtype=int)
    x = model_df[feature_columns].to_numpy(dtype=float)
    if len(np.unique(y)) < 2:
        raise RuntimeError(f"Cluster {cluster} has no outcome variation")

    model = LogisticRegression(C=10.0, solver="lbfgs", max_iter=2000, fit_intercept=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(x, y)
        probabilities = model.predict_proba(x)[:, 1]
    probabilities = np.clip(probabilities, 1e-9, 1 - 1e-9)
    deviance = 2 * log_loss(y, probabilities, normalize=False, labels=[0, 1])

    coefficients = [float(model.intercept_[0]), *[float(v) for v in model.coef_[0]]]
    terms = ["activity", *feature_columns]
    rows = []
    for term, coef in zip(terms, coefficients):
        rows.append(
            {
                "cluster": cluster,
                "term": term,
                "coefficient": coef,
                "odds_ratio": math.exp(coef) if -50 < coef < 50 else "",
                "method": "ALAAM_MPLE_L2_logit",
            }
        )

    y_nodes = [pid for pid, value in y_by_id.items() if value]
    edge_contagion = sum(
        1
        for source, target in graph.edges
        if y_by_id.get(int(source), 0) and y_by_id.get(int(target), 0)
    )
    diagnostics = {
        "cluster": cluster,
        "node_count": graph.number_of_nodes(),
        "edge_count": graph.number_of_edges(),
        "outcome_ones": int(sum(y)),
        "outcome_share_pct": pct(int(sum(y)), len(y)),
        "deviance": deviance,
        "mean_predicted_probability": float(probabilities.mean()),
        "observed_activity": len(y_nodes),
        "observed_contagion_edges": edge_contagion,
    }
    return rows, diagnostics


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Helper utility to export dictionaries into CSV files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_mpnet_files(
    prefix: Path,
    df: pd.DataFrame,
    graph: nx.Graph,
    clusters: list[tuple[str, int, int, str]],
) -> None:
    """Export network structure and node attributes formatted for MPNet software."""
    edge_path = prefix.with_name(prefix.name + "_mpnet_edges.txt")
    attr_path = prefix.with_name(prefix.name + "_mpnet_attributes.csv")
    id_map = {node_id: idx + 1 for idx, node_id in enumerate(sorted(graph.nodes))}
    with edge_path.open("w", encoding="utf-8") as handle:
        for source, target in sorted(graph.edges):
            handle.write(f"{id_map[int(source)]} {id_map[int(target)]}\n")
    attr_df = df[["id", "name_zh", "cluster", "degree", "betweenness", "clustering", "eigenvector"]].copy()
    attr_df.insert(0, "mpnet_id", attr_df["id"].map(id_map))
    for key, _start, _end, _label in clusters:
        attr_df[key] = (attr_df["cluster"] == key).astype(int)
    attr_df.sort_values("mpnet_id").to_csv(attr_path, index=False)




def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit ALAAM MPLE models on science official clusters.")
    parser.add_argument("--graph-mode", choices=["induced", "projection"], default="induced")
    parser.add_argument("--cluster-scheme", choices=sorted(CLUSTER_SCHEMES), default="break4")
    parser.add_argument("--output-prefix", default="")
    parser.add_argument("--test", action="store_true", help="Run unit tests and exit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.test:
        print("Running unit tests...")
        suite = unittest.TestLoader().loadTestsFromTestCase(TestALAAMFunctions)
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        sys.exit(0 if result.wasSuccessful() else 1)

    data = load_network()
    clusters = CLUSTER_SCHEMES[args.cluster_scheme]

    retained_ids = {
        int(node["id"])
        for node in data["nodes"]
        if node.get("isMain") and cluster_for_year(node.get("indexYear"), clusters) != "out_of_scope"
    }
    graph = build_science_graph(data, retained_ids, args.graph_mode)
    df = node_table(data, graph, clusters)
    df = df[df["cluster"] != "out_of_scope"].copy()

    feature_columns = BASE_FEATURES + BUREAU_DUMMIES
    result_rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for cluster, _start, _end, _label in clusters:
        rows, diag = fit_cluster_mple(df, graph, cluster, feature_columns)
        result_rows.extend(rows)
        diagnostics.append(diag)

    prefix_name = args.output_prefix or f"alaam_science_officials_{args.cluster_scheme}_{args.graph_mode}"
    prefix = OUT_DIR / prefix_name
    nodes_path = prefix.with_name(prefix.name + "_nodes.csv")
    edges_path = prefix.with_name(prefix.name + "_edges.csv")
    results_path = prefix.with_name(prefix.name + "_mple_results.csv")
    diagnostics_path = prefix.with_name(prefix.name + "_diagnostics.csv")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.sort_values("id").to_csv(nodes_path, index=False)
    edge_rows = [
        {
            "source": int(source),
            "target": int(target),
            "weight": graph[source][target].get("weight", 1),
        }
        for source, target in sorted(graph.edges)
    ]
    write_csv(edges_path, edge_rows)
    write_csv(results_path, result_rows)
    write_csv(diagnostics_path, diagnostics)
    write_mpnet_files(prefix, df, graph, clusters)

    print(f"Wrote {nodes_path}")
    print(f"Wrote {edges_path}")
    print(f"Wrote {results_path}")
    print(f"Wrote {diagnostics_path}")


if __name__ == "__main__":
    main()