#!/usr/bin/env python3
"""Spatial statistical analysis for science-official temporal clusters (Break3).

Includes spatial centroids, dispersion metrics, permutation tests for distance,
kNN Moran's I spatial autocorrelation, and multinomial logistic regression.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "analysis" / "outputs"
DEFAULT_INPUT = OUT_DIR / "alaam_science_officials_break3_induced_nodes.csv"

# Temporal cluster definitions
CLUSTERS = [
    ("cluster1_song_and_before", "Song and before", "#2b6cb0"),
    ("cluster2_yuan_ming", "Yuan and Ming", "#b7791f"),
    ("cluster3_qing_after", "Qing and after", "#2f855a"),
]
CLUSTER_ORDER = [key for key, _label, _color in CLUSTERS]
CLUSTER_LABELS = {key: label for key, label, _color in CLUSTERS}
CLUSTER_COLORS = {key: color for key, _label, color in CLUSTERS}

# Variable controls
BUREAU_DUMMIES = [
    "bureau_astronomical",
    "bureau_medical",
    "bureau_hydraulic",
    "bureau_military_industrial",
    "bureau_construction",
]
ENTRY_CONTROLS = [
    "has_civil_exam",
    "has_yin_privilege",
    "has_recommendation",
    "has_technical_exam",
]
NETWORK_CONTROLS = ["degree_z", "clustering_z", "eigenvector_z", "betweenness_z"]

EARTH_RADIUS_KM = 6371.0088


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--k-neighbors", type=int, default=8)
    parser.add_argument("--permutations", type=int, default=999)
    parser.add_argument("--seed", type=int, default=20260608)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--model-iterations", type=int, default=5000)
    parser.add_argument("--model-l2", type=float, default=1e-4)
    return parser.parse_args()


def safe_pct(part: int | float, total: int | float) -> float:
    """Calculate percentage, returning 0 if total is zero."""
    return round(float(part) / float(total) * 100.0, 4) if total else 0.0


def safe_z(values: pd.Series) -> pd.Series:
    """Z-score normalization handling zero variance and missing values safely."""
    values = pd.to_numeric(values, errors="coerce")
    mean = values.mean()
    std = values.std(ddof=0)
    if not std or np.isnan(std):
        return pd.Series(np.zeros(len(values)), index=values.index)
    return (values - mean) / std


def haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Calculate great-circle distance between two coordinates in km."""
    lon1_r, lat1_r, lon2_r, lat2_r = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2_r - lon1_r
    dlat = lat2_r - lat1_r
    a = math.sin(dlat / 2.0) ** 2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * math.asin(math.sqrt(min(1.0, a)))


def haversine_vec_km(lon: np.ndarray, lat: np.ndarray, lon0: float, lat0: float) -> np.ndarray:
    """Vectorized Haversine distance from coordinate arrays to a single reference point."""
    lon_r = np.radians(lon.astype(float))
    lat_r = np.radians(lat.astype(float))
    lon0_r = math.radians(float(lon0))
    lat0_r = math.radians(float(lat0))
    dlon = lon0_r - lon_r
    dlat = lat0_r - lat_r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat_r) * math.cos(lat0_r) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.minimum(1.0, a)))


def load_nodes(path: Path) -> pd.DataFrame:
    """Load node data and filter out invalid/zero coordinates."""
    df = pd.read_csv(path)
    missing = [cluster for cluster in CLUSTER_ORDER if cluster not in set(df["cluster"])]
    if missing:
        raise ValueError(f"Missing expected clusters: {missing}")
    
    for col in ["x_coord", "y_coord"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        
    df["raw_geo"] = (
        df["has_geo"].fillna(0).astype(int).eq(1)
        & df["x_coord"].notna()
        & df["y_coord"].notna()
    )
    df["invalid_zero_coord"] = df["raw_geo"] & df["x_coord"].eq(0) & df["y_coord"].eq(0)
    df["valid_geo"] = df["raw_geo"] & ~df["invalid_zero_coord"]
    df["invalid_geo_reason"] = ""
    df.loc[~df["raw_geo"], "invalid_geo_reason"] = "missing_coordinates"
    df.loc[df["invalid_zero_coord"], "invalid_geo_reason"] = "zero_zero_coordinate"
    df["cluster_label"] = df["cluster"].map(CLUSTER_LABELS)
    return df


def write_nodes(df: pd.DataFrame, out_path: Path) -> None:
    """Export cleaned node data to CSV."""
    columns = [
        "id", "name_zh", "name_en", "index_year", "cluster", "cluster_label",
        "dynasty_zh", "bureau_type", "bureau_zh", "is_high_official",
        "has_civil_exam", "has_yin_privilege", "has_recommendation", "has_technical_exam",
        "has_geo", "raw_geo", "valid_geo", "invalid_geo_reason", "x_coord", "y_coord",
        "addr_name_chn", "degree", "clustering", "eigenvector", "betweenness",
        "degree_z", "clustering_z", "eigenvector_z", "betweenness_z",
    ]
    extra = [col for col in BUREAU_DUMMIES if col in df.columns]
    df[columns + extra].to_csv(out_path, index=False)


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute centroids, bounding boxes, and dispersion metrics per cluster."""
    rows = []
    total_n = len(df)
    for cluster in CLUSTER_ORDER:
        all_cluster = df[df["cluster"].eq(cluster)]
        valid = all_cluster[all_cluster["valid_geo"]].copy()
        raw_geo = int(all_cluster["raw_geo"].sum())
        zero = int(all_cluster["invalid_zero_coord"].sum())
        
        row: dict[str, object] = {
            "cluster": cluster,
            "cluster_label": CLUSTER_LABELS[cluster],
            "total_nodes": len(all_cluster),
            "share_of_all_nodes_pct": safe_pct(len(all_cluster), total_n),
            "raw_geo_nodes": raw_geo,
            "invalid_zero_coord_nodes": zero,
            "valid_geo_nodes": len(valid),
            "valid_geo_share_of_cluster_pct": safe_pct(len(valid), len(all_cluster)),
        }
        if valid.empty:
            rows.append(row)
            continue
            
        x = valid["x_coord"].to_numpy(dtype=float)
        y = valid["y_coord"].to_numpy(dtype=float)
        centroid_x = float(x.mean())
        centroid_y = float(y.mean())
        dist = haversine_vec_km(x, y, centroid_x, centroid_y)
        top_location = (
            valid["addr_name_chn"].fillna("").replace("", "Unknown").value_counts().index[0]
        )
        row.update(
            {
                "centroid_x": centroid_x,
                "centroid_y": centroid_y,
                "x_min": float(x.min()),
                "x_max": float(x.max()),
                "y_min": float(y.min()),
                "y_max": float(y.max()),
                "x_std": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
                "y_std": float(y.std(ddof=1)) if len(y) > 1 else 0.0,
                "standard_distance_km": float(np.sqrt(np.mean(dist**2))),
                "mean_distance_to_centroid_km": float(dist.mean()),
                "median_distance_to_centroid_km": float(np.median(dist)),
                "top_location": top_location,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def build_top_locations(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """Get the top N geographical locations for each cluster."""
    rows = []
    valid = df[df["valid_geo"]].copy()
    valid["location"] = valid["addr_name_chn"].fillna("").replace("", "Unknown")
    
    for cluster in CLUSTER_ORDER:
        cluster_df = valid[valid["cluster"].eq(cluster)]
        counts = cluster_df["location"].value_counts().head(top_n)
        total = len(cluster_df)
        for rank, (location, count) in enumerate(counts.items(), start=1):
            rows.append(
                {
                    "cluster": cluster,
                    "cluster_label": CLUSTER_LABELS[cluster],
                    "rank": rank,
                    "addr_name_chn": location,
                    "count": int(count),
                    "share_of_valid_cluster_pct": safe_pct(int(count), total),
                }
            )
    return pd.DataFrame(rows)


def centroid_by_cluster(coords: np.ndarray, labels: np.ndarray) -> dict[str, tuple[float, float]]:
    """Compute mean (x, y) centroid coordinates for each cluster."""
    centroids: dict[str, tuple[float, float]] = {}
    for cluster in CLUSTER_ORDER:
        mask = labels == cluster
        if mask.any():
            centroids[cluster] = (float(coords[mask, 0].mean()), float(coords[mask, 1].mean()))
    return centroids


def centroid_distance_rows(df: pd.DataFrame, permutations: int, seed: int) -> pd.DataFrame:
    """Permutation test for pairwise distances between cluster centroids."""
    valid = df[df["valid_geo"]].copy()
    coords = valid[["x_coord", "y_coord"]].to_numpy(dtype=float)
    labels = valid["cluster"].to_numpy()
    rng = np.random.default_rng(seed)
    
    observed = centroid_by_cluster(coords, labels)
    pairs = [
        (CLUSTER_ORDER[0], CLUSTER_ORDER[1]),
        (CLUSTER_ORDER[0], CLUSTER_ORDER[2]),
        (CLUSTER_ORDER[1], CLUSTER_ORDER[2]),
    ]
    observed_distances = {
        pair: haversine_km(
            observed[pair[0]][0], observed[pair[0]][1],
            observed[pair[1]][0], observed[pair[1]][1],
        )
        for pair in pairs
    }
    greater_equal = {pair: 0 for pair in pairs}
    perm_sums = {pair: 0.0 for pair in pairs}
    perm_sq_sums = {pair: 0.0 for pair in pairs}

    for _ in range(permutations):
        shuffled = rng.permutation(labels)
        perm_centroids = centroid_by_cluster(coords, shuffled)
        for pair in pairs:
            dist = haversine_km(
                perm_centroids[pair[0]][0], perm_centroids[pair[0]][1],
                perm_centroids[pair[1]][0], perm_centroids[pair[1]][1],
            )
            perm_sums[pair] += dist
            perm_sq_sums[pair] += dist * dist
            if dist >= observed_distances[pair]:
                greater_equal[pair] += 1

    rows = []
    for pair in pairs:
        mean = perm_sums[pair] / permutations if permutations else float("nan")
        var = perm_sq_sums[pair] / permutations - mean * mean if permutations else float("nan")
        rows.append(
            {
                "cluster_a": pair[0],
                "cluster_a_label": CLUSTER_LABELS[pair[0]],
                "cluster_b": pair[1],
                "cluster_b_label": CLUSTER_LABELS[pair[1]],
                "observed_centroid_distance_km": observed_distances[pair],
                "permutation_mean_distance_km": mean,
                "permutation_sd_distance_km": math.sqrt(max(0.0, var)) if permutations else float("nan"),
                "permutations": permutations,
                "p_value_greater_equal": (greater_equal[pair] + 1) / (permutations + 1),
                "seed": seed,
            }
        )
    return pd.DataFrame(rows)


def knn_indices(coords: np.ndarray, k: int) -> np.ndarray:
    """Find K-nearest neighbor indices using Euclidean distance."""
    n = coords.shape[0]
    if not 0 < k < n:
        raise ValueError(f"k-neighbors must be between 1 and n-1, got k={k}, n={n}")
    neighbors = np.empty((n, k), dtype=np.int32)
    for i in range(n):
        diff = coords - coords[i]
        dist2 = np.einsum("ij,ij->i", diff, diff)
        dist2[i] = np.inf
        idx = np.argpartition(dist2, k)[:k]
        idx = idx[np.argsort(dist2[idx])]
        neighbors[i, :] = idx
    return neighbors


def morans_i(binary: np.ndarray, neighbors: np.ndarray) -> float:
    """Compute Moran's I spatial autocorrelation given a kNN spatial matrix."""
    values = binary.astype(float)
    centered = values - values.mean()
    denominator = float(np.sum(centered**2))
    if denominator == 0.0:
        return float("nan")
    neighbor_values = centered[neighbors]
    numerator = float(np.sum(centered[:, None] * neighbor_values) / neighbors.shape[1])
    n = len(values)
    return (n / float(n)) * numerator / denominator


def morans_i_rows(df: pd.DataFrame, k: int, permutations: int, seed: int) -> pd.DataFrame:
    """Permutation test for spatial autocorrelation (Moran's I) per cluster."""
    valid = df[df["valid_geo"]].copy()
    coords = valid[["x_coord", "y_coord"]].to_numpy(dtype=float)
    labels = valid["cluster"].to_numpy()
    neighbors = knn_indices(coords, k)
    rng = np.random.default_rng(seed)
    
    rows = []
    for cluster in CLUSTER_ORDER:
        binary = (labels == cluster).astype(int)
        observed = morans_i(binary, neighbors)
        ge = 0
        perm_values = []
        for _ in range(permutations):
            perm_binary = rng.permutation(binary)
            perm_i = morans_i(perm_binary, neighbors)
            perm_values.append(perm_i)
            if perm_i >= observed:
                ge += 1
        perm_arr = np.asarray(perm_values, dtype=float)
        rows.append(
            {
                "cluster": cluster,
                "cluster_label": CLUSTER_LABELS[cluster],
                "valid_geo_nodes": int(binary.sum()),
                "k_neighbors": k,
                "observed_morans_i": observed,
                "expected_morans_i": -1.0 / (len(binary) - 1),
                "permutation_mean_i": float(perm_arr.mean()),
                "permutation_sd_i": float(perm_arr.std(ddof=1)),
                "permutations": permutations,
                "p_value_greater_equal": (ge + 1) / (permutations + 1),
                "seed": seed,
            }
        )
    return pd.DataFrame(rows)


def softmax_with_baseline(eta: np.ndarray) -> np.ndarray:
    """Softmax probabilities with baseline category fixed to 0."""
    all_eta = np.column_stack([np.zeros(eta.shape[0]), eta])
    max_eta = all_eta.max(axis=1, keepdims=True)
    exp_eta = np.exp(all_eta - max_eta)
    return exp_eta / exp_eta.sum(axis=1, keepdims=True)


def fit_multinomial_baseline(
    x: np.ndarray,
    y: np.ndarray,
    max_iter: int,
    l2: float,
    seed: int,
) -> tuple[np.ndarray, dict[str, object]]:
    """Fit multinomial logit regression with L2 regularization using Adam."""
    n, p = x.shape
    k_minus_one = len(CLUSTER_ORDER) - 1
    y_nonbase = np.zeros((n, k_minus_one), dtype=float)
    for category in range(1, len(CLUSTER_ORDER)):
        y_nonbase[:, category - 1] = y == category
        
    rng = np.random.default_rng(seed)
    beta = rng.normal(0.0, 0.001, size=(p, k_minus_one))
    m, v = np.zeros_like(beta), np.zeros_like(beta)
    lr = 0.03
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    
    penalty_mask = np.ones_like(beta)
    penalty_mask[0, :] = 0.0  # Do not penalize intercepts
    
    previous = float("inf")
    converged = False
    final_loss = float("nan")
    iteration = 0

    for iteration in range(1, max_iter + 1):
        eta = x @ beta
        probs = softmax_with_baseline(eta)
        loss = -float(np.mean(np.log(np.maximum(probs[np.arange(n), y], 1e-15))))
        loss += 0.5 * l2 * float(np.sum((beta * penalty_mask) ** 2))
        
        gradient = x.T @ (probs[:, 1:] - y_nonbase) / n + l2 * beta * penalty_mask
        m = beta1 * m + (1.0 - beta1) * gradient
        v = beta2 * v + (1.0 - beta2) * (gradient**2)
        m_hat = m / (1.0 - beta1**iteration)
        v_hat = v / (1.0 - beta2**iteration)
        beta -= lr * m_hat / (np.sqrt(v_hat) + eps)

        if iteration % 100 == 0:
            if abs(previous - loss) < 1e-8:
                converged = True
                final_loss = loss
                break
            previous = loss
        final_loss = loss

    probs = softmax_with_baseline(x @ beta)
    log_likelihood = float(np.sum(np.log(np.maximum(probs[np.arange(n), y], 1e-15))))
    predictions = probs.argmax(axis=1)
    n_params = beta.size
    
    diagnostics = {
        "model_converged": converged,
        "model_iterations": iteration,
        "log_likelihood": log_likelihood,
        "mean_negative_log_likelihood": -log_likelihood / n,
        "aic": 2 * n_params - 2 * log_likelihood,
        "bic": math.log(n) * n_params - 2 * log_likelihood,
        "training_accuracy": float(np.mean(predictions == y)),
        "l2_penalty": l2,
        "n_model_rows": n,
        "n_model_features": p,
    }
    return beta, diagnostics


def build_model_outputs(
    df: pd.DataFrame,
    max_iter: int,
    l2: float,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Build polynomial spatial terms and fit multinomial logistic regression."""
    valid = df[df["valid_geo"]].copy()
    valid["x_model_z"] = safe_z(valid["x_coord"])
    valid["y_model_z"] = safe_z(valid["y_coord"])
    valid["xy_model_z"] = valid["x_model_z"] * valid["y_model_z"]
    valid["x_model_z_sq"] = valid["x_model_z"] ** 2
    valid["y_model_z_sq"] = valid["y_model_z"] ** 2
    
    candidate_features = [
        "intercept", "x_model_z", "y_model_z", "x_model_z_sq", "y_model_z_sq",
        "xy_model_z", "is_high_official", *ENTRY_CONTROLS, *NETWORK_CONTROLS,
        *[col for col in BUREAU_DUMMIES if col in valid.columns],
    ]
    valid["intercept"] = 1.0
    features = []
    for feature in candidate_features:
        if feature not in valid.columns:
            continue
        values = pd.to_numeric(valid[feature], errors="coerce").fillna(0.0)
        if feature != "intercept":
            std = values.std(ddof=0)
            if std == 0 or np.isnan(std):
                continue
        valid[feature] = values
        features.append(feature)

    x = valid[features].to_numpy(dtype=float)
    y = valid["cluster"].map({cluster: i for i, cluster in enumerate(CLUSTER_ORDER)}).to_numpy(dtype=int)
    beta, diagnostics = fit_multinomial_baseline(x, y, max_iter=max_iter, l2=l2, seed=seed)

    rows = []
    for category_index, cluster in enumerate(CLUSTER_ORDER[1:], start=1):
        for feature_index, feature in enumerate(features):
            coefficient = float(beta[feature_index, category_index - 1])
            rows.append(
                {
                    "outcome_cluster": cluster,
                    "outcome_cluster_label": CLUSTER_LABELS[cluster],
                    "baseline_cluster": CLUSTER_ORDER[0],
                    "baseline_cluster_label": CLUSTER_LABELS[CLUSTER_ORDER[0]],
                    "feature": feature,
                    "coefficient": coefficient,
                    "odds_ratio": math.exp(coefficient) if -700 < coefficient < 700 else float("inf"),
                    "model": "baseline_category_multinomial_logit_numpy_l2",
                }
            )
    diagnostics["model_features"] = ";".join(features)
    return pd.DataFrame(rows), diagnostics


def diagnostics_rows(
    df: pd.DataFrame,
    model_diagnostics: dict[str, object],
    k_neighbors: int,
    permutations: int,
    seed: int,
) -> pd.DataFrame:
    """Format run diagnostics into a structured summary table."""
    rows: list[dict[str, object]] = []

    def add(metric: str, value: object) -> None:
        rows.append({"metric": metric, "value": value})

    add("total_nodes", len(df))
    add("raw_geo_nodes", int(df["raw_geo"].sum()))
    add("invalid_zero_coord_nodes", int(df["invalid_zero_coord"].sum()))
    add("valid_geo_nodes", int(df["valid_geo"].sum()))
    add("valid_geo_share_pct", safe_pct(int(df["valid_geo"].sum()), len(df)))
    add("k_neighbors", k_neighbors)
    add("permutations", permutations)
    add("seed", seed)
    add("external_chgis_used", "no")
    add("external_hydraulic_population_frontier_variables_used", "no")
    
    for cluster in CLUSTER_ORDER:
        cluster_df = df[df["cluster"].eq(cluster)]
        add(f"{cluster}_total_nodes", len(cluster_df))
        add(f"{cluster}_valid_geo_nodes", int(cluster_df["valid_geo"].sum()))
        
    for key, value in model_diagnostics.items():
        add(key, value)
    return pd.DataFrame(rows)


def svg_escape(text: object) -> str:
    """Escape XML special characters for SVG rendering."""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def write_scatter_svg(df: pd.DataFrame, out_path: Path) -> None:
    """Render spatial cluster distribution as an SVG scatter plot."""
    valid = df[df["valid_geo"]].copy()
    width, height = 960, 640
    left, right, top, bottom = 80, 180, 50, 80
    plot_w = width - left - right
    plot_h = height - top - bottom
    
    x_min, x_max = valid["x_coord"].min(), valid["x_coord"].max()
    y_min, y_max = valid["y_coord"].min(), valid["y_coord"].max()
    x_pad = (x_max - x_min) * 0.05
    y_pad = (y_max - y_min) * 0.05
    x_min -= x_pad
    x_max += x_pad
    y_min -= y_pad
    y_max += y_pad

    def sx(x: float) -> float:
        return left + (float(x) - x_min) / (x_max - x_min) * plot_w

    def sy(y: float) -> float:
        return top + (y_max - float(y)) / (y_max - y_min) * plot_h

    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#1f2933}.axis{stroke:#444;stroke-width:1}.grid{stroke:#ddd;stroke-width:0.6}.legend{font-size:14px}.title{font-size:22px;font-weight:bold}.label{font-size:13px}.tick{font-size:11px}</style>',
        f'<text class="title" x="{left}" y="30">Spatial Distribution of Science-Official Clusters</text>',
    ]
    
    for i in range(6):
        x_val = x_min + (x_max - x_min) * i / 5
        y_val = y_min + (y_max - y_min) * i / 5
        x_pos = sx(x_val)
        y_pos = sy(y_val)
        parts.append(f'<line class="grid" x1="{x_pos:.2f}" y1="{top}" x2="{x_pos:.2f}" y2="{top + plot_h}"/>')
        parts.append(f'<line class="grid" x1="{left}" y1="{y_pos:.2f}" x2="{left + plot_w}" y2="{y_pos:.2f}"/>')
        parts.append(f'<text class="tick" x="{x_pos:.2f}" y="{top + plot_h + 18}" text-anchor="middle">{x_val:.1f}</text>')
        parts.append(f'<text class="tick" x="{left - 10}" y="{sy(y_val) + 4:.2f}" text-anchor="end">{y_val:.1f}</text>')
        
    parts.append(f'<line class="axis" x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}"/>')
    parts.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}"/>')
    parts.append(f'<text class="label" x="{left + plot_w / 2}" y="{height - 28}" text-anchor="middle">Longitude / x_coord</text>')
    parts.append(f'<text class="label" transform="translate(24 {top + plot_h / 2}) rotate(-90)" text-anchor="middle">Latitude / y_coord</text>')

    for cluster in CLUSTER_ORDER:
        cluster_df = valid[valid["cluster"].eq(cluster)]
        color = CLUSTER_COLORS[cluster]
        for row in cluster_df.itertuples(index=False):
            title = f"{getattr(row, 'name_zh', '')} {getattr(row, 'name_en', '')} ({CLUSTER_LABELS[cluster]})"
            parts.append(
                f'<circle cx="{sx(row.x_coord):.2f}" cy="{sy(row.y_coord):.2f}" r="2.1" '
                f'fill="{color}" fill-opacity="0.58"><title>{svg_escape(title)}</title></circle>'
            )

    legend_x = left + plot_w + 30
    legend_y = top + 20
    parts.append(f'<text class="legend" x="{legend_x}" y="{legend_y - 10}" font-weight="bold">Cluster</text>')
    for i, cluster in enumerate(CLUSTER_ORDER):
        y = legend_y + i * 28
        count = int(valid["cluster"].eq(cluster).sum())
        parts.append(f'<circle cx="{legend_x}" cy="{y}" r="6" fill="{CLUSTER_COLORS[cluster]}"/>')
        parts.append(f'<text class="legend" x="{legend_x + 14}" y="{y + 5}">{svg_escape(CLUSTER_LABELS[cluster])} ({count})</text>')
        
    parts.append("</svg>")
    out_path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    """Run spatial cluster analysis pipeline."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = load_nodes(args.input)
    
    nodes_path = args.output_dir / "spatial_cluster_nodes.csv"
    summary_path = args.output_dir / "spatial_cluster_summary.csv"
    top_locations_path = args.output_dir / "spatial_cluster_top_locations.csv"
    centroid_tests_path = args.output_dir / "spatial_cluster_centroid_tests.csv"
    morans_path = args.output_dir / "spatial_cluster_morans_i.csv"
    model_path = args.output_dir / "spatial_cluster_multinomial_logit.csv"
    diagnostics_path = args.output_dir / "spatial_cluster_model_diagnostics.csv"
    svg_path = args.output_dir / "spatial_cluster_scatter.svg"

    write_nodes(df, nodes_path)
    summary = build_summary(df)
    top_locations = build_top_locations(df, args.top_n)
    centroid_tests = centroid_distance_rows(df, args.permutations, args.seed)
    morans = morans_i_rows(df, args.k_neighbors, args.permutations, args.seed)
    
    model, model_diagnostics = build_model_outputs(
        df,
        max_iter=args.model_iterations,
        l2=args.model_l2,
        seed=args.seed,
    )
    diagnostics = diagnostics_rows(
        df,
        model_diagnostics,
        k_neighbors=args.k_neighbors,
        permutations=args.permutations,
        seed=args.seed,
    )

    summary.to_csv(summary_path, index=False)
    top_locations.to_csv(top_locations_path, index=False)
    centroid_tests.to_csv(centroid_tests_path, index=False)
    morans.to_csv(morans_path, index=False)
    model.to_csv(model_path, index=False)
    diagnostics.to_csv(diagnostics_path, index=False, quoting=csv.QUOTE_MINIMAL)
    write_scatter_svg(df, svg_path)

    output_paths = [
        nodes_path, summary_path, top_locations_path, centroid_tests_path,
        morans_path, model_path, diagnostics_path, svg_path,
    ]

    print("Spatial cluster analysis complete.")
    for path in output_paths:
        print(path)


if __name__ == "__main__":
    main()