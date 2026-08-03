#!/usr/bin/env python3
"""Career trajectory analysis for science-official break3 temporal clusters.

Links break3 science-official nodes to CBDB postings, constructs ordered career 
sequences, identifies career types using weighted k-medoids sequence analysis, 
and models high-status political access.
"""

from __future__ import annotations

import argparse
import csv
import math
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "analysis" / "outputs"
DB_PATH = ROOT / "latest" / "cbdb_20260516.sqlite3"
DEFAULT_INPUT = OUT_DIR / "alaam_science_officials_break3_induced_nodes.csv"

# Temporal clusters and labeling
CLUSTERS = [
    ("cluster1_song_and_before", "Song and before", "#2b6cb0"),
    ("cluster2_yuan_ming", "Yuan and Ming", "#b7791f"),
    ("cluster3_qing_after", "Qing and after", "#2f855a"),
]
CLUSTER_ORDER = [key for key, _label, _color in CLUSTERS]
CLUSTER_LABELS = {key: label for key, label, _color in CLUSTERS}

# Variable groupings for regression models
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

# Keyword domain matching rules for classifying official postings
DOMAIN_RULES = [
    (
        "astronomical",
        [
            "司天", "太史", "欽天", "天文", "曆", "历",
            "astronom", "astrolog", "calendar", "observatory",
        ],
    ),
    (
        "medical",
        [
            "醫", "藥", "太醫", "medicine", "medical",
            "physician", "doctor", "pharmacy",
        ],
    ),
    (
        "hydraulic",
        [
            "都水", "水部", "水利", "河", "漕",
            "waterway", "irrigation", "river", "canal",
        ],
    ),
    (
        "military_industrial",
        [
            "軍器", "火器", "兵器", "武器",
            "armament", "armory", "weapon", "military",
        ],
    ),
    (
        "construction",
        [
            "工部", "將作", "營造", "修造",
            "works", "construction", "building", "architecture",
        ],
    ),
]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--career-types", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260608)
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


def chunked(values: list[int], size: int = 800) -> Iterable[list[int]]:
    """Yield chunks of a list to avoid SQLite parameter limit errors."""
    for start in range(0, len(values), size):
        yield values[start : start + size]


def load_nodes(path: Path) -> pd.DataFrame:
    """Load node dataset and format numerical control variables."""
    df = pd.read_csv(path)
    missing = [cluster for cluster in CLUSTER_ORDER if cluster not in set(df["cluster"])]
    if missing:
        raise ValueError(f"Missing expected clusters: {missing}")
    for col in [
        "is_high_official",
        *ENTRY_CONTROLS,
        *NETWORK_CONTROLS,
        *[col for col in BUREAU_DUMMIES if col in df.columns],
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df["cluster_label"] = df["cluster"].map(CLUSTER_LABELS)
    return df


def fetch_postings(db_path: Path, person_ids: list[int]) -> pd.DataFrame:
    """Query CBDB database for posting records linked to target person IDs."""
    conn = sqlite3.connect(db_path)
    frames = []
    for chunk in chunked(person_ids):
        placeholders = ",".join("?" for _ in chunk)
        sql = f"""
            select
                p.c_personid, p.c_office_id, p.c_posting_id, p.c_sequence,
                p.c_firstyear, p.c_lastyear, p.c_office_category_id, p.c_dy,
                o.c_office_chn, o.c_office_trans, o.c_office_pinyin,
                oc.c_category_desc, oc.c_category_desc_chn
            from POSTED_TO_OFFICE_DATA p
            left join OFFICE_CODES o on p.c_office_id = o.c_office_id
            left join OFFICE_CATEGORIES oc on p.c_office_category_id = oc.c_office_category_id
            where p.c_personid in ({placeholders})
        """
        frames.append(pd.read_sql_query(sql, conn, params=chunk))
    conn.close()
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def normalize_category(value: object) -> str:
    """Normalize office category strings."""
    if pd.isna(value) or str(value).strip() == "":
        return "missing_category"
    value_s = str(value).strip()
    return "unknown_category" if value_s.lower() == "unknown" else value_s


def office_domain(row: pd.Series) -> str:
    """Classify posting office title into a technical domain or 'other'."""
    text = " ".join(
        str(row.get(col, "") or "")
        for col in ["c_office_chn", "c_office_trans", "c_office_pinyin"]
    ).lower()
    for domain, keywords in DOMAIN_RULES:
        if any(keyword.lower() in text for keyword in keywords):
            return domain
    return "other"


def build_posting_table(nodes: pd.DataFrame, postings: pd.DataFrame) -> pd.DataFrame:
    """Merge postings with node data and order career sequence chronologically."""
    postings = postings.copy()
    for col in ["c_sequence", "c_firstyear", "c_lastyear", "c_office_category_id", "c_dy"]:
        postings[col] = pd.to_numeric(postings[col], errors="coerce")
    postings["office_category"] = postings["c_category_desc"].map(normalize_category)
    postings["office_category_chn"] = postings["c_category_desc_chn"].fillna("")
    postings["office_domain"] = postings.apply(office_domain, axis=1)
    
    postings["career_state"] = np.where(
        postings["office_domain"].ne("other"),
        "domain_" + postings["office_domain"],
        "category_" + postings["office_category"],
    )
    
    postings["has_sequence"] = postings["c_sequence"].notna()
    postings["has_firstyear"] = postings["c_firstyear"].notna() & postings["c_firstyear"].gt(0)
    postings["has_lastyear"] = postings["c_lastyear"].notna() & postings["c_lastyear"].gt(0)
    
    postings["order_sequence"] = postings["c_sequence"].fillna(10**9)
    postings["order_firstyear"] = postings["c_firstyear"].where(postings["has_firstyear"], 10**9)
    postings = postings.sort_values(
        ["c_personid", "order_sequence", "order_firstyear", "c_posting_id", "c_office_id"],
        kind="mergesort",
    )
    postings["sequence_position"] = postings.groupby("c_personid").cumcount() + 1
    
    keep_cols = [
        "id", "cluster", "cluster_label", "name_zh", "name_en", "index_year",
        "bureau_type", "bureau_zh", "is_high_official", *ENTRY_CONTROLS,
        *NETWORK_CONTROLS, *[col for col in BUREAU_DUMMIES if col in nodes.columns],
    ]
    return postings.merge(
        nodes[keep_cols], left_on="c_personid", right_on="id", how="left"
    )


def collapse_adjacent(states: Iterable[str]) -> list[str]:
    """Collapse adjacent duplicate states in a career sequence."""
    collapsed: list[str] = []
    for state in states:
        state_s = str(state)
        if not collapsed or collapsed[-1] != state_s:
            collapsed.append(state_s)
    return collapsed


def build_node_features(nodes: pd.DataFrame, postings: pd.DataFrame) -> pd.DataFrame:
    """Aggregate individual posting histories into node-level career features."""
    rows = []
    for pid, group in postings.groupby("c_personid", sort=False):
        states = [str(x) for x in group["career_state"]]
        collapsed = collapse_adjacent(states)
        first_years = group.loc[group["has_firstyear"], "c_firstyear"]
        last_years = group.loc[group["has_lastyear"], "c_lastyear"]
        
        first_year = float(first_years.min()) if not first_years.empty else np.nan
        last_year = float(last_years.max()) if not last_years.empty else np.nan
        span = last_year - first_year if not np.isnan(first_year) and not np.isnan(last_year) else np.nan
        
        domain_counts = group["office_domain"].value_counts()
        category_counts = group["office_category"].value_counts()
        
        rows.append(
            {
                "id": int(pid),
                "posting_count": len(group),
                "sequenced_posting_count": int(group["has_sequence"].sum()),
                "firstyear_posting_count": int(group["has_firstyear"].sum()),
                "lastyear_posting_count": int(group["has_lastyear"].sum()),
                "category_posting_count": int(group["c_office_category_id"].notna().sum()),
                "sequence_length": len(states),
                "collapsed_sequence_length": len(collapsed),
                "career_sequence": " > ".join(states),
                "collapsed_career_sequence": " > ".join(collapsed),
                "first_state": collapsed[0] if collapsed else "",
                "last_state": collapsed[-1] if collapsed else "",
                "dominant_domain": domain_counts.index[0] if not domain_counts.empty else "",
                "dominant_category": category_counts.index[0] if not category_counts.empty else "",
                "technical_domain_postings": int(group["office_domain"].ne("other").sum()),
                "technical_domain_share": (
                    float(group["office_domain"].ne("other").mean()) if len(group) else 0.0
                ),
                "has_astronomical_posting": int(group["office_domain"].eq("astronomical").any()),
                "has_medical_posting": int(group["office_domain"].eq("medical").any()),
                "has_hydraulic_posting": int(group["office_domain"].eq("hydraulic").any()),
                "has_military_industrial_posting": int(group["office_domain"].eq("military_industrial").any()),
                "has_construction_posting": int(group["office_domain"].eq("construction").any()),
                "first_posting_year": first_year,
                "last_posting_year": last_year,
                "career_span_years": span,
                "has_any_year": int(not np.isnan(first_year)),
                "has_two_or_more_sequenced_postings": int(group["has_sequence"].sum() >= 2),
            }
        )
    node_features = pd.DataFrame(rows)
    merged = nodes.merge(node_features, on="id", how="left")
    
    # Fill missing values for nodes without recorded postings
    for col in [
        "posting_count", "sequence_length", "collapsed_sequence_length",
        "sequenced_posting_count", "firstyear_posting_count", "lastyear_posting_count",
        "category_posting_count", "technical_domain_postings", "has_astronomical_posting",
        "has_medical_posting", "has_hydraulic_posting", "has_military_industrial_posting",
        "has_construction_posting", "has_any_year", "has_two_or_more_sequenced_postings",
    ]:
        merged[col] = merged[col].fillna(0).astype(int)
        
    for col in [
        "career_sequence", "collapsed_career_sequence", "first_state",
        "last_state", "dominant_domain", "dominant_category",
    ]:
        merged[col] = merged[col].fillna("")
    return merged


def edit_distance(seq_a: tuple[str, ...], seq_b: tuple[str, ...], indel: float = 1.0) -> float:
    """Compute Optimal Matching Levenshtein edit distance between two sequences."""
    m, n = len(seq_a), len(seq_b)
    if m == 0:
        return n * indel
    if n == 0:
        return m * indel
        
    prev = np.arange(n + 1, dtype=float) * indel
    cur = np.zeros(n + 1, dtype=float)
    for i in range(1, m + 1):
        cur[0] = i * indel
        a = seq_a[i - 1]
        for j in range(1, n + 1):
            substitution = 0.0 if a == seq_b[j - 1] else 1.0
            cur[j] = min(
                prev[j] + indel,
                cur[j - 1] + indel,
                prev[j - 1] + substitution,
            )
        prev, cur = cur, prev
    return float(prev[n])


def unique_sequences(node_df: pd.DataFrame) -> tuple[list[tuple[str, ...]], np.ndarray, dict[tuple[str, ...], int]]:
    """Extract unique career sequences and frequencies from node table."""
    counter: Counter[tuple[str, ...]] = Counter()
    for sequence in node_df["collapsed_career_sequence"]:
        parts = tuple(part.strip() for part in str(sequence).split(">") if part.strip())
        if not parts:
            parts = ("no_observed_posting",)
        counter[parts] += 1
    sequences = list(counter.keys())
    weights = np.asarray([counter[seq] for seq in sequences], dtype=float)
    return sequences, weights, dict(counter)


def sequence_distance_matrix(sequences: list[tuple[str, ...]]) -> np.ndarray:
    """Compute symmetric pairwise edit distance matrix for unique sequences."""
    n = len(sequences)
    dist = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            value = edit_distance(sequences[i], sequences[j])
            dist[i, j] = value
            dist[j, i] = value
    return dist


def initialize_medoids(dist: np.ndarray, weights: np.ndarray, k: int) -> list[int]:
    """Select initial k-medoids using weighted greedy distance expansion."""
    first = int(np.argmax(weights))
    medoids = [first]
    while len(medoids) < k:
        nearest = np.min(dist[:, medoids], axis=1)
        score = nearest * np.sqrt(weights)
        score[medoids] = -1
        medoids.append(int(np.argmax(score)))
    return medoids


def weighted_k_medoids(
    dist: np.ndarray,
    weights: np.ndarray,
    k: int,
    max_iter: int = 100,
) -> tuple[np.ndarray, list[int]]:
    """Cluster sequence distance matrix using weighted Partitioning Around Medoids (PAM)."""
    k = min(k, dist.shape[0])
    medoids = initialize_medoids(dist, weights, k)
    assignments = np.zeros(dist.shape[0], dtype=int)
    
    for _ in range(max_iter):
        medoid_dist = dist[:, medoids]
        assignments = np.argmin(medoid_dist, axis=1)
        new_medoids: list[int] = []
        for cluster_index in range(k):
            members = np.where(assignments == cluster_index)[0]
            if len(members) == 0:
                new_medoids.append(medoids[cluster_index])
                continue
            sub = dist[np.ix_(members, members)]
            costs = sub @ weights[members]
            new_medoids.append(int(members[np.argmin(costs)]))
        if new_medoids == medoids:
            break
        medoids = new_medoids
        
    assignments = np.argmin(dist[:, medoids], axis=1)
    return assignments, medoids


def assign_career_types(node_df: pd.DataFrame, k: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Group unique career sequences into k empirical types via weighted k-medoids."""
    sequences, weights, sequence_counts = unique_sequences(node_df)
    dist = sequence_distance_matrix(sequences)
    unique_assignments, medoids = weighted_k_medoids(dist, weights, k)
    
    medoid_order = sorted(
        range(len(medoids)),
        key=lambda idx: (-weights[unique_assignments == idx].sum(), idx),
    )
    remap = {old: new + 1 for new, old in enumerate(medoid_order)}
    
    seq_to_type = {
        seq: f"career_type_{remap[int(unique_assignments[i])]}"
        for i, seq in enumerate(sequences)
    }
    seq_to_dist = {
        seq: float(dist[i, medoids[int(unique_assignments[i])]])
        for i, seq in enumerate(sequences)
    }
    
    updated = node_df.copy()
    sequence_tuples = []
    for sequence in updated["collapsed_career_sequence"]:
        parts = tuple(part.strip() for part in str(sequence).split(">") if part.strip())
        if not parts:
            parts = ("no_observed_posting",)
        sequence_tuples.append(parts)
        
    updated["career_type"] = [seq_to_type[seq] for seq in sequence_tuples]
    updated["distance_to_career_type_medoid"] = [seq_to_dist[seq] for seq in sequence_tuples]

    medoid_rows = []
    for old_index in medoid_order:
        medoid_unique_index = medoids[old_index]
        type_name = f"career_type_{remap[old_index]}"
        member_indices = np.where(unique_assignments == old_index)[0]
        member_weight = int(weights[member_indices].sum())
        medoid_sequence = sequences[medoid_unique_index]
        weighted_mean_distance = float(
            np.sum(dist[member_indices, medoid_unique_index] * weights[member_indices])
            / max(member_weight, 1)
        )
        medoid_rows.append(
            {
                "career_type": type_name,
                "node_count": member_weight,
                "unique_sequence_count": int(len(member_indices)),
                "medoid_sequence": " > ".join(medoid_sequence),
                "medoid_sequence_node_count": int(sequence_counts[medoid_sequence]),
                "weighted_mean_distance_to_medoid": weighted_mean_distance,
            }
        )
    return updated, pd.DataFrame(medoid_rows)


def build_transition_matrix(nodes_with_types: pd.DataFrame) -> pd.DataFrame:
    """Compute career state transition frequencies per temporal cluster."""
    rows = []
    for cluster in CLUSTER_ORDER:
        subset = nodes_with_types[nodes_with_types["cluster"].eq(cluster)]
        counts: Counter[tuple[str, str]] = Counter()
        for sequence in subset["collapsed_career_sequence"]:
            states = [part.strip() for part in str(sequence).split(">") if part.strip()]
            for source, target in zip(states, states[1:]):
                counts[(source, target)] += 1
        total = sum(counts.values())
        for (source, target), count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
            rows.append(
                {
                    "cluster": cluster,
                    "cluster_label": CLUSTER_LABELS[cluster],
                    "from_state": source,
                    "to_state": target,
                    "transition_count": int(count),
                    "share_of_cluster_transitions_pct": safe_pct(count, total),
                }
            )
    return pd.DataFrame(rows)


def mode_or_blank(values: pd.Series) -> str:
    """Return the most frequent non-blank string in a Series."""
    values = values.fillna("").astype(str)
    values = values[values.ne("")]
    return "" if values.empty else values.value_counts().index[0]


def cluster_summary(nodes_with_types: pd.DataFrame, postings: pd.DataFrame) -> pd.DataFrame:
    """Summarize career sequence statistics per temporal cluster."""
    rows = []
    for cluster in CLUSTER_ORDER:
        node_subset = nodes_with_types[nodes_with_types["cluster"].eq(cluster)]
        posting_subset = postings[postings["cluster"].eq(cluster)]
        spans = pd.to_numeric(node_subset["career_span_years"], errors="coerce").dropna()
        rows.append(
            {
                "cluster": cluster,
                "cluster_label": CLUSTER_LABELS[cluster],
                "nodes": len(node_subset),
                "posting_rows": len(posting_subset),
                "mean_postings_per_node": float(node_subset["posting_count"].mean()),
                "median_postings_per_node": float(node_subset["posting_count"].median()),
                "mean_collapsed_sequence_length": float(node_subset["collapsed_sequence_length"].mean()),
                "pct_nodes_with_two_or_more_sequenced_postings": safe_pct(
                    int(node_subset["has_two_or_more_sequenced_postings"].sum()),
                    len(node_subset),
                ),
                "pct_nodes_with_any_year": safe_pct(int(node_subset["has_any_year"].sum()), len(node_subset)),
                "pct_high_official": safe_pct(int(node_subset["is_high_official"].sum()), len(node_subset)),
                "mean_career_span_years": float(spans.mean()) if not spans.empty else np.nan,
                "median_career_span_years": float(spans.median()) if not spans.empty else np.nan,
                "top_first_state": mode_or_blank(node_subset["first_state"]),
                "top_last_state": mode_or_blank(node_subset["last_state"]),
                "top_dominant_domain": mode_or_blank(node_subset["dominant_domain"]),
                "top_career_type": mode_or_blank(node_subset["career_type"]),
            }
        )
    return pd.DataFrame(rows)


def career_type_summary(nodes_with_types: pd.DataFrame) -> pd.DataFrame:
    """Summarize career sequence attributes by estimated career type."""
    rows = []
    for career_type, group in nodes_with_types.groupby("career_type"):
        row = {
            "career_type": career_type,
            "nodes": len(group),
            "share_all_nodes_pct": safe_pct(len(group), len(nodes_with_types)),
            "medoid_like_sequence": mode_or_blank(group["collapsed_career_sequence"]),
            "mean_collapsed_sequence_length": float(group["collapsed_sequence_length"].mean()),
            "pct_high_official": safe_pct(int(group["is_high_official"].sum()), len(group)),
            "top_cluster": mode_or_blank(group["cluster"]),
            "top_dominant_domain": mode_or_blank(group["dominant_domain"]),
        }
        for cluster in CLUSTER_ORDER:
            row[f"{cluster}_nodes"] = int(group["cluster"].eq(cluster).sum())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("career_type")


def logistic_fit(
    x: np.ndarray,
    y: np.ndarray,
    max_iter: int,
    l2: float,
    seed: int,
) -> tuple[np.ndarray, dict[str, object]]:
    """Fit binary logistic regression model with L2 regularization using Adam."""
    rng = np.random.default_rng(seed)
    n, p = x.shape
    beta = rng.normal(0.0, 0.001, size=p)
    m, v = np.zeros_like(beta), np.zeros_like(beta)
    lr, beta1, beta2, eps = 0.03, 0.9, 0.999, 1e-8
    
    penalty_mask = np.ones_like(beta)
    penalty_mask[0] = 0.0  # Do not penalize intercept
    
    previous = float("inf")
    converged = False

    for iteration in range(1, max_iter + 1):
        logits = x @ beta
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -35, 35)))
        loss = -float(np.mean(y * np.log(np.maximum(probs, 1e-15)) + (1 - y) * np.log(np.maximum(1 - probs, 1e-15))))
        loss += 0.5 * l2 * float(np.sum((beta * penalty_mask) ** 2))
        
        gradient = x.T @ (probs - y) / n + l2 * beta * penalty_mask
        m = beta1 * m + (1.0 - beta1) * gradient
        v = beta2 * v + (1.0 - beta2) * (gradient**2)
        m_hat = m / (1.0 - beta1**iteration)
        v_hat = v / (1.0 - beta2**iteration)
        beta -= lr * m_hat / (np.sqrt(v_hat) + eps)
        
        if iteration % 100 == 0:
            if abs(previous - loss) < 1e-8:
                converged = True
                break
            previous = loss

    logits = x @ beta
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -35, 35)))
    log_likelihood = float(
        np.sum(y * np.log(np.maximum(probs, 1e-15)) + (1 - y) * np.log(np.maximum(1 - probs, 1e-15)))
    )
    predictions = (probs >= 0.5).astype(int)
    n_params = len(beta)
    
    diagnostics = {
        "high_status_logit_converged": converged,
        "high_status_logit_iterations": iteration,
        "high_status_logit_rows": n,
        "high_status_logit_features": p,
        "high_status_logit_positive_rate": float(y.mean()),
        "high_status_logit_accuracy": float(np.mean(predictions == y)),
        "high_status_logit_log_likelihood": log_likelihood,
        "high_status_logit_aic": 2 * n_params - 2 * log_likelihood,
        "high_status_logit_bic": math.log(n) * n_params - 2 * log_likelihood,
        "high_status_logit_l2_penalty": l2,
    }
    return beta, diagnostics


def build_high_status_model(
    nodes_with_types: pd.DataFrame,
    max_iter: int,
    l2: float,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Format controls and fit binary logit model for high-status official access."""
    df = nodes_with_types.copy()
    df["intercept"] = 1.0
    df["cluster2_yuan_ming"] = df["cluster"].eq("cluster2_yuan_ming").astype(int)
    df["cluster3_qing_after"] = df["cluster"].eq("cluster3_qing_after").astype(int)
    
    df["log_posting_count_z"] = safe_z(np.log1p(df["posting_count"]))
    df["collapsed_sequence_length_z"] = safe_z(df["collapsed_sequence_length"])
    df["career_span_years_z"] = safe_z(pd.to_numeric(df["career_span_years"], errors="coerce").fillna(df["career_span_years"].median()))
    df["technical_domain_share_z"] = safe_z(df["technical_domain_share"])
    
    candidate_features = [
        "intercept", "cluster2_yuan_ming", "cluster3_qing_after",
        "log_posting_count_z", "collapsed_sequence_length_z", "career_span_years_z",
        "technical_domain_share_z", *ENTRY_CONTROLS, *NETWORK_CONTROLS,
        *[col for col in BUREAU_DUMMIES if col in df.columns],
    ]
    
    features = []
    for feature in candidate_features:
        values = pd.to_numeric(df[feature], errors="coerce").fillna(0.0)
        if feature != "intercept":
            std = values.std(ddof=0)
            if std == 0 or np.isnan(std):
                continue
        df[feature] = values
        features.append(feature)
        
    x = df[features].to_numpy(dtype=float)
    y = pd.to_numeric(df["is_high_official"], errors="coerce").fillna(0).to_numpy(dtype=float)
    beta, diagnostics = logistic_fit(x, y, max_iter=max_iter, l2=l2, seed=seed)
    
    rows = []
    for feature, coefficient in zip(features, beta):
        rows.append(
            {
                "outcome": "is_high_official",
                "feature": feature,
                "coefficient": float(coefficient),
                "odds_ratio": math.exp(float(coefficient)) if -700 < coefficient < 700 else float("inf"),
                "model": "binary_logit_numpy_l2",
            }
        )
    diagnostics["high_status_logit_model_features"] = ";".join(features)
    return pd.DataFrame(rows), diagnostics


def build_diagnostics(
    nodes: pd.DataFrame,
    postings: pd.DataFrame,
    nodes_with_types: pd.DataFrame,
    medoids: pd.DataFrame,
    model_diagnostics: dict[str, object],
    career_types: int,
    seed: int,
) -> pd.DataFrame:
    """Format execution diagnostic metrics into a summary DataFrame."""
    rows: list[dict[str, object]] = []

    def add(metric: str, value: object) -> None:
        rows.append({"metric": metric, "value": value})

    add("total_science_official_nodes", len(nodes))
    add("nodes_with_any_posting", postings["c_personid"].nunique())
    add("posting_rows", len(postings))
    add("posting_rows_with_sequence", int(postings["has_sequence"].sum()))
    add("posting_rows_with_sequence_pct", safe_pct(int(postings["has_sequence"].sum()), len(postings)))
    add("posting_rows_with_positive_firstyear", int(postings["has_firstyear"].sum()))
    add("posting_rows_with_positive_firstyear_pct", safe_pct(int(postings["has_firstyear"].sum()), len(postings)))
    add("posting_rows_with_category", int(postings["c_office_category_id"].notna().sum()))
    add("posting_rows_with_category_pct", safe_pct(int(postings["c_office_category_id"].notna().sum()), len(postings)))
    add("nodes_with_two_or_more_sequenced_postings", int(nodes_with_types["has_two_or_more_sequenced_postings"].sum()))
    add("nodes_with_any_year", int(nodes_with_types["has_any_year"].sum()))
    add("unique_collapsed_career_sequences", nodes_with_types["collapsed_career_sequence"].nunique())
    add("career_types_requested", career_types)
    add("career_types_estimated", len(medoids))
    add("seed", seed)
    
    for cluster in CLUSTER_ORDER:
        subset = nodes_with_types[nodes_with_types["cluster"].eq(cluster)]
        add(f"{cluster}_nodes", len(subset))
        add(f"{cluster}_posting_rows", int(postings["cluster"].eq(cluster).sum()))
        add(f"{cluster}_high_official_nodes", int(subset["is_high_official"].sum()))
        
    for key, value in model_diagnostics.items():
        add(key, value)
    return pd.DataFrame(rows)


def main() -> None:
    """Run career trajectory analysis pipeline."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    nodes = load_nodes(args.input)
    postings_raw = fetch_postings(args.db, [int(x) for x in nodes["id"]])
    postings = build_posting_table(nodes, postings_raw)
    node_features = build_node_features(nodes, postings)
    
    nodes_with_types, medoids = assign_career_types(node_features, args.career_types)
    postings = postings.merge(
        nodes_with_types[["id", "career_type"]],
        left_on="c_personid",
        right_on="id",
        how="left",
        suffixes=("", "_career"),
    )
    
    transition_matrix = build_transition_matrix(nodes_with_types)
    cluster_stats = cluster_summary(nodes_with_types, postings)
    type_stats = career_type_summary(nodes_with_types)
    
    high_status_model, model_diagnostics = build_high_status_model(
        nodes_with_types,
        max_iter=args.model_iterations,
        l2=args.model_l2,
        seed=args.seed,
    )
    diagnostics = build_diagnostics(
        nodes,
        postings,
        nodes_with_types,
        medoids,
        model_diagnostics,
        career_types=args.career_types,
        seed=args.seed,
    )

    postings_path = args.output_dir / "career_trajectory_postings.csv"
    nodes_path = args.output_dir / "career_trajectory_nodes.csv"
    cluster_summary_path = args.output_dir / "career_trajectory_cluster_summary.csv"
    type_summary_path = args.output_dir / "career_trajectory_type_summary.csv"
    medoids_path = args.output_dir / "career_trajectory_type_medoids.csv"
    transitions_path = args.output_dir / "career_trajectory_transition_matrix.csv"
    model_path = args.output_dir / "career_trajectory_high_status_logit.csv"
    diagnostics_path = args.output_dir / "career_trajectory_model_diagnostics.csv"

    postings.to_csv(postings_path, index=False)
    nodes_with_types.to_csv(nodes_path, index=False)
    cluster_stats.to_csv(cluster_summary_path, index=False)
    type_stats.to_csv(type_summary_path, index=False)
    medoids.to_csv(medoids_path, index=False)
    transition_matrix.to_csv(transitions_path, index=False)
    high_status_model.to_csv(model_path, index=False)
    diagnostics.to_csv(diagnostics_path, index=False, quoting=csv.QUOTE_MINIMAL)

    output_paths = [
        postings_path, nodes_path, cluster_summary_path, type_summary_path,
        medoids_path, transitions_path, model_path, diagnostics_path,
    ]

    print("Career trajectory analysis complete.")
    for path in output_paths:
        print(path)


if __name__ == "__main__":
    main()