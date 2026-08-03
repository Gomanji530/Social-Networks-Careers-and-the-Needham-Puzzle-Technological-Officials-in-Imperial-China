#!/usr/bin/env python3
"""Intergenerational transmission analysis for CBDB father-son pairs.

This module tests whether scientific-official status is transmitted across
father-son ties, and benchmarks that transmission rate against civil-exam (jinshi)
status using binary logistic regression.
"""

from __future__ import annotations

import argparse
import csv
import math
import sqlite3
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# Path configuration
TASK_DIR = Path(__file__).resolve().parent
ROOT = TASK_DIR.parents[1]
DB_PATH = ROOT / "latest" / "cbdb_20260516.sqlite3"
SCIENCE_NODES = ROOT / "analysis" / "outputs" / "alaam_science_officials_break3_induced_nodes.csv"
NODE_FEATURES = ROOT / "analysis" / "outputs" / "node_features.csv"

# Historical temporal cluster definitions
CLUSTERS = [
    ("cluster1_song_and_before", "Song and before", -10000, 1278, "#2b6cb0"),
    ("cluster2_yuan_ming", "Yuan and Ming", 1279, 1643, "#b7791f"),
    ("cluster3_qing_after", "Qing and after", 1644, 3000, "#2f855a"),
]
CLUSTER_ORDER = [row[0] for row in CLUSTERS]
CLUSTER_LABELS = {row[0]: row[1] for row in CLUSTERS}

# Civil examination codes (jinshi, etc.)
CIVIL_EXAM_CODES = {26, 28, 29, 36, 37, 39, 42, 47, 48, 49, 50, 51, 52, 53, 54, 56, 57}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--science-nodes", type=Path, default=SCIENCE_NODES)
    parser.add_argument("--node-features", type=Path, default=NODE_FEATURES)
    parser.add_argument("--output-dir", type=Path, default=TASK_DIR)
    parser.add_argument("--seed", type=int, default=20260608)
    parser.add_argument("--model-iterations", type=int, default=5000)
    parser.add_argument("--model-l2", type=float, default=1e-4)
    return parser.parse_args()


def safe_pct(part: int | float, total: int | float) -> float:
    """Safely calculate percentage avoiding division by zero errors."""
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


def cluster_for_year(year: object) -> str:
    """Determine the temporal cluster key for a given index year."""
    if pd.isna(year):
        return "missing_year"
    year_i = int(year)
    for key, _label, start, end, _color in CLUSTERS:
        if start <= year_i <= end:
            return key
    return "out_of_scope"


def fetch_parent_child_pairs(conn: sqlite3.Connection) -> pd.DataFrame:
    """Query KIN_DATA to extract father-son kinship pairs in both directional records."""
    sql = """
        select
            k.c_personid,
            k.c_kin_id,
            k.c_kin_code,
            kc.c_kinrel_chn,
            kc.c_kinrel,
            kc.c_kinrel_simplified
        from KIN_DATA k
        join KINSHIP_CODES kc on k.c_kin_code = kc.c_kincode
        where kc.c_kinrel_simplified in ('F', 'S')
    """
    raw = pd.read_sql_query(sql, conn)
    
    # Process father records
    father_rows = raw[raw["c_kinrel_simplified"].eq("F")].copy()
    father_rows["father_id"] = father_rows["c_kin_id"].astype(int)
    father_rows["child_id"] = father_rows["c_personid"].astype(int)
    
    # Process son records
    son_rows = raw[raw["c_kinrel_simplified"].eq("S")].copy()
    son_rows["father_id"] = son_rows["c_personid"].astype(int)
    son_rows["child_id"] = son_rows["c_kin_id"].astype(int)
    
    pairs = pd.concat([father_rows, son_rows], ignore_index=True)
    pairs = pairs[pairs["father_id"].ne(pairs["child_id"])].copy()
    
    # Deduplicate by unique father-son ID pairs
    grouped = (
        pairs.groupby(["father_id", "child_id"])
        .agg(
            relation_codes=("c_kin_code", lambda s: ";".join(map(str, sorted(set(map(int, s)))))),
            relation_labels_chn=("c_kinrel_chn", lambda s: ";".join(sorted(set(map(str, s))))),
            relation_labels=("c_kinrel", lambda s: ";".join(sorted(set(map(str, s))))),
            relation_record_count=("c_kin_code", "size"),
        )
        .reset_index()
    )
    return grouped


def fetch_biog(conn: sqlite3.Connection, person_ids: list[int]) -> pd.DataFrame:
    """Query biographical metadata (names, index year, dynasty, sex) for person IDs."""
    frames = []
    for chunk in chunked(person_ids):
        placeholders = ",".join("?" for _ in chunk)
        frames.append(
            pd.read_sql_query(
                f"""
                select c_personid, c_name_chn, c_name, c_surname_chn, c_surname,
                       c_index_year, c_female, c_dy
                from BIOG_MAIN
                where c_personid in ({placeholders})
                """,
                conn,
                params=chunk,
            )
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def fetch_civil_exam(conn: sqlite3.Connection, person_ids: list[int]) -> pd.DataFrame:
    """Query whether individuals passed civil service examinations."""
    rows = []
    for chunk in chunked(person_ids):
        placeholders = ",".join("?" for _ in chunk)
        rows.append(
            pd.read_sql_query(
                f"""
                select c_personid, max(case when c_entry_code in ({','.join(map(str, sorted(CIVIL_EXAM_CODES)))}) then 1 else 0 end) as has_civil_exam
                from ENTRY_DATA
                where c_personid in ({placeholders})
                group by c_personid
                """,
                conn,
                params=chunk,
            )
        )
    if not rows:
        return pd.DataFrame(columns=["c_personid", "has_civil_exam"])
    return pd.concat(rows, ignore_index=True)


def build_person_table(args: argparse.Namespace, person_ids: list[int]) -> pd.DataFrame:
    """Construct comprehensive individual attribute table including civil and technical office flags."""
    conn = sqlite3.connect(args.db)
    biog = fetch_biog(conn, person_ids)
    civil = fetch_civil_exam(conn, person_ids)
    conn.close()
    
    people = biog.merge(civil, on="c_personid", how="left")
    people["has_civil_exam"] = people["has_civil_exam"].fillna(0).astype(int)

    science = pd.read_csv(args.science_nodes)
    science_cols = [
        "id", "cluster", "bureau_type", "bureau_zh", "is_high_official",
        "has_technical_exam", "degree_z", "clustering_z", "eigenvector_z", "betweenness_z",
    ]
    science = science[[col for col in science_cols if col in science.columns]].copy()
    science["is_science_official"] = 1

    if args.node_features.exists():
        features = pd.read_csv(args.node_features, usecols=["id", "is_high_official"])
        features = features.rename(columns={"is_high_official": "network_high_official"})
    else:
        features = pd.DataFrame(columns=["id", "network_high_official"])

    people = people.merge(science, left_on="c_personid", right_on="id", how="left")
    people = people.merge(features, left_on="c_personid", right_on="id", how="left", suffixes=("", "_feature"))
    
    people["is_science_official"] = people["is_science_official"].fillna(0).astype(int)
    people["cluster"] = people["cluster"].fillna("")
    people["child_cluster_by_year"] = people["c_index_year"].apply(cluster_for_year)
    people["is_high_official"] = people["is_high_official"].fillna(0).astype(int)
    people["network_high_official"] = people["network_high_official"].fillna(0).astype(int)
    people["is_any_high_official"] = people[["is_high_official", "network_high_official"]].max(axis=1).astype(int)
    
    people["c_female"] = pd.to_numeric(people["c_female"], errors="coerce")
    people["female_known"] = people["c_female"].notna().astype(int)
    people["female"] = people["c_female"].fillna(0).astype(int)
    people["c_index_year"] = pd.to_numeric(people["c_index_year"], errors="coerce")
    return people


def build_pair_table(pairs: pd.DataFrame, people: pd.DataFrame) -> pd.DataFrame:
    """Merge father and child attributes into a dyadic pair analysis table."""
    father = people.add_prefix("father_")
    child = people.add_prefix("child_")
    
    df = pairs.merge(father, left_on="father_id", right_on="father_c_personid", how="left")
    df = df.merge(child, left_on="child_id", right_on="child_c_personid", how="left")
    
    # Surname matching and standardized temporal index variables
    df["same_surname"] = (
        df["father_c_surname_chn"].fillna("").ne("")
        & df["father_c_surname_chn"].fillna("").eq(df["child_c_surname_chn"].fillna(""))
    ).astype(int)
    df["child_index_year_z"] = safe_z(df["child_c_index_year"])
    df["father_index_year_z"] = safe_z(df["father_c_index_year"])
    
    for prefix in ["father", "child"]:
        for col in ["is_science_official", "has_civil_exam", "is_any_high_official", "female_known", "female"]:
            full = f"{prefix}_{col}"
            df[full] = pd.to_numeric(df[full], errors="coerce").fillna(0).astype(int)
            
    df["child_cluster_by_year_label"] = (
        df["child_child_cluster_by_year"].map(CLUSTER_LABELS).fillna(df["child_child_cluster_by_year"])
    )
    return df


def logistic_fit(
    x: np.ndarray,
    y: np.ndarray,
    max_iter: int,
    l2: float,
    seed: int,
) -> tuple[np.ndarray, dict[str, object]]:
    """Fit binary logistic regression model with L2 regularization using Newton-Raphson optimizer."""
    n, p = x.shape
    beta = np.zeros(p, dtype=float)
    converged = False
    penalty_mask = np.ones_like(beta)
    penalty_mask[0] = 0.0  # Do not penalize intercept

    def objective(coefs: np.ndarray) -> float:
        logits = x @ coefs
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -35, 35)))
        nll = -float(
            np.sum(
                y * np.log(np.maximum(probs, 1e-15))
                + (1.0 - y) * np.log(np.maximum(1.0 - probs, 1e-15))
            )
        )
        return nll + 0.5 * l2 * float(np.sum((coefs * penalty_mask) ** 2))

    current = objective(beta)
    for iteration in range(1, max_iter + 1):
        logits = x @ beta
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -35, 35)))
        weights = np.maximum(probs * (1.0 - probs), 1e-9)
        gradient = x.T @ (probs - y) + l2 * beta * penalty_mask
        hessian = x.T @ (weights[:, None] * x) + np.diag(l2 * penalty_mask)
        
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(hessian) @ gradient

        # Line search
        step_scale = 1.0
        accepted = False
        while step_scale >= 1e-6:
            candidate = beta - step_scale * step
            candidate_obj = objective(candidate)
            if candidate_obj <= current:
                beta = candidate
                if abs(current - candidate_obj) < 1e-6 or np.max(np.abs(step_scale * step)) < 1e-7:
                    current = candidate_obj
                    converged = True
                else:
                    current = candidate_obj
                accepted = True
                break
            step_scale *= 0.5
            
        if converged or not accepted:
            if np.max(np.abs(gradient)) < 1e-5:
                converged = True
            break

    probs = 1.0 / (1.0 + np.exp(-np.clip(x @ beta, -35, 35)))
    log_likelihood = float(
        np.sum(y * np.log(np.maximum(probs, 1e-15)) + (1 - y) * np.log(np.maximum(1 - probs, 1e-15)))
    )
    predictions = (probs >= 0.5).astype(int)
    
    diagnostics = {
        "converged": converged,
        "iterations": iteration,
        "rows": n,
        "features": p,
        "positive_rate": float(y.mean()),
        "accuracy": float(np.mean(predictions == y)),
        "log_likelihood": log_likelihood,
        "aic": 2 * p - 2 * log_likelihood,
        "bic": math.log(n) * p - 2 * log_likelihood,
        "l2_penalty": l2,
    }
    return beta, diagnostics


def fit_model(
    df: pd.DataFrame,
    outcome: str,
    father_predictor: str,
    model_name: str,
    max_iter: int,
    l2: float,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Fit intergenerational transmission model (science official vs civil exam benchmark)."""
    sample = df[df["child_c_index_year"].notna()].copy()
    sample["intercept"] = 1.0
    features = [
        "intercept",
        father_predictor,
        "child_index_year_z",
        "father_index_year_z",
        "same_surname",
        "father_is_any_high_official",
        "child_female_known",
        "child_female",
    ]
    
    x = sample[features].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    y = pd.to_numeric(sample[outcome], errors="coerce").fillna(0).to_numpy(dtype=float)
    beta, diag = logistic_fit(x, y, max_iter=max_iter, l2=l2, seed=seed)
    
    rows = []
    for feature, coef in zip(features, beta):
        rows.append(
            {
                "model": model_name,
                "outcome": outcome,
                "feature": feature,
                "coefficient": float(coef),
                "odds_ratio": math.exp(float(coef)) if -700 < coef < 700 else float("inf"),
            }
        )
    return pd.DataFrame(rows), {f"{model_name}_{key}": value for key, value in diag.items()}


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute contingency summaries of child status rates conditioned on father attributes."""
    rows = []
    for model, father_col, child_col in [
        ("science_official_transmission", "father_is_science_official", "child_is_science_official"),
        ("civil_exam_transmission", "father_has_civil_exam", "child_has_civil_exam"),
    ]:
        for father_value in [0, 1]:
            subset = df[df[father_col].eq(father_value)]
            rows.append(
                {
                    "model": model,
                    "father_indicator": father_col,
                    "father_indicator_value": father_value,
                    "pairs": len(subset),
                    "child_positive_count": int(subset[child_col].sum()),
                    "child_positive_rate_pct": safe_pct(int(subset[child_col].sum()), len(subset)),
                }
            )
    return pd.DataFrame(rows)


def by_cluster_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize transmission statistics categorized by child's temporal cluster."""
    rows = []
    for cluster in CLUSTER_ORDER:
        subset = df[df["child_child_cluster_by_year"].eq(cluster)]
        rows.append(
            {
                "child_cluster": cluster,
                "child_cluster_label": CLUSTER_LABELS[cluster],
                "father_son_pairs": len(subset),
                "science_children": int(subset["child_is_science_official"].sum()),
                "science_child_rate_pct": safe_pct(int(subset["child_is_science_official"].sum()), len(subset)),
                "pairs_with_science_father": int(subset["father_is_science_official"].sum()),
                "science_father_rate_pct": safe_pct(int(subset["father_is_science_official"].sum()), len(subset)),
                "civil_exam_children": int(subset["child_has_civil_exam"].sum()),
                "civil_exam_child_rate_pct": safe_pct(int(subset["child_has_civil_exam"].sum()), len(subset)),
                "pairs_with_civil_exam_father": int(subset["father_has_civil_exam"].sum()),
                "civil_exam_father_rate_pct": safe_pct(int(subset["father_has_civil_exam"].sum()), len(subset)),
            }
        )
    return pd.DataFrame(rows)


def build_diagnostics(df: pd.DataFrame, model_diags: dict[str, object], seed: int) -> pd.DataFrame:
    """Format run metadata and model evaluation diagnostic metrics."""
    rows = []

    def add(metric: str, value: object) -> None:
        rows.append({"metric": metric, "value": value})

    add("father_son_pairs", len(df))
    add("pairs_with_child_index_year", int(df["child_c_index_year"].notna().sum()))
    add("pairs_with_father_index_year", int(df["father_c_index_year"].notna().sum()))
    add("science_fathers", int(df["father_is_science_official"].sum()))
    add("science_children", int(df["child_is_science_official"].sum()))
    add("civil_exam_fathers", int(df["father_has_civil_exam"].sum()))
    add("civil_exam_children", int(df["child_has_civil_exam"].sum()))
    add("same_surname_pairs", int(df["same_surname"].sum()))
    add("seed", seed)
    
    for key, value in model_diags.items():
        add(key, value)
    return pd.DataFrame(rows)


def main() -> None:
    """Run intergenerational transmission analysis pipeline."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(args.db)
    pairs = fetch_parent_child_pairs(conn)
    person_ids = sorted(set(pairs["father_id"]).union(set(pairs["child_id"])))
    conn.close()
    
    people = build_person_table(args, person_ids)
    pair_table = build_pair_table(pairs, people)
    summary = build_summary(pair_table)
    cluster = by_cluster_summary(pair_table)
    
    # Fit model for science official status transmission
    science_model, science_diag = fit_model(
        pair_table,
        outcome="child_is_science_official",
        father_predictor="father_is_science_official",
        model_name="science_official_transmission",
        max_iter=args.model_iterations,
        l2=args.model_l2,
        seed=args.seed,
    )
    
    # Fit benchmark model for civil examination status transmission
    civil_model, civil_diag = fit_model(
        pair_table,
        outcome="child_has_civil_exam",
        father_predictor="father_has_civil_exam",
        model_name="civil_exam_transmission",
        max_iter=args.model_iterations,
        l2=args.model_l2,
        seed=args.seed,
    )
    
    models = pd.concat([science_model, civil_model], ignore_index=True)
    diagnostics = build_diagnostics(pair_table, {**science_diag, **civil_diag}, args.seed)

    pairs_path = args.output_dir / "intergenerational_pairs.csv"
    summary_path = args.output_dir / "intergenerational_summary.csv"
    cluster_path = args.output_dir / "intergenerational_by_cluster.csv"
    model_path = args.output_dir / "intergenerational_logit.csv"
    diagnostics_path = args.output_dir / "intergenerational_model_diagnostics.csv"

    pair_table.to_csv(pairs_path, index=False)
    summary.to_csv(summary_path, index=False)
    cluster.to_csv(cluster_path, index=False)
    models.to_csv(model_path, index=False)
    diagnostics.to_csv(diagnostics_path, index=False, quoting=csv.QUOTE_MINIMAL)

    output_paths = [pairs_path, summary_path, cluster_path, model_path, diagnostics_path]

    print("Task4 intergenerational transmission complete.")
    for path in output_paths:
        print(path)


if __name__ == "__main__":
    main()