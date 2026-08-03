#!/usr/bin/env python3
"""Model-based structural breaks for scientific-official status.

The dependent variable is whether a network member is a scientific/technical
official (`is_main`). Predictors include network centrality, status/class
proxies, entry-route proxies, and locality coordinates. Breaks are selected by
minimizing segment-wise logistic deviance with a BIC penalty.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

# Define project root and input/output path structures
ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "latest" / "cbdb_20260516.sqlite3"
NODE_FEATURES = ROOT / "analysis" / "outputs" / "node_features.csv"
OUT_DIR = ROOT / "analysis" / "outputs"

# Define output file destinations
PANEL_CSV = OUT_DIR / "science_official_model_panel.csv"
BREAKS_CSV = OUT_DIR / "science_official_model_breaks.csv"
SEGMENTS_CSV = OUT_DIR / "science_official_model_segments.csv"
SUMMARY_JSON = OUT_DIR / "science_official_model_summary.json"
REPORT_MD = OUT_DIR / "science_official_model_report.md"

# List of major historical Chinese dynastic transitions for alignment verification
MAJOR_DYNASTIC_TRANSITIONS = [
    {"year": 220, "label": "Eastern Han collapse / Three Kingdoms"},
    {"year": 589, "label": "Sui reunification"},
    {"year": 618, "label": "Tang founding"},
    {"year": 907, "label": "Tang collapse / Five Dynasties"},
    {"year": 960, "label": "Song founding"},
    {"year": 1127, "label": "Northern Song collapse"},
    {"year": 1279, "label": "Song-Yuan transition"},
    {"year": 1368, "label": "Ming founding / Yuan collapse"},
    {"year": 1644, "label": "Ming-Qing transition"},
    {"year": 1912, "label": "Qing-Republic transition"},
]

# Mapping CBDB entry codes to sociological categories
CIVIL_EXAM_CODES = {26, 28, 29, 36, 37, 39, 42, 47, 48, 49, 50, 51, 52, 53, 54, 56, 57}
MEDICAL_EXAM_CODES = {43}
MILITARY_EXAM_CODES = {44, 46, 328}
RECOMMENDATION_CODES = {101, 225, 310, 319}
YIN_PRIVILEGE_CODES = {8, 59, 60, 62, 118, 138, 157, 163, 197, 198, 317, 318}
PURCHASE_CODES = {7}
DIRECT_APPOINTMENT_CODES = {13, 14, 15}

# Predictor variable columns used in the logistic regression model
FEATURE_COLUMNS = [
    "log_degree_z",
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
    "female_known",
    "female",
]


def pct(part: int | float, total: int | float) -> float:
    return round(part / total * 100, 2) if total else 0.0


def bin_start_for_year(year: int, bin_size: int) -> int:
    return math.floor(year / bin_size) * bin_size


def chunked(values: list[int], size: int = 800):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def nearest_transition(year: int | float) -> dict[str, Any]:
    nearest = min(MAJOR_DYNASTIC_TRANSITIONS, key=lambda item: abs(item["year"] - year))
    return {
        "nearest_transition_year": nearest["year"],
        "nearest_transition_label": nearest["label"],
        "distance_years": round(year - nearest["year"], 2),
        "absolute_distance_years": round(abs(year - nearest["year"]), 2),
    }


def entry_flags(conn: sqlite3.Connection, person_ids: list[int]) -> dict[int, dict[str, int]]:
    flags: dict[int, dict[str, int]] = defaultdict(lambda: {
        "has_civil_exam": 0,
        "has_yin_privilege": 0,
        "has_recommendation": 0,
        "has_technical_exam": 0,
        "has_purchase_or_direct": 0,
    })
    all_codes = (
        CIVIL_EXAM_CODES
        | YIN_PRIVILEGE_CODES
        | RECOMMENDATION_CODES
        | MEDICAL_EXAM_CODES
        | MILITARY_EXAM_CODES
        | PURCHASE_CODES
        | DIRECT_APPOINTMENT_CODES
    )
    # Query database in batches to respect SQLite variable bounds
    for chunk in chunked(person_ids):
        placeholders = ",".join("?" for _ in chunk)
        sql = f"select c_personid, c_entry_code from ENTRY_DATA where c_personid in ({placeholders})"
        for row in conn.execute(sql, chunk):
            pid = int(row["c_personid"])
            code = int(row["c_entry_code"])
            if code in CIVIL_EXAM_CODES:
                flags[pid]["has_civil_exam"] = 1
            if code in YIN_PRIVILEGE_CODES:
                flags[pid]["has_yin_privilege"] = 1
            if code in RECOMMENDATION_CODES:
                flags[pid]["has_recommendation"] = 1
            if code in MEDICAL_EXAM_CODES or code in MILITARY_EXAM_CODES:
                flags[pid]["has_technical_exam"] = 1
            if code in PURCHASE_CODES or code in DIRECT_APPOINTMENT_CODES:
                flags[pid]["has_purchase_or_direct"] = 1
            if code not in all_codes:
                flags[pid]
    return flags


def biog_and_geo(conn: sqlite3.Connection, person_ids: list[int]) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    for chunk in chunked(person_ids):
        placeholders = ",".join("?" for _ in chunk)
        sql = f"""
            select c_personid, c_index_year, c_female, c_ethnicity_code, c_dy
            from BIOG_MAIN
            where c_personid in ({placeholders})
        """
        for row in conn.execute(sql, chunk):
            rows[int(row["c_personid"])] = dict(row)

    geo: dict[int, dict[str, Any]] = {}
    for chunk in chunked(person_ids):
        placeholders = ",".join("?" for _ in chunk)
        # Prioritize primary address types (addr_type = 1) and lowest sequence
        sql = f"""
            select b.c_personid, a.c_addr_id, a.c_name_chn, a.x_coord, a.y_coord,
                   b.c_sequence, b.c_addr_type
            from BIOG_ADDR_DATA b
            join ADDR_CODES a on b.c_addr_id = a.c_addr_id
            where b.c_personid in ({placeholders})
              and a.x_coord is not null
              and a.y_coord is not null
            order by b.c_personid, case when b.c_addr_type = 1 then 0 else 1 end, b.c_sequence
        """
        # Merge geographic attributes into biographical dictionary
        for row in conn.execute(sql, chunk):
            pid = int(row["c_personid"])
            if pid not in geo:
                geo[pid] = dict(row)

    for pid, row in rows.items():
        g = geo.get(pid, {})
        row["addr_id"] = g.get("c_addr_id")
        row["addr_name_chn"] = g.get("c_name_chn")
        row["x_coord"] = g.get("x_coord")
        row["y_coord"] = g.get("y_coord")
    return rows


def load_panel(start_year: int, end_year: int, bin_size: int) -> pd.DataFrame:
    df = pd.read_csv(NODE_FEATURES)
    df["id"] = df["id"].astype(int)
    person_ids = df["id"].tolist()
    # Query SQLite database for extra biographical features
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    entry = entry_flags(conn, person_ids)
    biog = biog_and_geo(conn, person_ids)
    conn.close()

    # Build enriched rows
    extra_rows = []
    for pid in person_ids:
        b = biog.get(pid, {})
        e = entry.get(pid, {})
        extra_rows.append(
            {
                "id": pid,
                "model_year": b.get("c_index_year"),
                "female": b.get("c_female"),
                "ethnicity_code": b.get("c_ethnicity_code"),
                "biog_dynasty_code": b.get("c_dy"),
                "addr_id": b.get("addr_id"),
                "addr_name_chn": b.get("addr_name_chn"),
                "x_coord": b.get("x_coord"),
                "y_coord": b.get("y_coord"),
                **e,
            }
        )
    extra = pd.DataFrame(extra_rows)
    df = df.merge(extra, on="id", how="left")

    # Format numeric indicators and log-transform degree
    df["is_science_official"] = df["is_main"].astype(int)
    df["is_high_official"] = df["is_high_official"].astype(int)
    df["degree"] = pd.to_numeric(df["degree"], errors="coerce").fillna(0)
    df["log_degree"] = np.log1p(df["degree"])
    df["betweenness_centrality"] = pd.to_numeric(df["betweenness_centrality"], errors="coerce").fillna(0)
    df["clustering_coefficient"] = pd.to_numeric(df["clustering_coefficient"], errors="coerce").fillna(0)
    df["eigenvector_centrality"] = pd.to_numeric(df["eigenvector_centrality"], errors="coerce").fillna(0)
    df["female_known"] = df["female"].notna().astype(int)
    df["female"] = pd.to_numeric(df["female"], errors="coerce").fillna(0).astype(int)

    for col in ["has_civil_exam", "has_yin_privilege", "has_recommendation", "has_technical_exam", "has_purchase_or_direct"]:
        df[col] = df[col].fillna(0).astype(int)
    df["has_geo"] = df["x_coord"].notna().astype(int)
    df["x_coord"] = pd.to_numeric(df["x_coord"], errors="coerce")
    df["y_coord"] = pd.to_numeric(df["y_coord"], errors="coerce")

    # Filter observations by time range window and assign time bins
    df = df[df["model_year"].notna()].copy()
    df["model_year"] = df["model_year"].astype(int)
    df = df[(df["model_year"] >= start_year) & (df["model_year"] <= end_year)].copy()
    df["time_bin"] = df["model_year"].apply(lambda y: bin_start_for_year(int(y), bin_size))

    # Standardize continuous variables across panel (Z-score normalization)
    for raw_col, z_col in [
        ("log_degree", "log_degree_z"),
        ("betweenness_centrality", "betweenness_z"),
        ("clustering_coefficient", "clustering_z"),
        ("eigenvector_centrality", "eigenvector_z"),
        ("x_coord", "x_coord_z"),
        ("y_coord", "y_coord_z"),
    ]:
        values = df[raw_col].copy()
        if raw_col in {"x_coord", "y_coord"}:
            values = values.fillna(values.mean())
        mean = values.mean()
        std = values.std(ddof=0)
        if not std or np.isnan(std):
            df[z_col] = 0.0
        else:
            df[z_col] = (values - mean) / std
        df[z_col] = df[z_col].fillna(0.0)

    return df


def fit_logit_cost(df: pd.DataFrame, feature_columns: list[str]) -> tuple[float, list[float]]:
    y = df["is_science_official"].to_numpy(dtype=int)
    if len(np.unique(y)) < 2:
        return math.inf, []
    x = df[feature_columns].to_numpy(dtype=float)
    if not np.all(np.isfinite(x)):
        return math.inf, []
    model = LogisticRegression(
        C=100.0,
        solver="lbfgs",
        max_iter=1000,
        fit_intercept=True,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            model.fit(x, y)
        except Exception:
            return math.inf, []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            probabilities = model.predict_proba(x)[:, 1]
        except Exception:
            return math.inf, []
    if not np.all(np.isfinite(probabilities)):
        return math.inf, []

    # Clip probabilities to prevent log(0) numerical instability
    probabilities = np.clip(probabilities, 1e-9, 1 - 1e-9)
    # Calculate Deviance = 2 * LogLoss
    deviance = 2 * log_loss(y, probabilities, normalize=False, labels=[0, 1])
    coefficients = [float(model.intercept_[0]), *[float(v) for v in model.coef_[0]]]
    return float(deviance), coefficients


def compute_segment_costs(
    df: pd.DataFrame,
    bins: list[int],
    feature_columns: list[str],
    min_segment_bins: int,
    min_segment_obs: int,
) -> tuple[dict[tuple[int, int], float], dict[tuple[int, int], list[float]], dict[tuple[int, int], int]]:
    costs: dict[tuple[int, int], float] = {}
    coefficients: dict[tuple[int, int], list[float]] = {}
    obs_counts: dict[tuple[int, int], int] = {}
    bin_to_rows = {bin_value: df[df["time_bin"] == bin_value] for bin_value in bins}
    for start in range(len(bins)):
        segment_parts = []
        for end in range(start + 1, len(bins) + 1):
            segment_parts.append(bin_to_rows[bins[end - 1]])
            if end - start < min_segment_bins:
                continue
            segment_df = pd.concat(segment_parts, ignore_index=True)
            obs_counts[(start, end)] = len(segment_df)
            # Enforce minimum sample size constraint per segment
            if len(segment_df) < min_segment_obs:
                costs[(start, end)] = math.inf
                coefficients[(start, end)] = []
                continue
            deviance, coef = fit_logit_cost(segment_df, feature_columns)
            costs[(start, end)] = deviance
            coefficients[(start, end)] = coef
    return costs, coefficients, obs_counts


def select_breaks(
    costs: dict[tuple[int, int], float],
    n_bins: int,
    n_obs: int,
    n_params: int,
    max_breaks: int,
    min_segment_bins: int,
) -> dict[str, Any]:
    candidates = []
    for break_count in range(max_breaks + 1):
        segments = break_count + 1
        if n_bins < segments * min_segment_bins:
            continue

        # DP table: dp[s][e] stores minimum cumulative deviance up to bin e using s segments
        dp = [[math.inf] * (n_bins + 1) for _ in range(segments + 1)]
        prev = [[-1] * (n_bins + 1) for _ in range(segments + 1)]
        dp[0][0] = 0.0
        for segment in range(1, segments + 1):
            min_end = segment * min_segment_bins
            max_end = n_bins - (segments - segment) * min_segment_bins
            for end in range(min_end, max_end + 1):
                start_min = (segment - 1) * min_segment_bins
                start_max = end - min_segment_bins
                for start in range(start_min, start_max + 1):
                    cost = costs.get((start, end), math.inf)
                    if cost == math.inf or dp[segment - 1][start] == math.inf:
                        continue
                    score = dp[segment - 1][start] + cost
                    if score < dp[segment][end]:
                        dp[segment][end] = score
                        prev[segment][end] = start
        deviance = dp[segments][n_bins]
        if deviance == math.inf:
            continue

        # Backtrack optimal segment boundaries
        boundaries = []
        end = n_bins
        for segment in range(segments, 0, -1):
            start = prev[segment][end]
            boundaries.append((start, end))
            end = start
        boundaries.reverse()

        # Compute Bayesian Information Criterion (BIC)
        bic = deviance + (segments * n_params) * math.log(n_obs)
        candidates.append(
            {
                "break_count": break_count,
                "segments": boundaries,
                "deviance": deviance,
                "bic": bic,
            }
        )
    # Return the candidate configuration with the lowest BIC score
    return min(candidates, key=lambda row: row["bic"])


def build_output_rows(
    selected: dict[str, Any],
    bins: list[int],
    bin_size: int,
    costs: dict[tuple[int, int], float],
    coefficients: dict[tuple[int, int], list[float]],
    obs_counts: dict[tuple[int, int], int],
    tolerance_years: int,
    feature_columns: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    segment_rows = []
    break_rows = []
    coef_names = ["intercept", *feature_columns]
    for segment_index, (start, end) in enumerate(selected["segments"], start=1):
        coef = coefficients.get((start, end), [])
        row = {
            "segment_index": segment_index,
            "start_bin": bins[start],
            "end_bin": bins[end - 1] + bin_size - 1,
            "obs_count": obs_counts.get((start, end), 0),
            "segment_deviance": round(float(costs.get((start, end), math.nan)), 6),
            "selected_break_count": selected["break_count"],
            "selected_total_deviance": round(float(selected["deviance"]), 6),
            "selected_bic": round(float(selected["bic"]), 6),
        }
        for name, value in zip(coef_names, coef):
            row[f"coef_{name}"] = round(value, 8)
        segment_rows.append(row)

        # Record breakpoint meta-data for internal boundaries
        if segment_index > 1:
            break_year = bins[start]
            transition = nearest_transition(break_year)
            break_rows.append(
                {
                    "break_bin_start": break_year,
                    "break_bin_end": break_year + bin_size - 1,
                    "break_year_used": break_year,
                    **transition,
                    "aligned_with_dynastic_transition": int(
                        transition["absolute_distance_years"] <= tolerance_years
                    ),
                    "tolerance_years": tolerance_years,
                    "selected_break_count": selected["break_count"],
                    "selected_bic": round(float(selected["bic"]), 6),
                }
            )
    return break_rows, segment_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, default=600)
    parser.add_argument("--end-year", type=int, default=1912)
    parser.add_argument("--bin-size", type=int, default=50)
    parser.add_argument("--max-breaks", type=int, default=5)
    parser.add_argument("--min-segment-bins", type=int, default=2)
    parser.add_argument("--min-segment-obs", type=int, default=120)
    parser.add_argument("--transition-tolerance-years", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    """Main execution workflow."""
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load and construct panel data frame
    panel = load_panel(args.start_year, args.end_year, args.bin_size)
    bins = sorted(panel["time_bin"].unique().tolist())

    # Compute candidate segment model deviances
    costs, coefficients, obs_counts = compute_segment_costs(
        panel,
        bins,
        FEATURE_COLUMNS,
        min_segment_bins=args.min_segment_bins,
        min_segment_obs=args.min_segment_obs,
    )
    n_params = len(FEATURE_COLUMNS) + 1

    # Select optimal breakpoints via dynamic programming + BIC
    selected = select_breaks(
        costs,
        n_bins=len(bins),
        n_obs=len(panel),
        n_params=n_params,
        max_breaks=args.max_breaks,
        min_segment_bins=args.min_segment_bins,
    )

    # Format and export output artifacts
    break_rows, segment_rows = build_output_rows(
        selected,
        bins,
        args.bin_size,
        costs,
        coefficients,
        obs_counts,
        args.transition_tolerance_years,
        FEATURE_COLUMNS,
    )

    panel.to_csv(PANEL_CSV, index=False)
    pd.DataFrame(break_rows).to_csv(BREAKS_CSV, index=False)
    pd.DataFrame(segment_rows).to_csv(SEGMENTS_CSV, index=False)

    science_count = int(panel["is_science_official"].sum())
    summary = {
        "start_year": args.start_year,
        "end_year": args.end_year,
        "bin_size": args.bin_size,
        "n_bins": len(bins),
        "n_obs": len(panel),
        "science_official_count": science_count,
        "science_official_share_pct": pct(science_count, len(panel)),
        "feature_columns": FEATURE_COLUMNS,
        "selected": selected,
        "breaks": break_rows,
        "dynastic_transitions": MAJOR_DYNASTIC_TRANSITIONS,
        "time_bin_counts": {
            int(bin_value): {
                "obs": int(len(group)),
                "science_officials": int(group["is_science_official"].sum()),
            }
            for bin_value, group in panel.groupby("time_bin")
        },
        "notes": [
            "Dependent variable is is_main from the existing technical-centered network.",
            "This is exploratory because the comparison group is the working network, not the full CBDB population.",
            "Breakpoint values reflect changes in model coefficients, not death-count levels.",
        ],
    }

    SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote {PANEL_CSV}")
    print(f"Wrote {BREAKS_CSV}")
    print(f"Wrote {SEGMENTS_CSV}")
    print(f"Wrote {SUMMARY_JSON}")


if __name__ == "__main__":
    main()