#!/usr/bin/env python3
"""Unit tests for career_trajectory_analysis module."""

import unittest
import numpy as np
import pandas as pd

from career_trajectory_analysis import (
    safe_pct,
    safe_z,
    normalize_category,
    office_domain,
    collapse_adjacent,
    edit_distance,
    weighted_k_medoids,
)


class TestCareerTrajectoryAnalysis(unittest.TestCase):
    """Unit tests for career trajectory processing and sequence analysis."""

    def test_safe_pct(self) -> None:
        """Test safe percentage calculation."""
        self.assertEqual(safe_pct(25, 100), 25.0)
        self.assertEqual(safe_pct(10, 0), 0.0)

    def test_safe_z(self) -> None:
        """Test Z-score normalization logic."""
        series = pd.Series([10.0, 20.0, 30.0])
        z_scores = safe_z(series)
        self.assertAlmostEqual(z_scores.mean(), 0.0)
        self.assertAlmostEqual(z_scores.std(ddof=0), 1.0)

        flat_series = pd.Series([5.0, 5.0, 5.0])
        self.assertTrue((safe_z(flat_series) == 0.0).all())

    def test_normalize_category(self) -> None:
        """Test office category normalization."""
        self.assertEqual(normalize_category(" Central "), "Central")
        self.assertEqual(normalize_category("unknown"), "unknown_category")
        self.assertEqual(normalize_category(None), "missing_category")

    def test_office_domain(self) -> None:
        """Test keyword-based technical domain classification."""
        row_astronomy = pd.Series({"c_office_chn": "欽天監監正"})
        self.assertEqual(office_domain(row_astronomy), "astronomical")

        row_medical = pd.Series({"c_office_chn": "太醫院御醫"})
        self.assertEqual(office_domain(row_medical), "medical")

        row_general = pd.Series({"c_office_chn": "知州"})
        self.assertEqual(office_domain(row_general), "other")

    def test_collapse_adjacent(self) -> None:
        """Test collapsing of adjacent repeated career states."""
        sequence = ["domain_medical", "domain_medical", "domain_astronomical", "domain_medical"]
        collapsed = collapse_adjacent(sequence)
        self.assertEqual(collapsed, ["domain_medical", "domain_astronomical", "domain_medical"])

    def test_edit_distance(self) -> None:
        """Test Levenshtein Optimal Matching edit distance."""
        seq1 = ("A", "B", "C")
        seq2 = ("A", "B", "C")
        self.assertEqual(edit_distance(seq1, seq2), 0.0)

        seq3 = ("A", "X", "C")
        self.assertEqual(edit_distance(seq1, seq3), 1.0)  # 1 substitution

        seq4 = ("A", "B")
        self.assertEqual(edit_distance(seq1, seq4), 1.0)  # 1 deletion

    def test_weighted_k_medoids(self) -> None:
        """Test weighted k-medoids sequence clustering algorithm."""
        # Simple symmetric distance matrix for 3 unique sequences
        dist = np.array([
            [0.0, 1.0, 5.0],
            [1.0, 0.0, 5.0],
            [5.0, 5.0, 0.0],
        ])
        weights = np.array([10.0, 10.0, 1.0])
        assignments, medoids = weighted_k_medoids(dist, weights, k=2)
        
        self.assertEqual(len(medoids), 2)
        # Sequence 0 and 1 should cluster together
        self.assertEqual(assignments[0], assignments[1])


if __name__ == "__main__":
    unittest.main()