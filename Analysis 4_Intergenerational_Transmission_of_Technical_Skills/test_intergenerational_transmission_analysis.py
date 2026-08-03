#!/usr/bin/env python3
"""Unit tests for intergenerational_transmission_analysis module."""

import unittest
import pandas as pd
from intergenerational_transmission_analysis import (
    safe_pct,
    safe_z,
    cluster_for_year,
)


class TestIntergenerationalTransmissionAnalysis(unittest.TestCase):
    """Unit tests for intergenerational transmission helper functions."""

    def test_safe_pct(self) -> None:
        """Test safe percentage calculation."""
        self.assertEqual(safe_pct(50, 200), 25.0)
        self.assertEqual(safe_pct(0, 100), 0.0)
        self.assertEqual(safe_pct(10, 0), 0.0)

    def test_safe_z(self) -> None:
        """Test Z-score normalization and zero-variance boundary handling."""
        series = pd.Series([10.0, 20.0, 30.0])
        z_scores = safe_z(series)
        self.assertAlmostEqual(z_scores.mean(), 0.0)
        self.assertAlmostEqual(z_scores.std(ddof=0), 1.0)

        flat_series = pd.Series([5.0, 5.0, 5.0])
        self.assertTrue((safe_z(flat_series) == 0.0).all())

    def test_cluster_for_year(self) -> None:
        """Test year-to-temporal-cluster allocation logic."""
        self.assertEqual(cluster_for_year(1000), "cluster1_song_and_before")
        self.assertEqual(cluster_for_year(1400), "cluster2_yuan_ming")
        self.assertEqual(cluster_for_year(1700), "cluster3_qing_after")
        self.assertEqual(cluster_for_year(None), "missing_year")


if __name__ == "__main__":
    unittest.main()