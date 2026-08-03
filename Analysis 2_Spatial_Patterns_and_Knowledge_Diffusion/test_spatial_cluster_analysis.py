#!/usr/bin/env python3
"""Unit tests for spatial_cluster_analysis module."""

import unittest
import numpy as np
import pandas as pd

from spatial_cluster_analysis import (
    safe_pct,
    safe_z,
    haversine_km,
    knn_indices,
    morans_i,
    softmax_with_baseline,
)


class TestSpatialClusterAnalysis(unittest.TestCase):
    """Unit tests for mathematical and spatial computation functions."""

    def test_safe_pct(self) -> None:
        """Test percentage calculation with zero-division safety."""
        self.assertEqual(safe_pct(50, 200), 25.0)
        self.assertEqual(safe_pct(0, 100), 0.0)
        self.assertEqual(safe_pct(10, 0), 0.0)

    def test_safe_z(self) -> None:
        """Test Z-score normalization and zero variance handling."""
        series = pd.Series([10.0, 20.0, 30.0])
        z_scores = safe_z(series)
        self.assertAlmostEqual(z_scores.mean(), 0.0)
        self.assertAlmostEqual(z_scores.std(ddof=0), 1.0)

        flat_series = pd.Series([5.0, 5.0, 5.0])
        self.assertTrue((safe_z(flat_series) == 0.0).all())

    def test_haversine_km(self) -> None:
        """Test Haversine distance calculation against known distance."""
        # Distance between Beijing (116.4, 39.9) and Shanghai (121.47, 31.23) ~ 1068 km
        dist = haversine_km(116.4, 39.9, 121.47, 31.23)
        self.assertGreater(dist, 1000.0)
        self.assertLess(dist, 1150.0)
        # Distance to self should be zero
        self.assertAlmostEqual(haversine_km(100.0, 30.0, 100.0, 30.0), 0.0)

    def test_knn_indices(self) -> None:
        """Test nearest neighbor index generation."""
        coords = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [10.0, 10.0],
        ])
        knn = knn_indices(coords, k=2)
        self.assertEqual(knn.shape, (4, 2))
        # Nearest neighbors of (0,0) should be (1,0) and (0,1) -> indices 1 and 2
        self.assertEqual(set(knn[0]), {1, 2})

    def test_morans_i(self) -> None:
        """Test Moran's I spatial autocorrelation computation."""
        binary = np.array([1, 1, 0, 0])
        # Define 1-nearest neighbor matrix
        neighbors = np.array([[1], [0], [3], [2]])
        # Highly positive spatial autocorrelation
        mi = morans_i(binary, neighbors)
        self.assertGreater(mi, 0.0)

    def test_softmax_with_baseline(self) -> None:
        """Test multinomial softmax probability normalization."""
        eta = np.array([[0.0, 0.0], [2.0, -1.0]])
        probs = softmax_with_baseline(eta)
        self.assertEqual(probs.shape, (2, 3))
        # Check that row probabilities sum to 1.0
        np.testing.assert_allclose(probs.sum(axis=1), np.array([1.0, 1.0]))


if __name__ == "__main__":
    unittest.main()