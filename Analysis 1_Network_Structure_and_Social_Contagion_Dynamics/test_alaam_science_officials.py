#!/usr/bin/env python3
"""Unit tests for alaam_science_officials module."""

import unittest
import pandas as pd
import networkx as nx

from alaam_science_officials import (
    safe_z,
    cluster_for_year,
    distance_two_sets,
    build_science_graph,
    CLUSTER_SCHEMES,
)


class TestALAAMFunctions(unittest.TestCase):
    """Unit tests for network processing and mathematical helper functions."""

    def test_safe_z(self) -> None:
        """Test Z-score normalization logic including uniform value handling."""
        series = pd.Series([10.0, 20.0, 30.0])
        z_scores = safe_z(series)
        self.assertAlmostEqual(z_scores.mean(), 0.0)
        self.assertAlmostEqual(z_scores.std(ddof=0), 1.0)

        # Uniform series should return zeroes without error
        flat_series = pd.Series([5.0, 5.0, 5.0])
        self.assertTrue((safe_z(flat_series) == 0.0).all())

    def test_cluster_for_year(self) -> None:
        """Test temporal cluster categorization bounds."""
        clusters = CLUSTER_SCHEMES["break4"]
        self.assertEqual(cluster_for_year(800, clusters), "pre_song")
        self.assertEqual(cluster_for_year(1100, clusters), "song_to_yuan")
        self.assertEqual(cluster_for_year(1300, clusters), "yuan_to_ming")
        self.assertEqual(cluster_for_year(1500, clusters), "ming_to_qing")
        self.assertEqual(cluster_for_year(2025, clusters), "out_of_scope")
        self.assertEqual(cluster_for_year(None, clusters), "out_of_scope")

    def test_distance_two_sets(self) -> None:
        """Test 2-hop neighbor generation on a path graph (1 - 2 - 3 - 4)."""
        graph = nx.path_graph([1, 2, 3, 4])
        two_hops = distance_two_sets(graph)
        self.assertEqual(two_hops[1], {3})
        self.assertEqual(two_hops[2], {4})
        self.assertEqual(two_hops[3], {1})
        self.assertEqual(two_hops[4], {2})

    def test_build_science_graph_induced(self) -> None:
        """Test induced subgraph construction."""
        data = {
            "nodes": [{"id": 1}, {"id": 2}, {"id": 3}],
            "edges": [
                {"source": 1, "target": 2},
                {"source": 2, "target": 3},
                {"source": 1, "target": 4},
            ],
        }
        retained = {1, 2, 3}
        graph = build_science_graph(data, retained, graph_mode="induced")
        self.assertEqual(set(graph.nodes), {1, 2, 3})
        self.assertEqual(set(graph.edges), {(1, 2), (2, 3)})


if __name__ == "__main__":
    unittest.main()