"""Tests for TripleCluster."""

import pytest
from nonganon_tesseract.triple_cluster import Cluster, TripleCluster, _euclidean, _mean_vector


class TestHelpers:
    def test_euclidean_same_point(self):
        assert _euclidean([1.0, 2.0], [1.0, 2.0]) == 0.0

    def test_euclidean_known_distance(self):
        assert abs(_euclidean([0.0, 0.0], [3.0, 4.0]) - 5.0) < 1e-9

    def test_euclidean_different_lengths(self):
        with pytest.raises(ValueError):
            _euclidean([1.0], [1.0, 2.0])

    def test_mean_vector_single(self):
        assert _mean_vector([[3.0, 6.0]]) == [3.0, 6.0]

    def test_mean_vector_multiple(self):
        result = _mean_vector([[0.0, 0.0], [2.0, 4.0]])
        assert result == [1.0, 2.0]

    def test_mean_vector_empty(self):
        assert _mean_vector([]) == []


class TestCluster:
    def test_fit_single_point(self):
        c = Cluster("test")
        c.fit([[1.0, 2.0]])
        assert c.centroid == [1.0, 2.0]

    def test_fit_converges(self):
        data = [[float(i), float(i)] for i in range(10)]
        c = Cluster("test")
        c.fit(data)
        assert c.converged
        assert c.centroid is not None

    def test_fit_empty_raises(self):
        c = Cluster("test")
        with pytest.raises(ValueError):
            c.fit([])

    def test_fingerprint_is_hex(self):
        c = Cluster("test")
        c.fit([[1.0, 2.0]])
        fp = c.fingerprint()
        assert len(fp) == 64
        int(fp, 16)

    def test_repr_contains_id(self):
        c = Cluster("alpha")
        assert "alpha" in repr(c)


class TestTripleCluster:
    def _make_data(self, n=30, dims=2):
        return [[float(i % 10), float((i * 3) % 7)] for i in range(n)]

    def test_fit_produces_meta_centroid(self):
        tc = TripleCluster(seed=0)
        tc.fit(self._make_data())
        assert tc.meta_centroid is not None
        assert len(tc.meta_centroid) == 2

    def test_fit_requires_at_least_3_points(self):
        tc = TripleCluster()
        with pytest.raises(ValueError):
            tc.fit([[1.0, 2.0], [3.0, 4.0]])

    def test_summary_has_three_clusters(self):
        tc = TripleCluster(seed=1)
        tc.fit(self._make_data())
        s = tc.summary()
        assert len(s["clusters"]) == 3

    def test_summary_before_fit(self):
        tc = TripleCluster()
        assert tc.summary()["status"] == "not_fitted"

    def test_reproducibility_with_seed(self):
        data = self._make_data(60)
        tc1 = TripleCluster(seed=42)
        tc1.fit(data)
        tc2 = TripleCluster(seed=42)
        tc2.fit(data)
        assert tc1.meta_centroid == tc2.meta_centroid

    def test_all_clusters_converged(self):
        tc = TripleCluster(seed=7)
        tc.fit(self._make_data(30))
        for c in tc.clusters:
            assert c.converged

    def test_partition_equal_split(self):
        items = list(range(9))
        parts = TripleCluster._partition(items, 3)
        assert len(parts) == 3
        assert all(len(p) == 3 for p in parts)

    def test_partition_unequal_split(self):
        items = list(range(10))
        parts = TripleCluster._partition(items, 3)
        total = sum(len(p) for p in parts)
        assert total == 10
