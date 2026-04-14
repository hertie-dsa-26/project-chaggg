"""
Unit tests for MaxHeap and the updated knn_regression.

Run with:  python -m pytest test_max_heap.py -v
"""

import math
import pytest
from max_heap import MaxHeap, knn_regression, euclidean_distance


# ===========================================================================
# Helpers
# ===========================================================================

def is_valid_max_heap(heap: MaxHeap) -> bool:
    """Walk every parent-child triple and assert parent >= children."""
    h = heap._heap
    n = len(h)
    for i in range(n):
        left = 2 * i + 1
        right = 2 * i + 2
        if left < n and h[i][0] < h[left][0]:
            return False
        if right < n and h[i][0] < h[right][0]:
            return False
    return True


# ===========================================================================
# MaxHeap — initialisation
# ===========================================================================

class TestMaxHeapInit:
    def test_empty_on_creation(self):
        h = MaxHeap(capacity=3)
        assert len(h) == 0

    def test_zero_capacity_raises(self):
        with pytest.raises(ValueError):
            MaxHeap(capacity=0)

    def test_negative_capacity_raises(self):
        with pytest.raises(ValueError):
            MaxHeap(capacity=-5)

    def test_capacity_of_one_accepted(self):
        h = MaxHeap(capacity=1)
        assert len(h) == 0


# ===========================================================================
# MaxHeap — add / capacity enforcement
# ===========================================================================

class TestMaxHeapAdd:
    def test_insert_below_capacity(self):
        h = MaxHeap(capacity=3)
        h.add(1.0, 10.0)
        h.add(2.0, 20.0)
        assert len(h) == 2

    def test_insert_up_to_capacity(self):
        h = MaxHeap(capacity=3)
        h.add(1.0, 10.0)
        h.add(2.0, 20.0)
        h.add(3.0, 30.0)
        assert len(h) == 3

    def test_size_does_not_exceed_capacity(self):
        h = MaxHeap(capacity=3)
        for i in range(10):
            h.add(float(i), float(i * 10))
        assert len(h) == 3

    def test_closer_point_replaces_farthest(self):
        """When full, a closer point should evict the current farthest."""
        h = MaxHeap(capacity=2)
        h.add(5.0, 50.0)
        h.add(3.0, 30.0)
        # root = 5.0; add a closer point (1.0 < 5.0) → evicts 5.0
        h.add(1.0, 10.0)
        distances = {d for d, _ in h.get_all()}
        assert 5.0 not in distances
        assert 1.0 in distances

    def test_farther_point_is_ignored(self):
        """When full, a farther-or-equal point should be silently discarded."""
        h = MaxHeap(capacity=2)
        h.add(3.0, 30.0)
        h.add(1.0, 10.0)
        # root = 3.0; new point 6.0 > 3.0 → ignored
        h.add(6.0, 60.0)
        distances = {d for d, _ in h.get_all()}
        assert 6.0 not in distances
        assert len(h) == 2

    def test_heap_property_maintained_after_many_inserts(self):
        h = MaxHeap(capacity=5)
        import random
        random.seed(42)
        for _ in range(30):
            h.add(random.uniform(0, 100), random.uniform(0, 500))
        assert is_valid_max_heap(h)


# ===========================================================================
# MaxHeap — worst_distance
# ===========================================================================

class TestWorstDistance:
    def test_raises_on_empty_heap(self):
        h = MaxHeap(capacity=3)
        with pytest.raises(IndexError):
            h.worst_distance()

    def test_root_is_maximum(self):
        h = MaxHeap(capacity=4)
        for d in [2.0, 5.0, 1.0, 4.0]:
            h.add(d, d * 10)
        assert h.worst_distance() == 5.0

    def test_worst_distance_after_eviction(self):
        """After a closer point evicts the worst, root should update."""
        h = MaxHeap(capacity=3)
        h.add(4.0, 40.0)
        h.add(7.0, 70.0)
        h.add(2.0, 20.0)
        # heap full; worst = 7.0
        assert h.worst_distance() == 7.0
        # add 1.0 → evicts 7.0; new worst should be 4.0
        h.add(1.0, 10.0)
        assert h.worst_distance() == 4.0

    def test_single_element(self):
        h = MaxHeap(capacity=1)
        h.add(3.5, 99.0)
        assert h.worst_distance() == 3.5


# ===========================================================================
# MaxHeap — get_all
# ===========================================================================

class TestGetAll:
    def test_returns_copy_not_reference(self):
        h = MaxHeap(capacity=3)
        h.add(1.0, 10.0)
        snapshot = h.get_all()
        snapshot.clear()
        assert len(h) == 1  # original unaffected

    def test_returns_all_elements(self):
        h = MaxHeap(capacity=4)
        pairs = [(1.0, 10.0), (3.0, 30.0), (2.0, 20.0), (4.0, 40.0)]
        for d, t in pairs:
            h.add(d, t)
        result = h.get_all()
        assert len(result) == 4
        assert set(result) == set(pairs)

    def test_empty_heap_returns_empty_list(self):
        h = MaxHeap(capacity=3)
        assert h.get_all() == []


# ===========================================================================
# MaxHeap — bubble_up / bubble_down (structural)
# ===========================================================================

class TestHeapStructure:
    def test_bubble_up_single_insert(self):
        h = MaxHeap(capacity=5)
        h.add(10.0, 1.0)
        assert h._heap[0][0] == 10.0

    def test_bubble_up_preserves_max_at_root(self):
        h = MaxHeap(capacity=5)
        for d in [3.0, 1.0, 4.0, 1.5, 9.0]:
            h.add(d, 0.0)
        assert h._heap[0][0] == 9.0
        assert is_valid_max_heap(h)

    def test_bubble_down_after_replacement(self):
        """Replacing root and calling bubble_down should restore heap property."""
        h = MaxHeap(capacity=4)
        for d in [8.0, 5.0, 6.0, 3.0]:
            h.add(d, 0.0)
        # Force a tighter replacement — add a point closer than root (8.0)
        h.add(1.0, 0.0)
        assert is_valid_max_heap(h)
        assert h._heap[0][0] == 6.0  # new max after evicting 8.0


# ===========================================================================
# KNN regression
# ===========================================================================

class TestKNNRegression:
    X = [
        [1.0, 2.0],
        [2.0, 3.0],
        [3.0, 1.0],
        [6.0, 5.0],
        [7.0, 8.0],
        [8.0, 6.0],
        [1.5, 1.5],
        [5.0, 4.0],
    ]
    y = [10.0, 15.0, 12.0, 40.0, 55.0, 50.0, 11.0, 30.0]

    def test_k1_returns_nearest_target(self):
        """k=1 should return the target of the single closest point."""
        query = [1.0, 2.0]  # exact match with first training point
        result = knn_regression(self.X, self.y, query, k=1)
        assert result == 10.0

    def test_k3_matches_brute_force(self):
        """
        k=3 result must equal the average of some valid set of 3 nearest
        neighbours. When points tie at the k-th boundary, multiple answers
        are correct — we accept any of them.
        """
        query = [2.5, 2.5]
        k = 3
        distances = sorted(
            (euclidean_distance(f, query), t)
            for f, t in zip(self.X, self.y)
        )
        kth_dist = distances[k - 1][0]

        # All points strictly closer than the k-th, plus any tied at k-th
        candidates = [(d, t) for d, t in distances if d <= kth_dist + 1e-12]

        # Any combination of k points from `candidates` is a valid answer
        from itertools import combinations
        valid_predictions = {
            sum(t for _, t in combo) / k
            for combo in combinations(candidates, k)
        }

        result = knn_regression(self.X, self.y, query, k=k)
        assert any(math.isclose(result, v, rel_tol=1e-9) for v in valid_predictions)

    def test_k_equals_n_returns_mean(self):
        """When k equals the full training set, result is the global mean."""
        n = len(self.X)
        result = knn_regression(self.X, self.y, [4.0, 4.0], k=n)
        expected = sum(self.y) / n
        assert math.isclose(result, expected, rel_tol=1e-9)

    def test_invalid_k_zero_raises(self):
        with pytest.raises(ValueError):
            knn_regression(self.X, self.y, [1.0, 1.0], k=0)

    def test_invalid_k_too_large_raises(self):
        with pytest.raises(ValueError):
            knn_regression(self.X, self.y, [1.0, 1.0], k=len(self.X) + 1)

    def test_one_dimensional_features(self):
        X1 = [[1.0], [2.0], [5.0], [10.0]]
        y1 = [1.0, 2.0, 5.0, 10.0]
        result = knn_regression(X1, y1, [2.1], k=2)
        # Two nearest are 2.0 and 1.0 → average = 1.5
        assert math.isclose(result, 1.5, rel_tol=1e-9)

    def test_far_query_point(self):
        """A query far from all points should still return a valid prediction."""
        query = [100.0, 100.0]
        result = knn_regression(self.X, self.y, query, k=3)
        assert isinstance(result, float)
        assert not math.isnan(result)