"""
MaxHeap — a fixed-capacity max heap built from scratch (no heapq).

Each element is stored as a (distance, target) tuple.
The largest distance always sits at index 0 (the root), giving O(1) access
to the *farthest* of the current k neighbours — exactly what KNN needs to
decide whether a new point is worth keeping.

Heap property: parent distance >= both children's distances.

Time complexities
-----------------
add()            O(log k)  — bubble_up or bubble_down after insert/replace
worst_distance() O(1)      — root is always the maximum
get_all()        O(k)      — copies the internal list
"""

from typing import List, Tuple, Optional


class MaxHeap:
    """
    A capacity-bounded max heap for KNN regression.

    Stores (distance, target) pairs. The root is always the pair with the
    *largest* distance, so we can decide in O(1) whether a new candidate
    point is closer than the current worst neighbour.
    """

    def __init__(self, capacity: int) -> None:
        """
        Initialise an empty heap.

        Parameters
        ----------
        capacity : int
            Maximum number of elements (= k in KNN).
        """
        if capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        self._capacity: int = capacity
        self._heap: List[Tuple[float, float]] = []  # (distance, target)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def add(self, distance: float, target: float) -> None:
        """
        Insert a new (distance, target) pair into the heap.

        - If the heap is not yet full, append and bubble up.
        - If the heap is full and the new distance is smaller than the
          current worst (root), replace the root and bubble down.
        - If the heap is full and the new distance is >= the worst,
          discard the new point — it is not a better neighbour.

        Time: O(log k)
        """
        if len(self._heap) < self._capacity:
            self._heap.append((distance, target))
            self._bubble_up(len(self._heap) - 1)
        elif distance < self._heap[0][0]:
            # New point is closer than the current farthest → replace root
            self._heap[0] = (distance, target)
            self._bubble_down(0)
        # else: new point is farther away — ignore it

    def worst_distance(self) -> float:
        """
        Return the largest distance currently in the heap (the root).

        This is the distance to the *farthest* of the k stored neighbours.
        KNN uses this to decide whether a new candidate should evict it.

        Time: O(1)

        Raises
        ------
        IndexError
            If the heap is empty.
        """
        if not self._heap:
            raise IndexError("worst_distance() called on an empty heap")
        return self._heap[0][0]

    def get_all(self) -> List[Tuple[float, float]]:
        """
        Return a shallow copy of all (distance, target) pairs in the heap.

        The order is heap-internal, not sorted by distance.

        Time: O(k)
        """
        return list(self._heap)

    def __len__(self) -> int:
        return len(self._heap)

    def __repr__(self) -> str:  # pragma: no cover
        return f"MaxHeap(capacity={self._capacity}, size={len(self._heap)}, root={self._heap[0] if self._heap else None})"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parent(i: int) -> int:
        return (i - 1) // 2

    @staticmethod
    def _left(i: int) -> int:
        return 2 * i + 1

    @staticmethod
    def _right(i: int) -> int:
        return 2 * i + 2

    def _bubble_up(self, i: int) -> None:
        """
        Move element at index i upward until the heap property is restored.

        A node should rise if its distance is greater than its parent's.

        Time: O(log k)
        """
        while i > 0:
            parent = self._parent(i)
            if self._heap[i][0] > self._heap[parent][0]:
                self._heap[i], self._heap[parent] = self._heap[parent], self._heap[i]
                i = parent
            else:
                break  # heap property satisfied

    def _bubble_down(self, i: int) -> None:
        """
        Move element at index i downward until the heap property is restored.

        A node should sink if either child has a larger distance; it swaps
        with the *larger* of the two children to maintain the max-heap
        property throughout the subtree.

        Time: O(log k)
        """
        n = len(self._heap)
        while True:
            largest = i
            left = self._left(i)
            right = self._right(i)

            if left < n and self._heap[left][0] > self._heap[largest][0]:
                largest = left
            if right < n and self._heap[right][0] > self._heap[largest][0]:
                largest = right

            if largest == i:
                break  # heap property satisfied

            self._heap[i], self._heap[largest] = self._heap[largest], self._heap[i]
            i = largest


# ---------------------------------------------------------------------------
# Updated KNN regression — uses MaxHeap directly (no heapq, no negation)
# ---------------------------------------------------------------------------

import math


def euclidean_distance(a: List[float], b: List[float]) -> float:
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


def knn_regression(
    X_train: List[List[float]],
    y_train: List[float],
    query: List[float],
    k: int,
) -> float:
    """
    Predict the target value for `query` using k-nearest-neighbour regression.

    Uses MaxHeap so the farthest current neighbour is always at the root,
    enabling O(1) comparison and O(log k) replacement.

    Time:  O(n log k)
    Space: O(k)
    """
    if k <= 0:
        raise ValueError("k must be a positive integer")
    if k > len(X_train):
        raise ValueError("k cannot exceed the number of training points")

    heap = MaxHeap(capacity=k)

    for features, target in zip(X_train, y_train):
        dist = euclidean_distance(features, query)
        heap.add(dist, target)

    return sum(target for _, target in heap.get_all()) / k