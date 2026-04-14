"""
Max Heap implementation for KNN nearest-neighbor tracking.

Items are (distance, data) tuples. The heap keeps the K nearest neighbors
seen so far, with the worst (largest distance) at the root so it can be
ejected in O(log K) when a closer point arrives.

Time complexities:
    add()            – O(log K)
    worst_distance() – O(1)
    get_all()        – O(K)
"""


class MaxHeap:
    """Max heap bounded to capacity K, keyed on distance.

    Stores items as (distance, data) tuples.  The largest distance
    lives at index 0 (the root), so worst_distance() is O(1) and
    replacement of the worst element is O(log K).
    """

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        self._capacity = capacity
        self._heap: list[tuple[float, object]] = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def add(self, item: tuple[float, object]) -> None:
        """Add *item* to the heap.

        If the heap is not yet full the item is appended and bubbled up.
        If the heap is full and *item* has a smaller distance than the
        current worst (root), the root is replaced and bubbled down.
        Otherwise the item is discarded.

        Time complexity: O(log K)
        """
        distance = item[0]

        if len(self._heap) < self._capacity:
            self._heap.append(item)
            self._bubble_up(len(self._heap) - 1)
        elif distance < self._heap[0][0]:
            self._heap[0] = item
            self._bubble_down(0)

    def worst_distance(self) -> float:
        """Return the largest distance currently in the heap (O(1)).

        Raises IndexError if the heap is empty.
        """
        if not self._heap:
            raise IndexError("heap is empty")
        return self._heap[0][0]

    def get_all(self) -> list[tuple[float, object]]:
        """Return a shallow copy of all (distance, data) items (O(K))."""
        return list(self._heap)

    def __len__(self) -> int:
        return len(self._heap)

    # ------------------------------------------------------------------
    # Internal heap operations
    # ------------------------------------------------------------------

    def _bubble_up(self, index: int) -> None:
        """Move the node at *index* up until the heap property is restored."""
        heap = self._heap
        while index > 0:
            parent = (index - 1) // 2
            if heap[index][0] > heap[parent][0]:
                heap[index], heap[parent] = heap[parent], heap[index]
                index = parent
            else:
                break

    def _bubble_down(self, index: int) -> None:
        """Move the node at *index* down until the heap property is restored."""
        heap = self._heap
        size = len(heap)
        while True:
            largest = index
            left = 2 * index + 1
            right = 2 * index + 2

            if left < size and heap[left][0] > heap[largest][0]:
                largest = left
            if right < size and heap[right][0] > heap[largest][0]:
                largest = right

            if largest == index:
                break

            heap[index], heap[largest] = heap[largest], heap[index]
            index = largest
