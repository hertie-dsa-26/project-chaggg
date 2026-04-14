import unittest

from algorithms.max_heap import MaxHeap


class TestMaxHeapInit(unittest.TestCase):
    def test_starts_empty(self):
        h = MaxHeap(3)
        self.assertEqual(len(h), 0)

    def test_invalid_capacity(self):
        with self.assertRaises(ValueError):
            MaxHeap(0)
        with self.assertRaises(ValueError):
            MaxHeap(-1)


class TestMaxHeapProperty(unittest.TestCase):
    """Root must always hold the largest distance."""

    def _assert_heap_property(self, heap: MaxHeap) -> None:
        items = heap.get_all()
        for i, (dist, _) in enumerate(items):
            parent = (i - 1) // 2
            if i > 0:
                self.assertGreaterEqual(
                    items[parent][0],
                    dist,
                    f"Heap property violated: parent[{parent}]={items[parent][0]} < child[{i}]={dist}",
                )

    def test_insert_sequence_maintains_heap(self):
        h = MaxHeap(10)
        distances = [3.0, 1.0, 4.0, 1.5, 9.0, 2.6, 5.0]
        for d in distances:
            h.add((d, f"point_{d}"))
        self._assert_heap_property(h)

    def test_root_is_max(self):
        h = MaxHeap(5)
        for d in [2.0, 5.0, 1.0, 4.0, 3.0]:
            h.add((d, None))
        self.assertEqual(h.worst_distance(), 5.0)


class TestMaxHeapCapacity(unittest.TestCase):
    def test_does_not_exceed_capacity(self):
        h = MaxHeap(3)
        for d in [1.0, 2.0, 3.0, 4.0, 5.0]:
            h.add((d, None))
        self.assertEqual(len(h), 3)

    def test_keeps_k_smallest_distances(self):
        h = MaxHeap(3)
        for d in [5.0, 3.0, 1.0, 4.0, 2.0]:
            h.add((d, None))
        kept = sorted(item[0] for item in h.get_all())
        self.assertEqual(kept, [1.0, 2.0, 3.0])

    def test_worse_item_discarded_when_full(self):
        h = MaxHeap(2)
        h.add((1.0, "a"))
        h.add((2.0, "b"))
        h.add((10.0, "c"))  # worse than current worst (2.0) → discarded
        kept_distances = sorted(item[0] for item in h.get_all())
        self.assertEqual(kept_distances, [1.0, 2.0])

    def test_better_item_replaces_worst_when_full(self):
        h = MaxHeap(2)
        h.add((1.0, "a"))
        h.add((5.0, "b"))
        h.add((2.0, "c"))  # better than worst (5.0) → replaces it
        kept_distances = sorted(item[0] for item in h.get_all())
        self.assertEqual(kept_distances, [1.0, 2.0])

    def test_capacity_one(self):
        h = MaxHeap(1)
        h.add((3.0, "x"))
        h.add((1.0, "y"))  # better → replaces
        h.add((5.0, "z"))  # worse → discarded
        self.assertEqual(len(h), 1)
        self.assertEqual(h.worst_distance(), 1.0)


class TestWorstDistance(unittest.TestCase):
    def test_raises_on_empty_heap(self):
        h = MaxHeap(5)
        with self.assertRaises(IndexError):
            h.worst_distance()

    def test_single_element(self):
        h = MaxHeap(5)
        h.add((7.5, "a"))
        self.assertEqual(h.worst_distance(), 7.5)

    def test_updates_after_replacement(self):
        h = MaxHeap(3)
        for d in [3.0, 1.0, 2.0]:
            h.add((d, None))
        self.assertEqual(h.worst_distance(), 3.0)
        h.add((0.5, None))  # replaces 3.0
        self.assertEqual(h.worst_distance(), 2.0)


class TestGetAll(unittest.TestCase):
    def test_empty_heap(self):
        h = MaxHeap(3)
        self.assertEqual(h.get_all(), [])

    def test_returns_copy(self):
        h = MaxHeap(3)
        h.add((1.0, "a"))
        result = h.get_all()
        result.clear()
        self.assertEqual(len(h), 1)

    def test_all_inserted_items_present(self):
        h = MaxHeap(5)
        items = [(float(i), f"p{i}") for i in range(1, 4)]
        for item in items:
            h.add(item)
        self.assertCountEqual(h.get_all(), items)


class TestBubbleOperations(unittest.TestCase):
    """Verify bubble_up and bubble_down keep the heap valid after every op."""

    def _heap_valid(self, heap: MaxHeap) -> bool:
        items = heap.get_all()
        for i in range(1, len(items)):
            parent = (i - 1) // 2
            if items[i][0] > items[parent][0]:
                return False
        return True

    def test_bubble_up_after_each_insert(self):
        h = MaxHeap(20)
        for d in [10, 4, 7, 2, 9, 1, 6, 8, 3, 5]:
            h.add((float(d), None))
            self.assertTrue(self._heap_valid(h), f"Heap invalid after inserting {d}")

    def test_bubble_down_after_replacement(self):
        h = MaxHeap(5)
        for d in [5.0, 4.0, 3.0, 2.0, 1.0]:
            h.add((d, None))
        for new_d in [0.9, 0.8, 0.7]:
            h.add((new_d, None))
            self.assertTrue(self._heap_valid(h), f"Heap invalid after replacement with {new_d}")


if __name__ == "__main__":
    unittest.main()
