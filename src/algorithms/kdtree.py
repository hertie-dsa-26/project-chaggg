from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class KDNode:
    point: np.ndarray  # shape (k,)
    index: int
    axis: int
    left: KDNode | None = None
    right: KDNode | None = None


def build_kdtree(
    points: np.ndarray, indices: np.ndarray | None = None, depth: int = 0
) -> KDNode | None:
    """
    Build a KD-tree from points (n, k) using median split.
    Returns root KDNode or None if empty.
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2:
        raise ValueError("points must be (n, k)")
    n, k = pts.shape
    if n == 0:
        return None
    if indices is None:
        indices = np.arange(n, dtype=int)
    else:
        indices = np.asarray(indices, dtype=int)
        if len(indices) != n:
            raise ValueError("indices length must match points rows")

    axis = depth % k
    order = np.argsort(pts[:, axis], kind="mergesort")
    pts = pts[order]
    indices = indices[order]
    mid = n // 2

    return KDNode(
        point=pts[mid],
        index=int(indices[mid]),
        axis=int(axis),
        left=build_kdtree(pts[:mid], indices[:mid], depth + 1),
        right=build_kdtree(pts[mid + 1 :], indices[mid + 1 :], depth + 1),
    )


def _euclidean(a: np.ndarray, b: np.ndarray) -> float:
    d = a - b
    return float(np.dot(d, d))  # squared distance


def knn_bruteforce(points: np.ndarray, query: np.ndarray, k: int) -> list[tuple[float, int]]:
    """
    Exact k smallest squared Euclidean distances (debug / regression check vs KD-tree).
    Ties: stable sort by (distance, index) so behavior matches exhaustive reference.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    pts = np.asarray(points, dtype=float)
    q = np.asarray(query, dtype=float).reshape(-1)
    if pts.ndim != 2 or pts.shape[1] != q.shape[0]:
        raise ValueError("points must be (n, d) with len(query)==d")
    n = pts.shape[0]
    if n == 0:
        return []
    d2 = np.sum((pts - q) ** 2, axis=1)
    order = np.lexsort((np.arange(n, dtype=int), d2))
    out: list[tuple[float, int]] = []
    for j in order[: min(k, n)]:
        out.append((float(d2[j]), int(j)))
    return out


def knn_query(root: KDNode | None, query: np.ndarray, k: int) -> list[tuple[float, int]]:
    """
    Return up to k nearest neighbors as list of (squared_distance, index).
    Simple backtracking KD-tree query.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    if root is None:
        return []
    q = np.asarray(query, dtype=float).reshape(-1)

    best: list[tuple[float, int]] = []

    def push(dist2: float, idx: int) -> None:
        nonlocal best
        best.append((dist2, idx))
        best.sort(key=lambda t: t[0])
        if len(best) > k:
            best = best[:k]

    def worst_dist2() -> float:
        if len(best) < k:
            return float("inf")
        return best[-1][0]

    def search(node: KDNode | None) -> None:
        if node is None:
            return
        dist2 = _euclidean(q, node.point)
        push(dist2, node.index)

        axis = node.axis
        diff = q[axis] - node.point[axis]
        near, far = (node.left, node.right) if diff <= 0 else (node.right, node.left)
        search(near)

        # Only visit far branch if hypersphere crosses splitting plane
        if diff * diff <= worst_dist2():
            search(far)

    search(root)
    return best
