"""Persistent (immutable) segment tree supporting point updates and historical range queries.

Each update returns a NEW root; the previous root is untouched, so every version is
permanently queryable. Point updates cost O(log n) time and create O(log n) nodes.
Range queries on any version cost O(log n) time.

The tree is built once (O(n) nodes) from the initial array; subsequent point updates
reuse nodes from the version being updated.

Design decisions:
- Sum is the combine function. This is the most common use case; a monoid-general
  version adds API surface without paying for itself here.
- Identifiers are 1-based non-negative ints. Versions from updates are distinct from
  the initial build (which is always 0), so a tuple (root_ptr, version_int) is returned.
  Tests compare on both.
- Zeros (identity for sum) are cheap and simplify boundary handling.
- No explicit recursion-depth guards: Python's default limit of 1000 covers n up to
  2^500, far beyond practical memory. We do not call sys.setrecursionlimit so callers
  retain control of their own runtime configuration.
"""

from __future__ import annotations


class _Node:
    __slots__ = ("value", "left", "right")

    def __init__(self, value: int, left: "_Node | None" = None, right: "_Node | None" = None) -> None:
        self.value = value
        self.left = left
        self.right = right


class PersistentSegmentTree:
    """Persistent segment tree over a fixed-size integer array, supporting sum queries.

    Versions:
      - Version 0 is the array provided to the constructor.
      - Each successful update yields a new version number, starting at 1.
      - `update(version, index, value)` returns the new version number.
      - `query(version, left, right)` reads from a past version without mutation.

    Time complexity:  O(log n) per update, O(log n) per range query.
    Space complexity: O(n) for the initial build, O(log n) added nodes per update.
    """

    __slots__ = ("_size", "_roots")

    def __init__(self, data: list[int]) -> None:
        if not isinstance(data, list):
            raise TypeError("data must be a list of integers")
        for i, v in enumerate(data):
            if not isinstance(v, int) or isinstance(v, bool):
                raise TypeError(f"data[{i}] must be an int, got {type(v).__name__}")
        self._size = len(data)
        # roots[v] is the tree root for version v. Version 0 is the initial array.
        # We never mutate _roots[v] after setting it; updates only append new entries.
        self._roots: list[_Node | None] = []
        if self._size == 0:
            self._roots.append(None)
        else:
            self._roots.append(self._build(data, 0, self._size - 1))

    def _build(self, data: list[int], low: int, high: int) -> _Node:
        if low == high:
            return _Node(data[low])
        mid = (low + high) >> 1
        left = self._build(data, low, mid)
        right = self._build(data, mid + 1, high)
        return _Node(left.value + right.value, left, right)

    def version_count(self) -> int:
        """Number of versions that exist (>= 1; version 0 plus all updates)."""
        return len(self._roots)

    @property
    def size(self) -> int:
        """Length of the underlying array (fixed for the tree's lifetime)."""
        return self._size

    def query(self, version: int, left: int, right: int) -> int:
        """Return sum of elements in [left, right] (inclusive) at the given version."""
        if not isinstance(version, int) or isinstance(version, bool):
            raise TypeError("version must be an int")
        if not isinstance(left, int) or isinstance(left, bool):
            raise TypeError("left must be an int")
        if not isinstance(right, int) or isinstance(right, bool):
            raise TypeError("right must be an int")
        if version < 0 or version >= len(self._roots):
            raise ValueError(f"version out of range: {version}")
        if left < 0 or right >= self._size or left > right:
            raise ValueError(f"invalid query range [{left}, {right}] for size {self._size}")
        return self._query(self._roots[version], 0, self._size - 1, left, right)

    def _query(self, node: _Node | None, low: int, high: int, left: int, right: int) -> int:
        if node is None or left > high or right < low:
            return 0
        if left <= low and high <= right:
            return node.value
        mid = (low + high) >> 1
        return self._query(node.left, low, mid, left, right) + self._query(node.right, mid + 1, high, left, right)

    def update(self, version: int, index: int, value: int) -> int:
        """Create a new version identical to `version` but with element at `index` set to `value`.

        Returns the new version number. The previous version is unchanged.
        """
        if not isinstance(version, int) or isinstance(version, bool):
            raise TypeError("version must be an int")
        if not isinstance(index, int) or isinstance(index, bool):
            raise TypeError("index must be an int")
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError("value must be an int")
        if version < 0 or version >= len(self._roots):
            raise ValueError(f"version out of range: {version}")
        if index < 0 or index >= self._size:
            raise ValueError(f"index out of range: {index}")
        new_root = self._update(self._roots[version], 0, self._size - 1, index, value)
        new_version = len(self._roots)
        self._roots.append(new_root)
        return new_version

    def _update(self, node: _Node | None, low: int, high: int, index: int, value: int) -> _Node:
        if low == high:
            return _Node(value)
        mid = (low + high) >> 1
        if index <= mid:
            new_left = self._update(node.left, low, mid, index, value)
            right_val = node.right.value if node.right is not None else 0
            return _Node(new_left.value + right_val, new_left, node.right)
        new_right = self._update(node.right, mid + 1, high, index, value)
        left_val = node.left.value if node.left is not None else 0
        return _Node(left_val + new_right.value, node.left, new_right)
