# persistent_segment_tree

A persistent (immutable) segment tree for integer arrays. Each point update returns a new version number; previous versions remain queryable forever. Queries are range sums over `[left, right]` inclusive.

```python
from persistent_segment_tree import PersistentSegmentTree

tree = PersistentSegmentTree([1, 2, 3, 4, 5])
assert tree.query(0, 0, 4) == 15

v1 = tree.update(0, 2, 100)          # returns new version id (1)
assert tree.query(0, 0, 4) == 15      # version 0 unchanged
assert tree.query(v1, 0, 4) == 112    # version 1 reflects the change
```

## Why

Range-sum queries over historical snapshots of an array come up in audit logs, time-series analytics, and offline algorithm problems where you need to answer "what was the sum over `[l, r]` at the moment just before update `k`?". A naive approach — copying the array on every update — costs O(n) memory per snapshot. A persistent segment tree gives the same query interface at O(log n) per update and per query, adding only O(log n) nodes per update instead of O(n).

The trade-off: the combine operation is fixed to summation. If you need min/max/gcd or a custom monoid, this library is not it. Sum is the one operation worth the restriction because it is by far the most common ask and keeping the API tiny means the behaviour is easy to reason about.

## Edge cases worth knowing

- The constructor accepts an empty list. `size` will be 0 and `version_count()` will be 1 (version 0 exists but holds no elements). Any `query` or `update` on an empty tree raises `ValueError`, because there is no valid index.
- Version numbers are integers assigned sequentially: 0 is the initial array, and each `update` returns the next integer. You may branch — two updates from the same parent produce two independent versions.
- `left` and `right` are inclusive. `query(0, 2, 1)` is an error, not an empty-sum shortcut.
