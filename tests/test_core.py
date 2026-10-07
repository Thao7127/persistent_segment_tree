import unittest

from persistent_segment_tree import PersistentSegmentTree


class TestBuildAndQuery(unittest.TestCase):
    def test_basic_sum(self):
        tree = PersistentSegmentTree([1, 2, 3, 4, 5])
        self.assertEqual(tree.query(0, 0, 4), 15)
        self.assertEqual(tree.query(0, 1, 3), 9)

    def test_single_element(self):
        tree = PersistentSegmentTree([42])
        self.assertEqual(tree.query(0, 0, 0), 42)

    def test_full_range(self):
        tree = PersistentSegmentTree([10, -3, 7, 0, 2])
        self.assertEqual(tree.query(0, 0, 4), 16)


class TestUpdate(unittest.TestCase):
    def test_update_creates_new_version(self):
        tree = PersistentSegmentTree([1, 2, 3, 4, 5])
        v1 = tree.update(0, 2, 100)
        self.assertEqual(v1, 1)
        self.assertEqual(tree.query(0, 0, 4), 15)
        self.assertEqual(tree.query(1, 0, 4), 112)
        self.assertEqual(tree.query(1, 2, 2), 100)

    def test_update_chain(self):
        tree = PersistentSegmentTree([0, 0, 0])
        v1 = tree.update(0, 0, 5)
        v2 = tree.update(v1, 1, 7)
        v3 = tree.update(v2, 2, 9)
        self.assertEqual([v1, v2, v3], [1, 2, 3])
        self.assertEqual(tree.query(v3, 0, 2), 21)
        self.assertEqual(tree.query(v2, 0, 2), 12)
        self.assertEqual(tree.query(v1, 0, 2), 5)
        self.assertEqual(tree.query(0, 0, 2), 0)

    def test_branching_versions(self):
        tree = PersistentSegmentTree([0, 0, 0])
        a = tree.update(0, 0, 1)
        b = tree.update(0, 0, 2)
        self.assertEqual(tree.query(a, 0, 0), 1)
        self.assertEqual(tree.query(b, 0, 0), 2)

    def test_update_same_index_twice(self):
        tree = PersistentSegmentTree([1, 1, 1])
        v1 = tree.update(0, 1, 50)
        v2 = tree.update(v1, 1, 60)
        self.assertEqual(tree.query(v1, 1, 1), 50)
        self.assertEqual(tree.query(v2, 1, 1), 60)


class TestVersionCount(unittest.TestCase):
    def test_initial_count(self):
        tree = PersistentSegmentTree([1, 2, 3])
        self.assertEqual(tree.version_count(), 1)

    def test_grows_with_updates(self):
        tree = PersistentSegmentTree([1, 2, 3])
        tree.update(0, 0, 9)
        self.assertEqual(tree.version_count(), 2)


class TestEmpty(unittest.TestCase):
    def test_empty_array_builds(self):
        tree = PersistentSegmentTree([])
        self.assertEqual(tree.size, 0)
        self.assertEqual(tree.version_count(), 1)


class TestErrors(unittest.TestCase):
    def test_query_bad_version(self):
        tree = PersistentSegmentTree([1, 2, 3])
        with self.assertRaises(ValueError):
            tree.query(1, 0, 2)

    def test_query_bad_range(self):
        tree = PersistentSegmentTree([1, 2, 3])
        with self.assertRaises(ValueError):
            tree.query(0, -1, 2)
        with self.assertRaises(ValueError):
            tree.query(0, 0, 3)
        with self.assertRaises(ValueError):
            tree.query(0, 2, 1)

    def test_update_bad_index(self):
        tree = PersistentSegmentTree([1, 2, 3])
        with self.assertRaises(ValueError):
            tree.update(0, -1, 9)
        with self.assertRaises(ValueError):
            tree.update(0, 3, 9)

    def test_update_bad_version(self):
        tree = PersistentSegmentTree([1, 2, 3])
        with self.assertRaises(ValueError):
            tree.update(99, 0, 9)

    def test_type_errors(self):
        tree = PersistentSegmentTree([1, 2, 3])
        with self.assertRaises(TypeError):
            PersistentSegmentTree("not a list")
        with self.assertRaises(TypeError):
            PersistentSegmentTree([1, 2, "x"])
        with self.assertRaises(TypeError):
            tree.query(0, 0, 2.0)
        with self.assertRaises(TypeError):
            tree.update(0, 0, True)


if __name__ == "__main__":
    unittest.main()
