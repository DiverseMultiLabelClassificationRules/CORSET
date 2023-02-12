import pytest
from corset.trie import NodeV2, _trie_insert, _trie_search, _trie_search_subsets, _trie_search_supersets, SetTrie


class TestSetTrie:
    def get_supersets(self, query, records):
        """helper method to get all supersets of query"""
        query = set(query)
        return {tuple(sorted(r)) for r in records if query.issubset(r)}

    def get_subsets(self, query, records):
        """helper method to get all subsets of query"""
        query = set(query)
        return {tuple(sorted(r)) for r in records if query.issuperset(r)}

    def test__trie_search_simple(self):
        n = NodeV2(
            None, children=[NodeV2(1, children=[NodeV2(2, is_last=True)], is_last=True)]
        )
        assert _trie_search(n, [1], 0)
        assert _trie_search(n, [1, 2], 0)

    def test__trie_search_subsets_simple(self):
        trie = SetTrie()
        trie.insert_batch(
            [
                {1, 2, 3},
                {1, 4, 5},
                {
                    4,
                },
                {
                    5,
                },
            ]
        )
        
        root = trie._root

        result = set()
        _trie_search_subsets(root, (1, 2, 3, 4, 5), pos=0, result=result)

        expected = {(1, 2, 3), (1, 4, 5), (4,), (5,)}
        assert result == expected

        trie = SetTrie()
        trie.insert_batch(
            [
                {
                    1,
                },
                {
                    4,
                },
                {
                    5,
                },
            ]
        )
        root = trie._root

        result = set()
        _trie_search_subsets(root, (1, 2, 3, 4, 5), pos=0, result=result)

        expected = {(1,), (4,), (5,)}
        assert result == expected

    @pytest.fixture
    def case_1_input(self):
        return [{1, 2, 3}, {1, 4, 5}]

    @pytest.fixture
    def case_1_trie(self, case_1_input):
        trie = SetTrie()
        trie.insert_batch(case_1_input)
        return trie

    @pytest.fixture
    def case_1_root(self):
        return NodeV2(
            None,
            children=[
                NodeV2(
                    1,
                    children=[
                        NodeV2(2, children=[NodeV2(3, is_last=True)]),
                        NodeV2(4, children=[NodeV2(5, is_last=True)]),
                    ],
                )
            ],
        )

    @pytest.fixture
    def case_1_nested_tuples(self):
        return (None, (1, (2, (3, tuple())), (4, (5, tuple()))))

    def test_case1__trie_insert(self, case_1_input, case_1_nested_tuples):
        root = NodeV2(None)
        _trie_insert(root, list(case_1_input[0]), 0)
        actual = root.as_nested_tuple()
        expected = (None, (1, (2, (3, tuple()))))

        assert actual == expected

        _trie_insert(root, list(case_1_input[1]), 0)

        expected = case_1_nested_tuples

    def test_case_1insert(self, case_1_input, case_1_nested_tuples, case_1_root):
        trie = SetTrie()
        trie.insert(case_1_input[0])
        trie.insert(case_1_input[1])

        actual = trie._root.as_nested_tuple()
        expected = case_1_nested_tuples

        assert actual == expected

        actual = trie._root
        expected = case_1_root
        assert actual == expected

    def test_case_1_search(self, case_1_trie):
        successful_queries = [{1, 2, 3}, {1, 4, 5}]
        failed_queries = [{1, 2}, {2}, {2, 3}, {4, 5}, {5}, {}]

        for query in successful_queries:
            assert case_1_trie.search(query)

        for query in failed_queries:
            assert not case_1_trie.search(query)

    def test_case_1_superset_search(self, case_1_trie, case_1_input):
        queries = [{1}, {4}, {1, 2}, {1, 2, 5}, {33}, {1, 10}, {1, 2, 5}]
        for query in queries:
            actual = case_1_trie.search_supersets(query)
            expected = self.get_supersets(query, case_1_input)

            assert actual == expected

        # empty query is not allowed
        query = set()
        with pytest.raises(ValueError):
            actual = case_1_trie.search_supersets(query)

    def test_case_1_subset_search(self, case_1_trie, case_1_input):
        queries = [
            {1},
            {4},
            {1, 2},
            {1, 2, 5},
            {33},
            {1, 10},
            {1, 2, 5},
            set(range(100)),
        ]
        for query in queries:
            actual = case_1_trie.search_subsets(query)
            expected = self.get_subsets(query, case_1_input)

            assert actual == expected

        # empty query is not allowed
        query = set()
        actual = case_1_trie.search_subsets(query)
        expected = set()
        assert actual == expected

    @pytest.fixture
    def case_2_input(self):
        return [{1, 2, 3, 5}, {1, 2, 5}, {1, 2, 5, 6}, {4}, {4, 8, 9}, {11, 15, 22}]

    @pytest.fixture
    def case_2_root(self):
        return NodeV2(
            item=None,
            children=[
                NodeV2(
                    1,
                    children=[
                        NodeV2(
                            2,
                            children=[
                                NodeV2(3, children=[NodeV2(5, is_last=True)]),
                                NodeV2(5, children=[NodeV2(6, is_last=True)], is_last=True),
                            ],
                        )
                    ],
                ),
                NodeV2(
                    4,
                    children=[NodeV2(8, children=[NodeV2(9, is_last=True)])],
                    is_last=True,
                ),
                NodeV2(11, children=[NodeV2(15, children=[NodeV2(22, is_last=True)])]),
            ],
        )

    @pytest.fixture
    def case_2_trie(self, case_2_input):
        trie = SetTrie()
        trie.insert_batch(case_2_input)
        return trie

    def test_insert_2(self, case_2_input, case_2_root):
        trie = SetTrie()
        for a_set in case_2_input:
            trie.insert(a_set)
        assert trie._root == case_2_root

    def test_search_2(self, case_2_trie):
        successful_queries = [
            {1, 2, 3, 5},
            {1, 2, 5},
            {1, 2, 5, 6},
            {4},
            {4, 8, 9},
            {11, 15, 22},
        ]
        failed_queries = [{1, 2}, {1, 5}, {1, 3}, {4, 8}, {23}, {11}, {6}, {1}]

        for query in successful_queries:
            assert case_2_trie.search(query)

        for query in failed_queries:
            assert not case_2_trie.search(query)

    def test_case_2_superset_search(self, case_2_trie, case_2_input):
        queries = [
            {1},
            {1, 2},
            {4},
            {4, 8},
            {11},
            {22},
            {3, 5},
            {6},
            {-1},
            {
                1000,
            },
            {1, 18},  # queries with no results
        ]
        for query in queries:
            actual = case_2_trie.search_supersets(query)
            expected = self.get_supersets(query, case_2_input)

            assert actual == expected

    def test_case_2_subset_search(self, case_2_trie, case_2_input):
        queries = [
            {1},
            {1, 2},
            {4},
            {4, 8},
            {11},
            {22},
            {3, 5},
            {6},
            set(range(10)),
            {-1},
            {
                1000,
            },
            {1, 18},
        ]
        for query in queries:
            actual = case_2_trie.search_subsets(query)
            expected = self.get_subsets(query, case_2_input)

            assert actual == expected
