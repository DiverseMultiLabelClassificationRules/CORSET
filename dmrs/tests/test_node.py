import pytest
from dmrs.trie.node import Node, NodeV2

@pytest.mark.parametrize('NodeCls', [Node, NodeV2])
class TestNode:
    def test_eq(self, NodeCls):
        n1 = NodeCls(1, children=[NodeCls(2), NodeCls(3, children=[NodeCls(4)])])
        n2 = NodeCls(1, children=[NodeCls(2), NodeCls(3, children=[NodeCls(4)])])
        n3 = NodeCls(1, children=[NodeCls(2), NodeCls(3, children=[NodeCls(4)], is_last=True)])
        n4 = NodeCls(1, children=[NodeCls(2), NodeCls(3, children=[NodeCls(5)])])

        assert n1 == n2
        assert n1 != n3
        assert n1 != n4

    def test_insert(self, NodeCls):
        n = NodeCls(None)
        c2 = n.add_child(2)

        c4 = n.add_child(4)
        c3 = n.add_child(3)

        assert n.parent is None
        for c in (c2, c3, c4):
            assert c.parent == n

        with pytest.raises(ValueError):
            c2.add_child(0)

        with pytest.raises(ValueError):
            c2.add_child(2)

        assert c3.item == 3
        assert c4.item == 4

        assert n.children_items == [2, 3, 4]

    @pytest.fixture
    def a_node(self, NodeCls):
        n = NodeCls(1)
        for i in [2, 3, 4]:
            n.add_child(i)
        return n

    def test_has_child(self, a_node):
        for i in [2, 3, 4]:
            assert a_node.has_child(i)
        for i in [1, 0, 5, 6]:
            assert not a_node.has_child(i)

    def test_get_child(self, NodeCls):
        n = NodeCls(1)
        c2 = n.add_child(2)
        c3 = n.add_child(3)
        # c2 = NodeCls(2)
        # c3 = NodeCls(3)
        # n.children.append(c2)
        # n.children.append(c3)

        assert n.get_child(2) == c2
        assert n.get_child(3) == c3
        with pytest.raises(KeyError):
            n.get_child(4)

    def test_as_nested_tuple(self, NodeCls):
        root = NodeCls(0)
        n1 = root.add_child(1)
        root.add_child(3)
        n2 = n1.add_child(2)
        n2.add_child(4)

        actual = root.as_nested_tuple()
        expected = (0, (1, (2, (4, tuple()))), (3, tuple()))

        assert actual == expected

    @pytest.fixture
    def case_1_nodes(self, NodeCls):
        root = NodeCls(None)
        n3 = root.add_child(3, is_last=True)

        n1 = root.add_child(1)
        n2 = n1.add_child(2, is_last=True)
        n4 = n2.add_child(4, is_last=True)
        return root, n1, n2, n3, n4

    def test_get_set(self, case_1_nodes):
        root, n1, n2, n3, n4 = case_1_nodes
        if hasattr(root, 'populate_sets'):
            root.populate_sets()
        assert root.get_set() == None
        assert n1.get_set() == None
        assert n2.get_set() == (1, 2)
        assert n4.get_set() == (1, 2, 4)
        assert n3.get_set() == (3,)

    def test_all_supersets(self, case_1_nodes):
            
        root, n1, n2, n3, n4 = case_1_nodes
        if hasattr(root, 'populate_sets'):
            root.populate_sets()
            
        actual = root.all_supersets()
        expected = {(3, ), (1, 2), (1, 2, 4)}
        assert actual == expected
        
        actual = n1.all_supersets()
        expected = {(1, 2), (1, 2, 4)}
        assert actual == expected
        
        actual = n2.all_supersets()
        expected = {(1, 2), (1, 2, 4)}
        assert actual == expected
        
        actual = n4.all_supersets()
        expected = {(1, 2, 4)}
        assert actual == expected

