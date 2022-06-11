from .node import NodeV2


def _trie_insert(node, a_list, pos):
    if pos < len(a_list):
        item = a_list[pos]
        if not node.has_child(item):
            # create this node since the item does not exist
            child_node = node.add_child(item)
        else:
            child_node = node.get_child(item)
        _trie_insert(child_node, a_list, pos + 1)

        if pos == (len(a_list) - 1):
            child_node.is_last = True


def _trie_search(node, query, pos):
    try:
        child_node = node.get_child(query[pos])
        if pos == (len(query) - 1):
            return child_node.is_last
        else:
            return _trie_search(child_node, query, pos + 1)
    except KeyError:
        # print(f"cannot find child {query[pos]}")
        pass
    return False


def _trie_search_supersets(node: NodeV2, query: list, pos: int, result: set):
    """
    find all supersets of a query set

    pos: current position in query
    """
    item = query[pos]
    for child_node in node.children:
        child_item = child_node.item
        if child_item <= item:
            if child_item == item:  # a match
                if pos == (len(query) - 1):  # query can be matched
                    # add all descendants
                    result |= child_node.all_supersets()
                else:
                    # continue the search
                    _trie_search_supersets(child_node, query, pos + 1, result)
                    # break the loop because 'item' cannot appear in the subtrees of remaining children
                    break
            else:
                _trie_search_supersets(child_node, query, pos, result)

# @profile
def _trie_search_subsets(node: NodeV2, query: list, pos: int, result: set):
    """
    find all subsets of a query set

    pos: current position of query
    """
    if pos < len(query):
        item = query[pos]
        try:
            child_node = node.get_child(item)
            if child_node.is_last:
                to_add = child_node.get_set()
                result.add(to_add)
            _trie_search_subsets(child_node, query, pos+1, result)
        except KeyError:
            pass
        _trie_search_subsets(node, query, pos+1, result)


class SetTrie:
    def __init__(self):
        self._root = NodeV2(item=None, children=[], is_last=False)

    def insert(self, a_set: set):
        a_list = list(sorted(a_set))
        _trie_insert(self._root, a_list, 0)

    def insert_batch(self, sets):
        for a_set in sets:
            self.insert(a_set)
        if hasattr(self._root, 'populate_sets'):
            self._root.populate_sets()

    def search(self, query: set):
        """exact match"""
        if len(query) == 0:
            return False
        return _trie_search(self._root, list(sorted(query)), 0)

    def _check_query(self, query):
        if len(query) == 0:
            raise ValueError("len(query) should be positive, but is zero")

    def search_supersets(self, query: set):
        self._check_query(query)
        result = set()
        _trie_search_supersets(self._root, list(sorted(query)), 0, result)
        return result

    def search_subsets(self, query: set):
        result = set()
        if len(query) > 0:
            _trie_search_subsets(self._root, list(sorted(query)), 0, result)
        return result
