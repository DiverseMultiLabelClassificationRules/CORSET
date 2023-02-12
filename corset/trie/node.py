import bisect
from .utils import binary_search


class Node:
    def __init__(self, item, children=None, is_last=False, parent=None):
        self.item = item
        self.children = children if children is not None else []
        self.children_items = [c.item for c in children] if children is not None else []
        self.is_last = is_last
        self.parent = parent

    @property
    def is_leaf(self):
        return self.num_children == 0

    @property
    def num_children(self):
        return len(self.children)

    def has_child(self, item):
        items = [c.item for c in self.children]
        pos = binary_search(items, item)
        return pos != -1

    def get_child(self, item):
        items = [c.item for c in self.children]
        pos = binary_search(items, item)
        if pos != -1:
            return self.children[pos]
        else:
            raise KeyError(f'child "{item}" does not exist')

    def add_child(self, item, is_last=False):
        if self.item is not None and item <= self.item:
            raise ValueError(
                f"item to insert is smaller or equal to self.item: {item} <= {self.item} "
            )

        if self.has_child(item):
            raise ValueError(f"{item} is already present!")
        else:
            child_node = Node(item, is_last=is_last, parent=self)

            pos = bisect.bisect_left(
                self.children_items, item
            )  # determine where to insert item.

            self.children_items.insert(pos, item)  # insert key of item to keys list.
            self.children.insert(
                pos, child_node
            )  # insert the item itself in the corresponding place.

            return child_node

    def get_set(self):
        if not self.is_last:
            return None
        else:

            def _recurse_up(node):
                if node.parent is None:
                    return tuple()
                else:
                    return _recurse_up(node.parent) + (node.item,)

            return _recurse_up(self)

    def all_supersets(self):
        """
        return all sets in the subtree of this node
        """

        def _recurse_down(node, result: set):
            if node.is_last:
                result.add(node.get_set())
            for c in node.children:
                _recurse_down(c, result)

        result = set()
        _recurse_down(self, result)
        return result

    def __repr__(self):
        return f"Node({self.item}) with {self.num_children} children"

    def __eq__(self, other):
        if (self.item != other.item) or (self.is_last != other.is_last):
            return False

        if len(self.children) != len(other.children):
            return False
        else:
            for ci, cj in zip(self.children, other.children):
                if ci != cj:
                    return False
        return True

    def as_nested_tuple(self):
        ret = list([self.item])
        if self.is_leaf:
            return (self.item, tuple())
        for child_node in self.children:
            ret.append(child_node.as_nested_tuple())
        return tuple(ret)



class NodeV2(Node):
    """faster version of the class Node, main changes:

    - faster has_child and get_child using dict to store node internally
    - faster get_set by storing the set for each node during preprocessing
    """

    def __init__(self, item, children=None, is_last=False, parent=None):
        self.item = item
        self.children = (children if children is not None else [])
        self.children_items = [c.item for c in children] if children is not None else []
        self.is_last = is_last
        self.parent = parent

        self.item2child = {c.item: c for c in self.children}
        self._set = None

    @property
    def num_children(self):
        return len(self.item2child)
    
    def has_child(self, item):
        return item in self.item2child

    # @profile
    def get_child(self, item):
        return self.item2child[item]

    def add_child(self, item, is_last=False):
        if self.item is not None and item <= self.item:
            raise ValueError(
                f"item to insert is smaller or equal to self.item: {item} <= {self.item} "
            )

        if self.has_child(item):
            raise ValueError(f"{item} is already present!")
        else:
            child_node = NodeV2(item, is_last=is_last, parent=self)

            self.item2child[item] = child_node

            # do we really need to maintain the children array?
            pos = bisect.bisect_left(
                self.children_items, item
            )  # determine where to insert item.

            self.children_items.insert(pos, item)  # insert key of item to keys list.
            self.children.insert(
                pos, child_node
            )  # insert the item itself in the corresponding place.

            return child_node

    # @profile
    def populate_sets(self):
        """pre-compute sets for each node in the tree, maybe called only once after the tree is built"""
        print('populating sets')
        # @profile
        def aux(n):
            if n._set is None:
                n._set = super(NodeV2, n).get_set()
            
            if not n.is_leaf:
                for c in n.children:
                    aux(c)
        aux(self)

    # @profile
    def get_set(self):
        return self._set

