class Node: 
    def __init__(self, e):
        self.v_list = set()
        self.children = dict() 


class PTrie:
    def __init__(self, InvInd: list, n_rows: int):
        self._root = Node(e=None)
        self._root.v_list = set([x for x in range(n_rows)])
        self.InvInd = InvInd
        self.trie_new_rows = [[] for _ in range(n_rows)]


    #def get_children(current_node, a): 
    #    for child in current_node.children: 
    #        if child.v==a: 
    #            return child 
    #    return None 
        

    def ProcessRecord(self, r_id: int, a_set: set, ordering: list):
        ''' build trie on-the-fly while processing records''' 
        
        # we first sort the attributes by the pre-computed frequency ordering 
        a_list = [x for x in ordering if x in a_set] 
        current_node = self._root
        
        for a in a_list: 
           # child = self.get_children(current_node, a) # we can optimize this by using a dictionary instead of the linear search 
            if a in current_node.children: # in this case the current node has child already computed, we do not need to do intersection 
                #continue tree traversal 
                current_node = current_node.children[a] 
                
            else:
                #create a new node 
                new_node = Node(a)
                new_node.v_list = self.InvInd[a].intersection(current_node.v_list) 
                current_node.children[a] =  new_node 
                #self.nodes.add(new_node)
                #continue tree traversal 
                current_node = new_node 
                
        # update new rows 
        for row in current_node.v_list: 
            self.trie_new_rows[row].append(r_id)
