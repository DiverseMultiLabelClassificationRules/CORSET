from dmrs.PRETTI.PRETTI_Trie import PTrie
from dmrs.PRETTI.PRETTI_invertedIndex import InvertedIndex
import numpy as np 
from dmrs.samplers.assignment import PRETTISampleAssignmentMixin
from scipy import sparse as sp

def test_inverted_index(): 
    Y = np.array([ [0,1,0], [1,1,0], [0,0,1]  ])
    expected=[set([1]), set([0,1]), set([2])]
    actual=InvertedIndex(Y).build_index()
    assert actual == expected
    Y = sp.csr_matrix(
        np.array([
            [1, 1, 1, 1, 1, 1],  # 1-5
            [0, 1, 0, 0, 1, 1],  # 1, 4, 5
            [0, 1, 0, 0, 0, 1]   # 1, 5
        ]))
    expected=[{0}, {0,1,2}, {0}, {0},{0, 1}, {0, 1, 2}]
    actual=InvertedIndex(Y).build_index()
    assert actual == expected
    

def test_ordering(): 
    obj = PRETTISampleAssignmentMixin() 
    obj._sample_space = [{1, 4, 5}, {1, 2, 3}, {4,}, {4, }, {5, }]
    actual=obj.find_ordering()
    expected=[4, 1, 5, 2, 3]
    assert actual==expected
    
    
def test_trie(): 
    obj = PRETTISampleAssignmentMixin() 
    obj._sample_space = [{1, 4, 5}, {1, 2, 3}, {4,1}, {4, }, {4, }, {5, }]
    
    Y = sp.csr_matrix(
        np.array([
            [1, 1, 1, 1, 1, 1],  # 1-5
            [0, 1, 0, 0, 1, 1],  # 1, 4, 5
            [0, 1, 0, 0, 0, 1]   # 1, 5
        ]))
    
    ordering=obj.find_ordering()
    InvInd = InvertedIndex(Y).build_index()
    Trie = PTrie(InvInd, Y.shape[0])  # initialize
    # build trie on the fly
    for sample_id, sample in enumerate(obj._sample_space):
        Trie.ProcessRecord(sample_id, sample, ordering)
        
    actual = Trie._root.v_list
    expected =  {0,1,2}
    assert actual==expected
    
    actual = Trie._root.children[4].v_list
    expected = {0,1}
    assert actual==expected
    
    
    actual = Trie._root.children[4].children[1].v_list
    expected = {0,1}
    assert actual==expected
    
    
