# distutils: language = c++
cimport numpy as np
import networkx as nx
from time import time
from libcpp cimport bool
from libcpp.unordered_set cimport unordered_set
from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector
from libc.stdlib cimport malloc, free
from tqdm import tqdm

ctypedef unordered_map[int, float] ifmap
ctypedef unordered_set[int] nbrset

cdef class Graph:
    """a wrapper for networkx.Graph"""
    cdef int N
    cdef nbrset **nbr_map  # mapping a node to its neighbour set
    cdef ifmap **edge_weight_map  # mapping a source and a target to the edge weight

    def __cinit__(self,
                  int num_nodes,
                  np.ndarray[double, ndim=1] data,
                  np.ndarray[int, ndim=1] indices,
                  np.ndarray[int, ndim=1] indptr):

        self.N = num_nodes
        self.nbr_map = <nbrset**> malloc(
            self.N * sizeof(nbrset*)
        )
        self.edge_weight_map = <ifmap**> malloc(
            self.N * sizeof(ifmap*)
        )
        
        cdef int u;
        for u in range(self.N):
            self.nbr_map[u] = new nbrset()
            self.edge_weight_map[u] = new ifmap()
            for v, weight in zip(indices[indptr[u]:indptr[u+1]],
                                 data[indptr[u]:indptr[u+1]]):
                self.nbr_map[u][0].insert(v)
                self.edge_weight_map[u][0][v] = weight  # the [0] is used to deference self.edge_weight_map[u], which is a pointer

    cdef double _edge_weight(self, int u, int v):
        return self.edge_weight_map[u][0][v]

    def edge_weight(self, int u, int v):
        return self._edge_weight(u, v)

    cdef nbrset* _neighbors_ptr(self, int u):
        """return a pointer to the neighbor set"""
        return self.nbr_map[u]

    def neighbors(self, int u):
        """return the neighbours of u, as a python set"""
        return self._neighbors_ptr(u)[0]
    
    def __dealloc__(self):
        cdef int u
        for u in range(self.N):
            free(self.nbr_map[u])
            free(self.edge_weight_map[u])
        free(self.nbr_map)
        free(self.edge_weight_map)

    
cdef void intersection(nbrset &a, nbrset &b, nbrset &out):
    """
    do set intersection between `a` and `b`, output to `out`
    """
    if not out.empty():
        out.clear()
        
    cdef int el;
    for el in a:
        if b.count(el) > 0:
            out.insert(el)
    
        
def dfs_v2(g, float alpha, pc_list, proba_key, bool return_proba):
    """
    new data structure:
    - nbrs: map<int, set<int>> or an array of set<int>
    - edge proba dict: map<int, map<int, float>> or an array of map<int, float>
    """
    s = time()
    cdef vector[int] S;
    cdef nbrset shared_nbrs;
    mat = nx.to_scipy_sparse_matrix(g, weight=proba_key)

    g_wrapper = Graph(mat.shape[0], mat.data, mat.indices, mat.indptr)
    dfs_v2_aux(S, g_wrapper,
               1.0, shared_nbrs, alpha, pc_list, return_proba)
    print("elapsed in dfs_v2_aux: {}".format(time() - s))

    return pc_list
        
cpdef dfs_v2_aux(vector[int] &S,
                 Graph g_wrapper,
                 float proba,
                 nbrset &shared_nbrs,
                 float alpha,
                 pc_list,
                 bool return_proba):
    if return_proba:
        pc_list.append((tuple(S), proba))
    else:
        pc_list.append(tuple(S))
        
    cdef float new_proba
    cdef int u, v
    cdef nbrset new_shared_nbrs
    
    if S.empty():
        # 1-clique has probability 1.0 by default
        for u in tqdm(range(g_wrapper.N)):
            S.push_back(u)
            dfs_v2_aux(S, g_wrapper,
                        1.0, g_wrapper._neighbors_ptr(u)[0],
                        alpha, pc_list, return_proba)
            S.pop_back()
    else:
        for u in shared_nbrs:
            if u > S.back():
                # incrementally determine if go down the recursion
                success = True
                new_proba = proba
                for v in S:
                    # print('u, v:', u, v)
                    # print("g._adj[u]: ", g._adj[u])
                    new_proba *= g_wrapper._edge_weight(u, v)
                    if new_proba < alpha:
                        success = False
                        break                    
                if success:
                    intersection(shared_nbrs, g_wrapper._neighbors_ptr(u)[0], new_shared_nbrs)

                    S.push_back(u)
                    dfs_v2_aux(S, g_wrapper,
                                new_proba, new_shared_nbrs, alpha, pc_list, return_proba)
                    S.pop_back()
