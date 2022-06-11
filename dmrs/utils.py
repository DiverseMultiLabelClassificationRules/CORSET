import json
import os
import pickle as pkl
import tempfile
import numpy as np
import mlflow

from os.path import join
from scipy import sparse as sp


pjoin = join


def flatten(stuff):
    """flatten an array"""
    return np.asarray(stuff).flatten()


def support_size(Y: sp.csc_matrix, ids):
    """return number of rows that have all columns specified by ids > 0, assuming Y is a 0/1 matrix"""
    matches_flag = flatten(Y[:, list(ids)].sum(axis=1)) == len(ids)
    return matches_flag.sum()


def get_tempdir(prefix=None, suffix=None, dir=None):
    if dir is not None:
        makedir(dir, usedir=False)
    return tempfile.mkdtemp(prefix=prefix, suffix=suffix, dir=dir)


def makedir(d, usedir=True):
    if usedir:
        d = os.path.dirname(d)

    if not os.path.exists(d):
        os.makedirs(d)


def save_pickle(obj, path):
    return pkl.dump(obj, open(path, 'wb'))

def save_file(string, path):
    with open(path, 'w') as f:
        f.write(string)
        
def load_pickle(path):
    return pkl.load(open(path, 'rb'))


def save_json(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=4)


def _ensure_is_csrmatrix(m):
    if not isinstance(m, sp.csr_matrix):
        m = m.tocsr()
    return m


def convert_matrix_to_sets(m):
    """given a binary sparse matrix, return a list of sets, each of which is the non-zero indices of the ith row"""
    m = _ensure_is_csrmatrix(m)
    return [set(m[i, :].nonzero()[1]) for i in range(m.shape[0])]


def convert_matrix_to_sets_v2(m):
    """
    improved version of `convert_matrix_to_sets` using `indices` and `indptr`
    """
    m = _ensure_is_csrmatrix(m)
    indices, indptr = m.indices, m.indptr
    ret = []
    for i in range(m.shape[0]):
        ret.append(set(indices[indptr[i]:indptr[i+1]]))
    return ret


def filter_rows_with_no_labels(X: sp.csr_matrix, Y: sp.csr_matrix):
    mask = (flatten(Y.sum(axis=1)) > 0)
    return X[mask], Y[mask]


# @jit(nopython=True)
def array_product(arr):
    res = 1.
    for e in arr:
        res *= e
    return res


def counter2proba(counter):
    total = sum(counter.values())
    proba = {}
    for k, freq in counter.items():
        proba[k] = freq / total
    return proba

# @profile
def conjunctive_collapse(matrix: sp.csc_matrix, cols: tuple):
    """select columns of mat indicated by cols
    evaluate each row conjunctively
    """
    # TODO: simply iterate and add up the columns
    m_sub = matrix[:, cols]
    m_sub_csr = m_sub.tocsr()
    return flatten(m_sub_csr.sum(axis=1)) == len(cols)


def conjunctive_collapse_v2(matrix: sp.csc_matrix, cols: tuple):
    """implementation of conjunctive_collapse using set operation
    """
    assert isinstance(matrix, sp.csc_matrix)

    assert len(cols) > 0
    indices, indptr = matrix.indices, matrix.indptr

    shared_rows = []

    list_of_sets = [set(indices[indptr[i]:indptr[i+1]]) for i in cols]
    list_of_sets = list(sorted(list_of_sets, key=len))
    smallest_set = list_of_sets[0]

    for el in smallest_set:
        # check if el is in every set
        # if so, add it to shared_rows
        success = True
        for a_set in list_of_sets[1:]:
            if el not in a_set:
                # no need to add this element
                success = False
                break
        if success:
            shared_rows.append(el)

    ret = np.zeros(matrix.shape[0], dtype=bool)
    ret[shared_rows] = 1
    return ret


def binary_vector_to_set(vect):
    """extract indices of non-zero entries and put into a set"""
    return set((vect > 0).nonzero()[0])


def convert_sets_to_matrix(sets, L: int):
    """
    input: 
    sets: a list of sets, e.g., a list of label sets
    L: size of universe where the sets reside

    output: the corresponding sparse matrix in csr_matrix format
    """
    N = len(sets)
    m = sp.lil_matrix((N, L), dtype=bool)
    for i in range(N):
        for j in sets[i]:
            m[i, j] = 1
    return m.tocsr()

def csr_matrix_equal(m1, m2):
    """
    test whether two csr_matrix are equal
    """
    return (m1 != m2).nnz == 0


def create_experiment_if_needed(name):
    exp = mlflow.get_experiment_by_name(name)
    if exp is None:
        exp_id = mlflow.create_experiment(name)
        exp = mlflow.get_experiment_by_name(exp_id)
    return exp


def get_experiment_id_by_name(name):
    exp = create_experiment_if_needed(name)
    return exp.experiment_id
