import pytest
import numpy as np

from scipy import sparse as sp

from dmrs.utils import (convert_matrix_to_sets, convert_matrix_to_sets_v2, convert_sets_to_matrix, csr_matrix_equal, conjunctive_collapse, conjunctive_collapse_v2)


def test_convert_matrix_to_sets():
    m = sp.csr_matrix(np.eye(3))
    actual = convert_matrix_to_sets(m)
    expected = [{0}, {1}, {2}]
    assert actual == expected

    actual = convert_matrix_to_sets_v2(m)
    assert actual == expected


def test_convert_sets_to_matrix():
    sets = [{0}, {1}, {2}]
    actual = convert_sets_to_matrix(sets, 3)
    expected = sp.csr_matrix(np.eye(3))
    assert csr_matrix_equal(actual, expected)

@pytest.mark.parametrize('func', [
    conjunctive_collapse, conjunctive_collapse_v2
])
def test_conjunctive_collapse(func):
    m = sp.csc_matrix(np.array([
        [1, 0, 1],
        [1, 1, 1],
        [1, 0, 1]]))
    cols = [0]
    expected = np.array([1, 1, 1])
    actual = func(m, cols)
    np.testing.assert_allclose(actual, expected)

    cols = [0, 2]
    expected = np.array([1, 1, 1])
    actual = func(m, cols)
    np.testing.assert_allclose(actual, expected)

    cols = [1]
    expected = np.array([0, 1, 0])
    actual = func(m, cols)
    np.testing.assert_allclose(actual, expected)
    
    cols = [1, 2]
    expected = np.array([0, 1, 0])
    actual = func(m, cols)
    np.testing.assert_allclose(actual, expected)
    
    
def test_conjunctive_collapse_v2_patch():
    """"""
    m = sp.csc_matrix(np.array([
        [1, 1],
        [0, 1],
        [1, 0]
    ]))
    cols = [0, 1]
    expected = np.array([1, 0, 0])
    actual = conjunctive_collapse_v2(m, cols)
    np.testing.assert_allclose(actual, expected)
