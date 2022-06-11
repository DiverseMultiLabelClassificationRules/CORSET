import pytest
from scipy import sparse as sp
import numpy as np
from dmrs.samplers.assignment import TrieSampleAssignmentMixin, NaiveSampleAssignmentMixin, PRETTISampleAssignmentMixin

@pytest.mark.parametrize('cls', [NaiveSampleAssignmentMixin, TrieSampleAssignmentMixin, PRETTISampleAssignmentMixin])
def test_sample_assignment_toy(cls):
    obj = cls()
    # mockup
    obj._sample_space = [{1, 4, 5}, {1, 2, 3}, {4, }, {5, }]

    Y = sp.csr_matrix(
        np.array([
            [1, 1, 1, 1, 1, 1],  # 1-5
            [0, 1, 0, 0, 1, 1],  # 1, 4, 5
            [0, 1, 0, 0, 0, 1]   # 1, 5
        ]))

    obj._generate_new_rows(Y)

    actual = list(map(set, obj.new_rows))
    expected = [
        {1, 0, 2, 3},
        {0, 2, 3},
        {3}
    ]
    assert actual == expected
