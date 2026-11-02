import unittest

import numpy as np

from delay_stationary import joint_transition_operator, stationary_queue
from sparse_phase_queue import sparse_phase_transition_operator


class SparsePhaseQueueTest(unittest.TestCase):
    def test_sparse_factorization_matches_independent_dense_convolution(self):
        states = np.array([0., 0., 1., 1.])
        capacities = np.array([0, 2, 1, 2])
        transition = np.array([[.2, .3, .3, .2], [.1, .2, .2, .5],
                               [.2, .2, .3, .3], [.4, .1, .1, .4]])
        rng = np.random.default_rng(12)
        joint = rng.random((31, 4))
        joint /= joint.sum()
        dense = joint_transition_operator(states, transition, 1., .7, capacities, 30)(joint)
        sparse = sparse_phase_transition_operator(states, transition, 1., .7, capacities, 30)(joint)
        np.testing.assert_allclose(sparse, dense, atol=3e-14)
        expected = stationary_queue(states, transition, 1., .7, capacities, 80,
                                    tolerance=1e-12)
        actual = stationary_queue(states, transition, 1., .7, capacities, 80,
                                  tolerance=1e-12, operator_factory=sparse_phase_transition_operator)
        for key in ['mean_residual', 'mean_queue', 'variance_queue', 'mean_wait', 'variance_wait']:
            self.assertAlmostEqual(actual[key], expected[key], places=10)


if __name__ == '__main__':
    unittest.main()
