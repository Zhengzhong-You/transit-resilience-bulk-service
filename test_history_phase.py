import unittest

import numpy as np
from scipy.sparse import csr_matrix

from delay_stationary import stationary_queue
from history_phase import (backlog_classes, fit_history_phase, net_input_calibration,
                           net_input_chain, stationary_phase, upstream_backlog)
from sparse_phase_queue import fit_joint_phase, sparse_phase_transition_operator
from sparse_stationary import stationary_queue_sparse


class HistoryPhaseTest(unittest.TestCase):
    def test_backlog_counts_only_upstream_passengers_travelling_beyond_the_station(self):
        destinations = np.zeros((4, 4))
        destinations[0, 1:] = [.5, .25, .25]
        destinations[1, 2:] = [.5, .5]
        destinations[2, 3] = 1.
        residuals = np.array([[4., 2., 7., 0.]])
        self.assertAlmostEqual(upstream_backlog(residuals, destinations, 3)[0], 4 * .25 + 2 * .5)
        self.assertEqual(upstream_backlog(residuals, destinations, 1)[0], 0.)

    def test_backlog_classes_reserve_class_zero_for_an_empty_backlog(self):
        backlog = np.array([0., 0., 1., 2., 3., 4., 5., 6.])
        classes = backlog_classes(backlog, 4)
        self.assertTrue(np.all(classes[:2] == 0) and np.all(classes[2:] >= 1))
        self.assertEqual(sorted(set(classes[2:])), [1, 2, 3])
        np.testing.assert_array_equal(backlog_classes(backlog, 1), 0)

    def test_one_backlog_class_recovers_the_first_order_kernel(self):
        rng = np.random.default_rng(3)
        displacement = np.maximum(0, rng.normal(.5, 1, 4000))
        capacity = rng.integers(3, 6, 4000)
        delta, seats, transition, _ = fit_joint_phase(displacement, capacity, step=.4)
        fitted = fit_history_phase(displacement, capacity, np.zeros(4000), .4, 1)
        np.testing.assert_allclose(fitted['displacement'], delta)
        np.testing.assert_array_equal(fitted['capacity'], seats)
        np.testing.assert_allclose(fitted['transition'].toarray(), transition.toarray())
        self.assertEqual(fitted['removed_transition_fraction'], 0)

    def test_kernel_is_restricted_to_the_recurrent_class(self):
        # Phase 9 occurs once at the start and is never revisited.
        capacity = np.r_[9, np.tile([1, 2, 3], 50)]
        fitted = fit_history_phase(np.zeros(len(capacity)), capacity, np.zeros(len(capacity)), 1., 1)
        self.assertNotIn(9, fitted['capacity'])
        self.assertAlmostEqual(fitted['removed_transition_fraction'], 1 / (len(capacity) - 1))
        np.testing.assert_allclose(np.asarray(fitted['transition'].sum(axis=1)).ravel(), 1)

    def test_sparse_stationary_queue_matches_the_dense_calculation(self):
        states = np.array([0., 0., 1., 1.])
        capacities = np.array([0, 2, 1, 2])
        transition = np.array([[.2, .3, .3, .2], [.1, .2, .2, .5],
                               [.2, .2, .3, .3], [.4, .1, .1, .4]])
        dense = stationary_queue(states, transition, 1., .7, capacities, 80, tolerance=1e-12,
                                 operator_factory=sparse_phase_transition_operator)
        sparse = stationary_queue_sparse(states, csr_matrix(transition), 1., .7, capacities, 80,
                                         tolerance=1e-12)
        for key in ['mean_residual', 'mean_queue', 'variance_queue', 'mean_wait', 'variance_wait',
                    'mean_headway', 'residual_times_next_headway']:
            self.assertAlmostEqual(sparse[key], dense[key], places=11)

    def test_net_input_variance_of_the_chain_matches_a_long_sample_path(self):
        transition = csr_matrix(np.array([[.9, .1], [.2, .8]]))
        delta, capacity = np.array([0., .5]), np.array([3, 1])
        rng = np.random.default_rng(5)
        path = [0]
        for _ in range(400000):
            path.append(int(rng.random() < transition[path[-1], 1]))
        path = np.array(path)
        exact = net_input_chain(delta, capacity, transition, stationary_phase(transition), .8, 1., (1, 10, 100))
        sample = net_input_calibration(delta[path], capacity[path], .8, 1., (1, 10, 100))
        for m in (1, 10, 100):
            self.assertAlmostEqual(exact[m] / sample[m], 1, delta=.03)


if __name__ == '__main__':
    unittest.main()
