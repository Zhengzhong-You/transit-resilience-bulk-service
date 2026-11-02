import unittest

import numpy as np
from scipy.stats import poisson

from delay_stationary import joint_transition_operator, stationary_queue


class DelayStationaryTest(unittest.TestCase):
    def test_fft_operator_matches_the_full_transition_probabilities(self):
        states = np.array([0., 1.])
        transition = np.array([[0.6, 0.4], [0.3, 0.7]])
        capacity, cutoff, arrival_rate = 1, 7, 0.5
        dense = np.zeros(((cutoff + 1) * 2, (cutoff + 1) * 2))
        for r in range(cutoff + 1):
            for i in range(2):
                for j in range(2):
                    mean = arrival_rate * (1 + states[j] - states[i])
                    dense[2 * r + i, j] = transition[i, j] * poisson.cdf(capacity - r, mean)
                    for s in range(1, cutoff):
                        dense[2 * r + i, 2 * s + j] = (
                            transition[i, j] * poisson.pmf(s + capacity - r, mean)
                        )
                    dense[2 * r + i, 2 * cutoff + j] = (
                        transition[i, j] * poisson.sf(cutoff + capacity - r - 1, mean)
                    )
        np.testing.assert_allclose(dense.sum(axis=1), 1, atol=1e-14)
        joint = np.arange(1., (cutoff + 1) * 2 + 1).reshape(-1, 2)
        joint /= joint.sum()
        actual = joint_transition_operator(states, transition, 1., arrival_rate, capacity, cutoff)(joint)
        np.testing.assert_allclose(actual.ravel(), joint.ravel() @ dense, atol=2e-14)

    def test_single_phase_poisson_bulk_queue_has_known_moments(self):
        result = stationary_queue(np.array([0.]), np.array([[1.]]), 1., 0.5, 1, 40,
                                  tolerance=1e-13)
        self.assertAlmostEqual(result['mean_residual'], 0.25, places=9)
        self.assertAlmostEqual(result['mean_queue'], 0.75, places=9)
        self.assertAlmostEqual(result['variance_queue'], 43.0 / 48.0, places=8)
        self.assertAlmostEqual(result['mean_wait'], 1.0, places=9)
        self.assertAlmostEqual(result['variance_wait'], 2.0 / 3.0, places=8)

    def test_capacity_marks_use_the_destination_phase(self):
        states = np.array([0., 1.])
        transition = np.array([[0.6, 0.4], [0.3, 0.7]])
        capacities = np.array([0, 2])
        cutoff, rate = 9, 0.4
        joint = np.arange(1., 21.).reshape(10, 2)
        joint /= joint.sum()
        expected = np.zeros_like(joint)
        for r in range(cutoff + 1):
            for i in range(2):
                for j, c in enumerate(capacities):
                    mean = rate * (1 + states[j] - states[i])
                    weight = joint[r, i] * transition[i, j]
                    expected[0, j] += weight * poisson.cdf(c - r, mean)
                    for s in range(1, cutoff):
                        expected[s, j] += weight * poisson.pmf(s + c - r, mean)
                    expected[cutoff, j] += weight * poisson.sf(cutoff + c - r - 1, mean)
        actual = joint_transition_operator(states, transition, 1., rate, capacities, cutoff)(joint)
        np.testing.assert_allclose(actual, expected, atol=3e-14)
        solution = stationary_queue(states, transition, 1., rate, capacities, 100,
                                    tolerance=1e-12)
        self.assertAlmostEqual(solution['mean_available_capacity'], 8 / 7, places=10)
        self.assertLess(solution['upper_boundary_probability'], 1e-12)

    def test_edge_capacity_kernel_matches_independent_dense_operator(self):
        states = np.array([0., 1.])
        transition = np.array([[0.6, 0.4], [0.3, 0.7]])
        marks = np.array([[[.1, .2], [.1, .1]],
                          [[.3, .0], [.1, .2]],
                          [[.2, .2], [.1, .4]]])
        cutoff, rate = 9, .4
        joint = np.arange(1., 21.).reshape(10, 2)
        joint /= joint.sum()
        expected = np.zeros_like(joint)
        for r in range(cutoff + 1):
            for i in range(2):
                for j in range(2):
                    mean = rate * (1 + states[j] - states[i])
                    for c in range(3):
                        weight = joint[r, i] * marks[c, i, j]
                        expected[0, j] += weight * poisson.cdf(c - r, mean)
                        for s in range(1, cutoff):
                            expected[s, j] += weight * poisson.pmf(s + c - r, mean)
                        expected[cutoff, j] += weight * poisson.sf(cutoff + c - r - 1, mean)
        actual = joint_transition_operator(states, transition, 1., rate, 2, cutoff, marks)(joint)
        np.testing.assert_allclose(actual, expected, atol=3e-14)


if __name__ == '__main__':
    unittest.main()
