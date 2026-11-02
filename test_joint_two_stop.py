import unittest

import numpy as np

from joint_two_stop import arrival_probabilities, joint_operator, stationary_joint


class DirectJointQueue(unittest.TestCase):
    def test_fft_operator_matches_independent_state_enumeration(self):
        cutoff, a, b, p = 4, .35, .25, .5
        p1, p2 = arrival_probabilities(a, cutoff), arrival_probabilities(b, cutoff)
        initial = np.random.default_rng(41).random((cutoff + 1, cutoff + 1))
        initial /= initial.sum()
        reference = np.zeros_like(initial)
        for i in range(cutoff + 1):
            for j in range(cutoff + 1):
                for y1, f1 in enumerate(p1):
                    for y2, f2 in enumerate(p2):
                        boarded = int(i + y1 > 0)
                        r1 = min(cutoff, max(0, i + y1 - 1))
                        for mark, probability in [(0, 1 - p), (1, p)]:
                            r2 = min(cutoff, max(0, j + y2 - 1 + boarded * mark))
                            reference[r1, r2] += initial[i, j] * f1 * f2 * probability
        np.testing.assert_allclose(joint_operator(a, b, p, cutoff)(initial), reference,
                                   atol=1e-13, rtol=1e-12)

    def test_first_station_matches_known_bulk_queue_and_zero_p_decouples(self):
        _, result = stationary_joint(.35, .25, 0., 32)
        self.assertAlmostEqual(result['mean_residual'][0], .35 ** 2 / (2 * .65), places=8)
        self.assertAlmostEqual(result['mean_residual'][1], .25 ** 2 / (2 * .75), places=8)
        self.assertLess(result['joint_product_total_variation'], 1e-9)


if __name__ == '__main__':
    unittest.main()
