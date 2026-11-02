import unittest

import numpy as np

from delay_stationary import stationary_queue
from rootfree_bulk import contour_moments, effective_arrival_pgf


class RootFreeBulkTest(unittest.TestCase):
    def test_complex_mixed_poisson_pgf_stays_finite_on_the_contour(self):
        values, _, _ = effective_arrival_pgf(50, .99, .7, sigma=2 * np.sqrt(8.))
        z = 1.001 * np.exp(2j * np.pi * np.arange(1000) / 1000)
        a, derivative = values(z)
        self.assertTrue(np.all(np.isfinite(a)))
        self.assertTrue(np.all(np.isfinite(derivative)))

    def test_single_capacity_exact_first_and_second_moments(self):
        for a in [.2, .5, .9, .99]:
            values, _, _ = effective_arrival_pgf(1, a, headway=1.)
            result = contour_moments(1, values)
            mean = a ** 2 / (2 * (1 - a))
            variance = mean + a ** 3 / (3 * (1 - a)) + a ** 4 / (4 * (1 - a) ** 2)
            self.assertAlmostEqual(result['mean_residual'], mean, delta=1e-7)
            self.assertAlmostEqual(result['variance_residual'], variance, delta=1e-5)

    def test_random_capacity_reduction_matches_an_independent_stationary_queue(self):
        # Two iid capacity phases give S=0 or2, with probabilities .3 and .7.
        transition = np.array([[.3, .7], [.3, .7]])
        result = stationary_queue(np.zeros(2), transition, 1., .8, [0, 2], 160,
                                  tolerance=1e-13)
        def pgf(z):
            y = np.exp(.8 * (z - 1))
            return y * (.7 + .3 * z ** 2), y * (.8 * (.7 + .3 * z ** 2) + .6 * z)
        contour = contour_moments(2, pgf)
        self.assertAlmostEqual(contour['mean_residual'], result['mean_residual'], places=8)
        self.assertAlmostEqual(contour['variance_residual'], result['variance_residual'], places=7)


if __name__ == '__main__':
    unittest.main()
