import unittest

import numpy as np

from rootfree_bulk import (deterministic_bulk_roots, effective_arrival_pgf,
                           moments_from_roots, contour_moments)


class DeterministicBulkRoots(unittest.TestCase):
    def test_capacity50_has_all_characteristic_zeros(self):
        for rho in [.5, .9, .97, .99]:
            roots = deterministic_bulk_roots(50, rho)
            residual = roots ** 50 - np.exp(50 * rho * (roots - 1))
            self.assertEqual(len(roots), 50)
            self.assertLess(np.max(np.abs(residual)), 1e-10)
            self.assertLess(np.max(np.abs(roots[1:])), 1.)
            self.assertGreater(np.min(np.abs(roots[:, None] - roots[None, :])
                                      + np.eye(50)), 1e-5)

    def test_moments_match_an_independent_contour_evaluation(self):
        for capacity in [1, 8, 34, 50]:
            pgf, derivatives, _ = effective_arrival_pgf(capacity, .97)
            roots = deterministic_bulk_roots(capacity, .97)
            first, variance = moments_from_roots(roots, capacity, derivatives)
            numerical = contour_moments(capacity, pgf)
            self.assertAlmostEqual(first.real, numerical['mean_residual'], places=6)
            self.assertAlmostEqual(variance.real, numerical['variance_residual'], places=6)
        self.assertAlmostEqual(moments_from_roots(
            deterministic_bulk_roots(1, .5), 1,
            effective_arrival_pgf(1, .5)[1])[0].real, .25, places=12)


if __name__ == '__main__':
    unittest.main()
