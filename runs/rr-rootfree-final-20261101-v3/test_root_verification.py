import unittest

import numpy as np

from root_solver import certified_interpolation_roots
from rootfree_bulk import effective_arrival_pgf


class RootVerificationTest(unittest.TestCase):
    def test_two_nearby_simple_zeros_are_not_reported_as_one_repeated_root(self):
        f = lambda z: (z - 1) * (z - .5) * (z - .50005)
        with self.assertRaisesRegex(RuntimeError, 'Unresolved root cluster'):
            certified_interpolation_roots(f, 3, .5, max_iterations=2, max_interpolation=2)

    def test_separated_simple_zeros_are_still_returned(self):
        f = lambda z: (z - 1) * (z - .2) * (z - .6)
        roots, diagnostics = certified_interpolation_roots(f, 3, .5)
        np.testing.assert_allclose(np.sort(roots.real), [.2, .6, 1.], atol=1e-8)
        self.assertEqual(diagnostics['multiplicities'], [1, 1, 1])

    def test_nearby_exterior_zero_does_not_count_as_a_repeated_unit_root(self):
        _, _, f = effective_arrival_pgf(1, .99, headway=1.)
        roots, diagnostics = certified_interpolation_roots(f, 1, .99)
        np.testing.assert_allclose(roots, [1.], atol=1e-12)
        self.assertEqual(diagnostics['multiplicities'], [1])


if __name__ == '__main__':
    unittest.main()
