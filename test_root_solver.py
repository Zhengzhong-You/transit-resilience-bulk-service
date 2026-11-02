import unittest

from root_solver import certified_interpolation_roots, winding_number


class RootSolverTest(unittest.TestCase):
    @staticmethod
    def characteristic(z):
        return (z - 1.0) * (z + 0.25) ** 2

    def test_winding_number_counts_multiplicity(self):
        self.assertEqual(winding_number(self.characteristic, 1.0 + 0.0j, 0.05), 1)
        self.assertEqual(winding_number(self.characteristic, -0.25 + 0.0j, 0.05), 2)

    def test_search_does_not_infer_multiplicity_from_a_cluster_count(self):
        # The numerical search cannot distinguish a double zero from two
        # unresolved simple zeros using a sampled enclosing contour alone.
        with self.assertRaisesRegex(RuntimeError, 'Unresolved root cluster'):
            certified_interpolation_roots(self.characteristic, capacity=3,
                                          utilization=0.5)


if __name__ == '__main__':
    unittest.main()
