import unittest

import numpy as np

from revision_inputs import (BASE_RATES, all_scenarios, heterogeneous_destinations,
                             validation_profiles)


class RevisionInputs(unittest.TestCase):
    def test_general_destination_law_preserves_original_hazards(self):
        p = heterogeneous_destinations()
        np.testing.assert_allclose(p[:9].sum(axis=1), 1.)
        self.assertTrue(np.all(np.tril(p) == 0))
        self.assertEqual(p[0, 1], 0.)
        self.assertAlmostEqual(p[0, 2], .1)
        self.assertAlmostEqual(p[0, 3], .9 * .25)
        self.assertEqual(p[8, 9], 1.)

    def test_all_19_validation_profiles_are_strictly_stable_and_identifiable(self):
        profiles = validation_profiles()
        self.assertEqual(len(profiles), 19)
        for case in profiles:
            self.assertLess(case['maximum_cut_load'], 1.)
            self.assertEqual(case['rates'][-1], 0.)
        high = next(c for c in profiles if c['name'] == 'heterogeneous-e097')
        self.assertAlmostEqual(high['maximum_cut_load'], .97)
        self.assertEqual(np.argmax(high['cut_loads']) + 1, 3)
        low = next(c for c in profiles if c['name'] == 'original-reference')
        np.testing.assert_array_equal(low['rates'], BASE_RATES * .75)

    def test_aliases_deduplicate_shared_inputs_without_losing_policy_cases(self):
        cases = all_scenarios()
        self.assertEqual(len({c['name'] for c in cases}),len(cases))
        self.assertEqual(sum(c['phase_validation'] for c in cases), 19)
        self.assertEqual(sum(len(c['aliases']) for c in cases), 43)
        self.assertTrue(any(c['first_failing_station'] is not None for c in cases))


if __name__ == '__main__':
    unittest.main()
