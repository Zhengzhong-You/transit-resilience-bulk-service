import unittest

import numpy as np

from service_dynamics import fifo_departures, stationwise_departures


class ServiceDynamicsTest(unittest.TestCase):
    def test_upstream_hold_survives_a_later_incident(self):
        dispatch = [0.0, 1.0]
        delays = np.array([[10.0, 0.0], [0.0, 1.0]])
        actual = fifo_departures(dispatch, 1.0, delays)
        np.testing.assert_array_equal(actual, [[11.0, 12.0], [11.0, 13.0]])
        legacy = stationwise_departures(dispatch, 1.0, delays)
        self.assertLess(legacy[1, 1] - legacy[1, 0], 1.0 + delays[1, 1])

    def test_no_incidents_preserve_the_schedule_at_every_station(self):
        dispatch = np.arange(20) * 4.8
        actual = fifo_departures(dispatch, [2.0, 3.0, 5.0], np.zeros((20, 3)))
        np.testing.assert_allclose(actual, dispatch[:, None] + [2.0, 5.0, 10.0])

    def test_trajectories_respect_traversal_and_service_order(self):
        rng = np.random.default_rng(20261031)
        delays = rng.exponential(size=(100, 5))
        dispatch = np.arange(100) * 2.0
        actual = fifo_departures(dispatch, 5.0, delays)
        upstream = np.column_stack([dispatch, actual[:, :-1]])
        self.assertTrue(np.all(actual >= upstream + 5.0 + delays))
        self.assertTrue(np.all(np.diff(actual, axis=0) >= 0))


if __name__ == '__main__':
    unittest.main()
