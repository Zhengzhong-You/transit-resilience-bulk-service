import unittest

import numpy as np

from route_validation import (CalendarPassengers, cut_loads, destination_law,
                              matched_destination_law, simulate_route, arrival_cohort_summaries)
from service_dynamics import fifo_departures


class RouteValidationTest(unittest.TestCase):
    def test_zero_dwell_chronology_and_passenger_conservation(self):
        delays = np.random.default_rng(1).exponential(2., (100, 4))
        rates = np.array([1., 0.7, 0.5, 0.])
        result = simulate_route(rates, destination_law(4), delays, 5)
        expected = fifo_departures(np.arange(100) * 4.8, 5., delays)
        np.testing.assert_allclose(result.starts, expected)
        self.assertEqual(result.boarded.sum(), result.alighted.sum())
        self.assertTrue(np.all(result.queues - result.boarded == result.residuals))

    def test_calendar_stream_does_not_change_with_service_queries(self):
        p = np.array([0., 0.4, 0.6])
        first, second = CalendarPassengers(2, p, 7), CalendarPassengers(2, p, 7)
        for time in [10, 20000, 70000]:
            first.through(time)
        second.through(70000)
        np.testing.assert_array_equal(first.times, second.times)
        np.testing.assert_array_equal(first.destinations, second.destinations)

    def test_fleet_reuse_and_single_berth_dwell(self):
        delays = np.zeros((30, 3))
        result = simulate_route(np.zeros(3), np.zeros((3, 3)), delays, 9,
                                fleet=2, return_delays=delays,
                                dwell_coefficients=(1., 0., 0.))
        self.assertTrue(np.all(result.origin[2:] >= result.returned[:-2]))
        self.assertTrue(np.all(result.starts[1:] >= result.departures[:-1]))
        np.testing.assert_allclose(result.returned - result.origin, 33.)

    def test_od_changes_preserve_station_flows_and_cut_loads(self):
        rates = np.r_[np.ones(9), 0.]
        original = destination_law()
        altered = matched_destination_law(rates, original)
        np.testing.assert_allclose(cut_loads(rates, original, 4.8),
                                   cut_loads(rates, altered, 4.8), atol=1e-11)
        self.assertGreater(np.abs(original - altered).sum(), 1.)

    def test_passing_uses_arrival_order_and_preserves_vehicle_load(self):
        delays = np.array([[12., 0., 0.], [0., 0., 0.], [1., 0., 0.]])
        result = simulate_route(np.array([3., 2., 0.]), destination_law(3),
                                delays, 12, capacity=4, passing=True)
        self.assertLess(result.starts[1, 0], result.starts[0, 0])
        self.assertEqual(result.boarded.sum(), result.alighted.sum())

    def test_arrival_cohort_tracks_completion_and_exposes_right_censoring(self):
        rates = np.array([2., 0.])
        p = destination_law(2)
        stable = simulate_route(rates, p, np.zeros((1000, 2)), 17, capacity=20)
        cohort = arrival_cohort_summaries(stable, rates, p, 17)[0]
        self.assertGreater(cohort['arrival_cohort_passengers'], 0)
        self.assertEqual(cohort['arrival_cohort_censored_fraction'], 0.)
        overloaded = simulate_route(rates, p, np.zeros((1000, 2)), 17, capacity=1)
        cohort = arrival_cohort_summaries(overloaded, rates, p, 17)[0]
        self.assertGreater(cohort['arrival_cohort_censored_fraction'], .9)

    def test_common_calendar_cohort_has_identical_members_under_different_fleets(self):
        rates = np.array([2., 0.])
        p = destination_law(2)
        delays = np.zeros((1000, 2))
        unlimited = simulate_route(rates, p, delays, 17, capacity=20)
        closed = simulate_route(rates, p, delays, 17, capacity=20, fleet=3,
                                return_delays=delays)
        first = arrival_cohort_summaries(unlimited, rates, p, 17)[0]
        second = arrival_cohort_summaries(closed, rates, p, 17)[0]
        self.assertEqual(first['arrival_cohort_passengers'], second['arrival_cohort_passengers'])


if __name__ == '__main__':
    unittest.main()
