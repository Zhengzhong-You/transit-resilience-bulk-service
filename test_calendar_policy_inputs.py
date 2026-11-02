"""Matched calendar inputs for headway policies and longer cohort follow-up."""

import unittest

import numpy as np

from route_validation import CalendarPassengers, incident_delays
from validate_route_phase import queue_metrics


class CalendarPolicyInputs(unittest.TestCase):
    def test_incident_prefix_survives_longer_followup(self):
        for law in ['exponential', 'deterministic', 'lognormal']:
            short = incident_delays(713, 137, 10, .2, law, duration_seed=812)
            extended = incident_delays(713, 439, 10, .2, law, duration_seed=812)
            np.testing.assert_array_equal(short, extended[:137])

    def test_same_calendar_arrivals_under_different_headways(self):
        rows = []
        for h in [2., 4.]:
            stream = CalendarPassengers(2., np.array([0., 1.]), 317)
            clock = np.arange(0., 44., h)
            metrics, _ = queue_metrics(clock, np.full(len(clock), 1000), stream,
                scheduled_headway=h, observation_window=(16., 36.),
                cohort_window=(20., 30.))
            rows.append(metrics)
        self.assertGreater(rows[0]['cohort_passengers'], 0)
        self.assertEqual(rows[0]['cohort_passengers'], rows[1]['cohort_passengers'])
        self.assertEqual(rows[0]['cohort_served'], rows[1]['cohort_served'])
        self.assertEqual(rows[0]['mean_headway'], 2.)
        self.assertEqual(rows[1]['mean_headway'], 4.)

    def test_followup_censoring_counts_all_arrivals_in_window(self):
        stream = CalendarPassengers(2., np.array([0., 1.]), 317)
        clock = np.arange(0., 24., 4.)
        metrics, _ = queue_metrics(clock, np.full(len(clock), 1000), stream,
            cohort_window=(20., 30.))
        self.assertGreater(metrics['cohort_passengers'], 0)
        self.assertEqual(metrics['cohort_served'], 0)
        self.assertEqual(metrics['censored_fraction'], 1.)


if __name__ == '__main__':
    unittest.main()
