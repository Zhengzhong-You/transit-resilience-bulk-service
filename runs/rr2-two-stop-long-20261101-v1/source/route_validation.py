"""Explicit FCFS passenger/destination benchmark for the R&R operational tests."""

from dataclasses import dataclass

import numpy as np


def destination_law(stations=10, alighting=0.25):
    probabilities = np.zeros((stations, stations))
    for origin in range(stations - 1):
        survival = 1.0
        for destination in range(origin + 1, stations):
            p = 1.0 if destination == stations - 1 else alighting
            probabilities[origin, destination] = survival * p
            survival *= 1 - p
    return probabilities


def cut_loads(rates, destinations, headway):
    flows = np.asarray(rates)[:, None] * destinations
    return headway * np.array([flows[:n + 1, n + 1:].sum()
                              for n in range(len(rates))])


def matched_destination_law(rates, destinations, seed=2026103400):
    """Change OD dependence by triangular 2x2 exchanges, preserving both margins."""
    rng = np.random.default_rng(seed)
    flows = np.asarray(rates)[:, None] * destinations
    altered = flows.copy()
    stations = len(rates)
    for _ in range(5000):
        origins = np.sort(rng.choice(stations - 2, 2, replace=False))
        choices = np.arange(origins[-1] + 1, stations)
        if len(choices) < 2:
            continue
        d1, d2 = rng.choice(choices, 2, replace=False)
        a, b = origins
        amount = 0.9 * min(altered[a, d2], altered[b, d1])
        altered[a, d1] += amount
        altered[b, d2] += amount
        altered[a, d2] -= amount
        altered[b, d1] -= amount
    np.testing.assert_allclose(altered.sum(axis=0), flows.sum(axis=0), atol=1e-12)
    np.testing.assert_allclose(altered.sum(axis=1), flows.sum(axis=1), atol=1e-12)
    result = np.divide(altered, np.asarray(rates)[:, None],
                       out=np.zeros_like(altered), where=np.asarray(rates)[:, None] > 0)
    return result


def incident_delays(seed, vehicles, stations, gamma=0.2, duration='exponential',
                    segment_time=5.0, duration_seed=None):
    """Incidents accrue during normal travel only; all duration means are one minute."""
    rng = np.random.default_rng(seed)
    counts = rng.poisson(gamma * segment_time, (vehicles, stations))
    # A separate duration stream preserves the same incident prefix when
    # a calendar-cohort follow-up requires additional vehicle runs.
    duration_rng = rng if duration_seed is None else np.random.default_rng(duration_seed)
    if duration == 'exponential':
        return duration_rng.gamma(counts, 1.0)
    if duration == 'deterministic':
        return counts.astype(float)
    if duration == 'lognormal':
        # sigma=sqrt(log(2)) matches the exponential first TWO moments.
        sigma = np.sqrt(np.log(2.0))
        values = duration_rng.lognormal(-sigma ** 2 / 2, sigma, counts.sum())
        result = np.zeros(counts.size)
        nonempty = np.flatnonzero(counts.ravel())
        if len(nonempty):
            result[nonempty] = np.add.reduceat(values, np.r_[0, counts.ravel().cumsum()[:-1]][nonempty])
        return result.reshape(counts.shape)
    raise ValueError('Unknown incident duration distribution.')


class CalendarPassengers:
    """A reproducible calendar stream, independent of policy and service queries."""

    def __init__(self, rate, probabilities, seed, initial=0):
        self.rate = float(rate)
        self.probabilities = probabilities
        self.time_rng = np.random.default_rng(seed)
        self.mark_rng = np.random.default_rng(seed + 100000000)
        self.last_generated = 0.0
        self.times = -np.arange(initial, 0, -1, dtype=float) / max(rate, 1.0)
        initial_rng = np.random.default_rng(seed + 200000000)
        self.destinations = self._mark(initial_rng, initial)
        self.sums = np.r_[0., self.times.cumsum()]
        self.squares = np.r_[0., (self.times ** 2).cumsum()]

    def _mark(self, rng, number):
        if not number:
            return np.empty(0, dtype=int)
        return np.searchsorted(self.probabilities.cumsum(), rng.random(number), side='right')

    def through(self, time):
        if self.rate > 0:
            while self.last_generated <= time:
                values = self.last_generated + self.time_rng.exponential(1 / self.rate, 50000).cumsum()
                self.last_generated = values[-1]
                self.times = np.r_[self.times, values]
                self.destinations = np.r_[self.destinations, self._mark(self.mark_rng, len(values))]
                self.sums = np.r_[self.sums, self.sums[-1] + values.cumsum()]
                self.squares = np.r_[self.squares, self.squares[-1] + (values ** 2).cumsum()]
        return int(np.searchsorted(self.times, time, side='right'))


@dataclass
class RouteResult:
    starts: np.ndarray
    departures: np.ndarray
    residuals: np.ndarray
    queues: np.ndarray
    capacities: np.ndarray
    boarded: np.ndarray
    alighted: np.ndarray
    dwell: np.ndarray
    wait_sum: np.ndarray
    wait_square_sum: np.ndarray
    origin: np.ndarray
    returned: np.ndarray

    def summaries(self, headway, measurement_start=None):
        vehicles, stations = self.starts.shape
        begin = vehicles // 2 if measurement_start is None else measurement_start
        sample = slice(begin, vehicles)
        rows = []
        for n in range(stations):
            order = np.argsort(self.starts[:, n], kind='stable')
            measured = order[sample]
            previous, last = order[begin - 1], order[-1]
            intervals = np.diff(self.starts[order, n])[begin - 1:]
            departures = np.diff(self.departures[order, n])[begin - 1:]
            b = self.boarded[measured, n].sum()
            mean_wait = self.wait_sum[measured, n].sum() / b if b else float('nan')
            variance_wait = self.wait_square_sum[measured, n].sum() / b - mean_wait ** 2 if b else float('nan')
            corr = (float(np.corrcoef(intervals[:-1], intervals[1:])[0, 1])
                    if intervals.std() > 1e-8 else 0.0)
            rows.append({'station': n + 1, 'mean_headway': float(intervals.mean()),
                         'variance_headway': float(intervals.var()),
                         'headway_lag1_correlation': corr,
                         'variance_departure_headway': float(departures.var()),
                         'mean_residual': float(self.residuals[measured, n].mean()),
                         'terminal_residual': int(self.residuals[last, n]),
                         'queue_drift_per_minute': float((self.residuals[last, n] - self.residuals[previous, n])
                             / (self.starts[last, n] - self.starts[previous, n])),
                         'mean_queue': float(self.queues[measured, n].mean()),
                         'variance_queue': float(self.queues[measured, n].var()),
                         'mean_available_capacity': float(self.capacities[measured, n].mean()),
                         'mean_boarded': float(self.boarded[measured, n].mean()),
                         'mean_alighted': float(self.alighted[measured, n].mean()),
                         'mean_dwell': float(self.dwell[measured, n].mean()),
                         'mean_wait_boarded': float(mean_wait),
                         'variance_wait_boarded': float(variance_wait),
                         'boarded_observations': int(b),
                         'achieved_origin_headway': float(np.diff(self.origin)[begin - 1:].mean()),
                         'origin_lateness_drift': float(((self.origin[-1] - (vehicles - 1) * headway)
                             - (self.origin[begin - 1] - (begin - 1) * headway))
                             / (self.origin[-1] - self.origin[begin - 1])),
                         'mean_vehicle_cycle': float((self.returned - self.origin)[sample].mean())})
        return rows


def arrival_cohort_summaries(result, rates, destinations, seed, initial_backlog=0,
                              scheduled_headway=4.8):
    """Follow a common nominal-calendar arrival cohort through the complete run.

    Unserved members remain explicitly censored; their waiting times are never
    imputed as completed waits. Reconstructing streams preserves the same inputs.
    """
    vehicles, stations = result.starts.shape
    initial = np.broadcast_to(initial_backlog, stations)
    rows = []
    for n in range(stations):
        order = np.argsort(result.starts[:, n], kind='stable')
        clock = result.starts[order, n]
        stream = CalendarPassengers(rates[n], destinations[n], seed + n * 1000, int(initial[n]))
        stream.through(clock[-1])
        first = np.searchsorted(stream.times, .5 * vehicles * scheduled_headway, side='right')
        stop = np.searchsorted(stream.times, .75 * vehicles * scheduled_headway, side='right')
        cumulative_boarded = result.boarded[order, n].cumsum()
        served_stop = max(first, min(stop, int(cumulative_boarded[-1])))
        identities = np.arange(first, served_stop)
        service = np.searchsorted(cumulative_boarded, identities + 1, side='left')
        waits = clock[service] - stream.times[identities]
        assert np.all(waits >= -1e-8)
        total, served = stop - first, len(waits)
        rows.append({'arrival_cohort_passengers': int(total), 'arrival_cohort_served': int(served),
                     'arrival_cohort_censored_fraction': float(1 - served / total) if total else 0.,
                     'arrival_cohort_mean_wait_completed': float(waits.mean()) if served else float('nan'),
                     'arrival_cohort_variance_wait_completed': float(waits.var()) if served else float('nan')})
    return rows


def simulate_route(rates, destinations, delays, seed, headway=4.8, capacity=50,
                   segment_time=5., dwell_coefficients=(0., 0., 0.),
                   prescribed_dwell=None, fleet=None, return_delays=None,
                   initial_backlog=0, passing=False):
    """Gated boarding at service start; a following vehicle waits for berth release.

    With passing=True only exogenous zero-dwell service is supported. Vehicles
    then board in physical arrival order at each station, retaining their loads.
    """
    rates = np.asarray(rates)
    vehicles, stations = delays.shape
    if passing and (fleet is not None or prescribed_dwell is not None
                    or any(dwell_coefficients)):
        raise ValueError('Passing comparison requires zero dwell and exogenous dispatch.')
    if fleet is not None and (fleet < 1 or return_delays is None):
        raise ValueError('A finite fleet requires a positive size and explicit return incidents.')
    if np.any(rates < 0) or np.any(np.tril(destinations) != 0):
        raise ValueError('Demand must be nonnegative with strictly downstream destinations.')
    np.testing.assert_allclose(destinations.sum(axis=1)[rates > 0], 1.0, atol=1e-12)
    shape = (vehicles, stations)
    result = RouteResult(*[np.zeros(shape, dtype=dtype) for dtype in
        [float, float, int, int, int, int, int, float, float, float]],
        origin=np.arange(vehicles, dtype=float) * headway, returned=np.zeros(vehicles))
    initial = np.broadcast_to(initial_backlog, stations)
    streams = [CalendarPassengers(rate, destinations[n], seed + n * 1000, int(initial[n]))
               for n, rate in enumerate(rates)]
    cursors = np.zeros(stations, dtype=int)
    previous_departures = np.zeros(stations)
    previous_return = np.zeros(stations)
    a, b, c = dwell_coefficients

    def board(vehicle, station, start, onboard):
        stream = streams[station]
        arrived = stream.through(start)
        old = cursors[station]
        q = arrived - old
        alighted = onboard[station]
        onboard[station] = 0
        available = capacity - onboard.sum()
        number = min(available, q)
        stop = old + number
        if number:
            onboard += np.bincount(stream.destinations[old:stop], minlength=stations)
        cursors[station] = stop
        dwell = (prescribed_dwell[vehicle, station] if prescribed_dwell is not None
                 else a + b * number + c * alighted)
        result.starts[vehicle, station] = start
        result.departures[vehicle, station] = start + dwell
        result.residuals[vehicle, station] = q - number
        result.queues[vehicle, station] = q
        result.capacities[vehicle, station] = available
        result.boarded[vehicle, station] = number
        result.alighted[vehicle, station] = alighted
        result.dwell[vehicle, station] = dwell
        result.wait_sum[vehicle, station] = number * start - (stream.sums[stop] - stream.sums[old])
        result.wait_square_sum[vehicle, station] = (number * start ** 2
            - 2 * start * (stream.sums[stop] - stream.sums[old])
            + stream.squares[stop] - stream.squares[old])
        return start + dwell

    if passing:
        times = result.origin[:, None] + np.cumsum(segment_time + delays, axis=1)
        onboard = np.zeros((vehicles, stations), dtype=int)
        for n in range(stations):
            for vehicle in np.argsort(times[:, n], kind='stable'):
                board(vehicle, n, times[vehicle, n], onboard[vehicle])
        result.returned = result.departures[:, -1] + stations * segment_time
    else:
        for vehicle in range(vehicles):
            if fleet is not None and vehicle >= fleet:
                result.origin[vehicle] = max(result.origin[vehicle], result.returned[vehicle - fleet])
            time = result.origin[vehicle]
            onboard = np.zeros(stations, dtype=int)
            for n in range(stations):
                start = max(time + segment_time + delays[vehicle, n], previous_departures[n])
                time = board(vehicle, n, start, onboard)
                previous_departures[n] = time
            if return_delays is None:
                result.returned[vehicle] = time + stations * segment_time
            else:
                for n in range(stations):
                    time = max(time + segment_time + return_delays[vehicle, n], previous_return[n])
                    previous_return[n] = time
                result.returned[vehicle] = time
    assert np.all(result.residuals >= 0) and np.all(result.capacities >= 0)
    assert np.all(result.boarded <= result.capacities)
    assert result.boarded.sum() == result.alighted.sum()  # every passenger exits at the terminal
    return result
