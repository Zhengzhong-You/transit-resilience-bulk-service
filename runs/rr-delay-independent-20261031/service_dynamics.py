"""Zero-dwell service times for matched R&R validation."""

import numpy as np


def fifo_departures(dispatch_times, segment_times, segment_delays):
    """Carry every upstream hold into the next segment before enforcing order."""
    dispatch_times = np.asarray(dispatch_times, dtype=float)
    segment_delays = np.asarray(segment_delays, dtype=float)
    segment_times = np.broadcast_to(segment_times, segment_delays.shape[1:])
    if segment_delays.ndim != 2 or dispatch_times.shape != segment_delays.shape[:1]:
        raise ValueError("One incident delay is required for each vehicle and segment.")
    if (not np.all(np.isfinite(segment_delays)) or np.any(segment_delays < 0)
            or not np.all(np.isfinite(dispatch_times))
            or np.any(np.diff(dispatch_times) < 0)
            or not np.all(np.isfinite(segment_times)) or np.any(segment_times <= 0)):
        raise ValueError("Service inputs must be finite, ordered, and nonnegative.")
    departures = np.empty_like(segment_delays)
    upstream = dispatch_times
    for station in range(segment_delays.shape[1]):
        upstream = np.maximum.accumulate(
            upstream + segment_times[station] + segment_delays[:, station]
        )
        departures[:, station] = upstream
    return departures


def stationwise_departures(dispatch_times, segment_times, segment_delays):
    """Reproduce the legacy benchmark without carrying upstream holds forward."""
    free_times = np.asarray(dispatch_times)[:, None] + np.cumsum(
        np.asarray(segment_delays) + segment_times, axis=1
    )
    return np.maximum.accumulate(free_times, axis=0)
