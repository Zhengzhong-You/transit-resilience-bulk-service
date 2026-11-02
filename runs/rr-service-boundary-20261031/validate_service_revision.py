"""Matched zero-dwell service and fixed-capacity stability checks for the R&R."""

import argparse
import csv
import hashlib
import json
import platform
import shutil
import time
from pathlib import Path

import numpy as np
from scipy.stats import norm

from service_dynamics import fifo_departures, stationwise_departures


def residual_queues(arrivals, capacity, initial=0):
    net = initial + np.cumsum(np.asarray(arrivals, dtype=np.int64) - capacity)
    return net - np.minimum(0, np.minimum.accumulate(net))


def write_rows(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--vehicles', type=int, default=80000)
    parser.add_argument('--replications', type=int, default=20)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    for name in ["service_dynamics.py", "validate_service_revision.py"]:
        shutil.copy2(name, args.output / name)
    if args.vehicles < 1000 or args.replications < 2:
        raise ValueError('Use at least 1000 vehicles and two independent replications.')
    start = time.perf_counter()
    h, gamma, theta, segment_time, capacity = 4.8, 0.2, 1.0, 5.0, 34
    stations, chosen_station, burn = 10, 7, 10000
    times = segment_time * np.arange(1, stations + 1)
    sigma = 2 * np.sqrt(gamma * times) / theta
    normal_means = h * norm.cdf(h / sigma) + sigma * norm.pdf(h / sigma)
    factors = h / normal_means
    ratios = [0.80, 0.87, 0.90, 0.97, 1.0, 1.03, 1.10]
    horizons = sorted(set([args.vehicles // 16, args.vehicles // 4, args.vehicles]))
    headway_rows, queue_rows, trace_rows, diagnostic_rows = [], [], [], []
    seeds = []
    for replicate in range(args.replications):
        seed = 2026103100 + replicate
        seeds.append(seed)
        rng = np.random.default_rng(seed)
        counts = rng.poisson(gamma * segment_time, (args.vehicles + burn + 1, stations))
        delays = rng.gamma(counts, 1 / theta)
        dispatch = np.arange(len(delays)) * h
        propagated = fifo_departures(dispatch, segment_time, delays)
        stationwise = stationwise_departures(dispatch, segment_time, delays)
        upstream_sw = np.column_stack([dispatch, stationwise[:, :-1]])
        violations = stationwise < upstream_sw + segment_time + delays - 1e-10
        diagnostic_rows.append({'replicate': replicate, 'seed': seed,
                                'legacy_traversal_violations': int(violations[burn:].sum()),
                                'tested_vehicle_segments': int(violations[burn:].size)})
        fifo_h = np.diff(propagated, axis=0)[burn:]
        sw_h = np.diff(stationwise, axis=0)[burn:]
        raw_h = h + np.diff(np.cumsum(delays, axis=1), axis=0)[burn:]
        normal_h = np.maximum(0, h + sigma * rng.standard_normal(fifo_h.shape))
        models = {'propagated_fifo': fifo_h, 'legacy_stationwise': sw_h,
                  'raw_shared_delays': raw_h, 'unscaled_normal': normal_h,
                  'rate_matched_normal': normal_h * factors}
        for model, headways in models.items():
            for station in range(stations):
                values = headways[:, station]
                headway_rows.append({'replicate': replicate, 'seed': seed, 'model': model,
                                     'station': station + 1, 'mean_headway': float(values.mean()),
                                     'sd_headway': float(values.std(ddof=1)),
                                     'lag1_correlation': float(np.corrcoef(values[:-1], values[1:])[0, 1]),
                                     'zero_fraction': float(np.mean(values == 0))})
        selected = {name: values[:, chosen_station] for name, values in models.items()
                    if name in ['propagated_fifo', 'unscaled_normal', 'rate_matched_normal']}
        selected['fifo_permuted'] = rng.permutation(selected['propagated_fifo'])
        clocks = {name: np.cumsum(values) for name, values in selected.items()}
        for ratio in ratios:
            rate = capacity / h * ratio
            end = max(clock[-1] for clock in clocks.values())
            # One calendar-time Poisson stream is shared across service models.
            arrivals = np.sort(rng.uniform(0, end, rng.poisson(rate * end)))
            for model, clock in clocks.items():
                cumulative_arrivals = np.searchsorted(arrivals, clock, side='right')
                per_service = np.diff(np.r_[0, cumulative_arrivals])
                for initial in [0, 20 * capacity]:
                    backlog = residual_queues(per_service, capacity, initial)
                    assert np.all(backlog >= 0)
                    for horizon in horizons:
                        tail = backlog[horizon // 2:horizon]
                        elapsed = clock[horizon - 1] - clock[horizon // 2 - 1]
                        growth = (backlog[horizon - 1] - backlog[horizon // 2 - 1]) / elapsed
                        queue_rows.append({'replicate': replicate, 'seed': seed, 'model': model,
                                           'station': chosen_station + 1, 'load_ratio': ratio,
                                           'arrival_rate': rate, 'initial_backlog': initial,
                                           'vehicles': horizon, 'elapsed_time': float(clock[horizon - 1]),
                                           'final_backlog': int(backlog[horizon - 1]),
                                           'backlog_per_time': float(backlog[horizon - 1] / clock[horizon - 1]),
                                           'second_half_mean_backlog': float(tail.mean()),
                                           'second_half_growth_per_time': float(growth)})
                    if replicate == 0 and initial == 0:
                        for index in np.arange(99, len(backlog), 100):
                            trace_rows.append({'model': model, 'load_ratio': ratio,
                                               'vehicles': int(index + 1), 'time': float(clock[index]),
                                               'backlog': int(backlog[index])})
        print(f'replication {replicate + 1}/{args.replications}', flush=True)
    write_rows(args.output / 'headway_replications.csv', headway_rows)
    write_rows(args.output / 'stability_replications.csv', queue_rows)
    write_rows(args.output / 'queue_trajectories.csv', trace_rows)
    write_rows(args.output / 'chronology_diagnostics.csv', diagnostic_rows)
    protocol = {'h': h, 'gamma': gamma, 'theta': theta, 'segment_time': segment_time,
                'stations': stations, 'capacity': capacity, 'queue_station': chosen_station + 1,
                'burn_in_vehicles': burn, 'vehicles': args.vehicles, 'replications': args.replications,
                'horizons': horizons, 'load_ratios': ratios, 'seeds': seeds,
                'normal_positive_part_means': normal_means.tolist(),
                'normal_scaling_factors': factors.tolist(),
                'elapsed_seconds': time.perf_counter() - start, 'python': platform.python_version(),
                'numpy': np.__version__,
                'source_sha256': {name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                                  for name in ['validate_service_revision.py', 'service_dynamics.py']},
                'scope': 'Zero dwell, exogenous dispatch, one station with deterministic available capacity. Does not validate downstream endogenous capacity, dwell feedback or finite fleet feasibility.',
                'critical_case': 'Finite-horizon diagnostics at load_ratio=1 are not a proof or classification of stability.'}
    (args.output / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    print(json.dumps({'output': str(args.output), 'elapsed_seconds': protocol['elapsed_seconds']}))


if __name__ == '__main__':
    main()
