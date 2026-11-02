"""Independent-data pilot of a mean-preserving departure-displacement kernel."""

import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from service_dynamics import fifo_departures
from validate_service_revision import residual_queues


def fit_kernel(displacements, step, tail_probability):
    maximum = int(np.floor(np.quantile(displacements, 1 - tail_probability) / step))
    labels = np.minimum(np.floor(displacements / step + 1e-8).astype(int), maximum)
    states, encoded = np.unique(labels, return_inverse=True)
    counts = np.zeros((len(states), len(states)), dtype=np.int64)
    np.add.at(counts, (encoded[:-1], encoded[1:]), 1)
    if np.any(counts.sum(axis=1) == 0):
        raise ValueError('A fitted phase has no observed outgoing transitions.')
    transition = counts / counts.sum(axis=1, keepdims=True)
    frequencies = np.bincount(encoded[:-1], minlength=len(states)) / (len(encoded) - 1)
    return states * step, transition, frequencies, maximum * step, counts


def sample_phases(states, transition, frequencies, length, rng):
    cumulative = transition.cumsum(axis=1)
    cumulative[:, -1] = 1.0
    phases = np.empty(length, dtype=int)
    phases[0] = rng.choice(len(states), p=frequencies)
    for index, uniform in enumerate(rng.random(length - 1), start=1):
        phases[index] = np.searchsorted(cumulative[phases[index - 1]], uniform, side='right')
    return states[phases]


def displacement_trace(seed, vehicles, burn, h):
    rng = np.random.default_rng(seed)
    incidents = rng.poisson(1.0, (vehicles + burn + 1, 10))
    delays = rng.gamma(incidents, 1.0)
    dispatch = np.arange(len(delays)) * h
    departures = fifo_departures(dispatch, 5.0, delays)[:, 7]
    return departures[burn:] - dispatch[burn:] - 40.0


def main():
    output = Path('runs/rr-delay-kernel-pilot-20261031')
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    h, capacity, vehicles, burn, step = 4.8, 34, 80000, 10000, 0.4
    calibration_seed = 2026103200
    calibration = displacement_trace(calibration_seed, 400000, burn, h)
    states, transition, frequencies, upper, counts = fit_kernel(calibration, step, 0.0005)
    assert np.allclose(transition.sum(axis=1), 1)
    allowed_headways = h + states[None, :] - states[:, None]
    assert np.all(allowed_headways[transition > 0] >= -1e-8)
    np.savez_compressed(output / 'calibrated_kernel.npz', states=states,
                        transition=transition, frequencies=frequencies, counts=counts)
    rows = []
    for replicate in range(5):
        seed = 2026103100 + replicate
        rng = np.random.default_rng(seed + 100000)
        displacement = displacement_trace(seed, vehicles, burn, h)
        quantized = np.minimum(np.floor(displacement / step + 1e-8) * step, upper)
        sampled = sample_phases(states, transition, frequencies, vehicles + burn + 1, rng)[burn:]
        models = {'propagated_fifo': displacement, 'quantized_fifo': quantized,
                  'delay_state_kernel': sampled}
        clocks = {}
        for name, phases in models.items():
            headways = h + np.diff(phases)
            assert headways.min() >= -1e-8
            headways = np.maximum(0, headways)
            assert np.isclose(headways.sum(), vehicles * h + phases[-1] - phases[0])
            clocks[name] = np.cumsum(headways)
        rate = 0.97 * capacity / h
        end = max(clock[-1] for clock in clocks.values())
        arrivals = np.sort(rng.uniform(0, end, rng.poisson(rate * end)))
        for name, clock in clocks.items():
            arrivals_by_service = np.diff(np.r_[0, np.searchsorted(arrivals, clock, side='right')])
            backlog = residual_queues(arrivals_by_service, capacity)
            rows.append({'replicate': replicate, 'seed': seed, 'model': name,
                         'mean_residual_queue': float(backlog[vehicles // 2:].mean()),
                         'mean_headway': float(clock[-1] / vehicles),
                         'terminal_backlog': int(backlog[-1])})
        print(f'pilot replication {replicate + 1}/5', flush=True)
    with (output / 'replications.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    protocol = {'calibration_seed': calibration_seed, 'calibration_vehicles': 400000,
                'independent_evaluation_seeds': list(range(2026103100, 2026103105)),
                'evaluation_vehicles': vehicles, 'burn': burn, 'h': h, 'capacity': capacity,
                'load_ratio': 0.97, 'grid_step': step, 'upper_displacement': upper,
                'capped_calibration_fraction': float(np.mean(calibration > upper)),
                'states': len(states), 'elapsed_seconds': time.perf_counter() - start,
                'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'status': 'pilot only: empirical first-order phase closure, five evaluation replications; no stationary queue solver, grid/tail convergence or route-capacity validation yet'}
    (output / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    (output / Path(__file__).name).write_text(Path(__file__).read_text())
    print(json.dumps(protocol))
    for name in ['propagated_fifo', 'quantized_fifo', 'delay_state_kernel']:
        values = [r['mean_residual_queue'] for r in rows if r['model'] == name]
        print(name, np.mean(values))


if __name__ == '__main__':
    main()
