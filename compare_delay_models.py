"""Use one matched calendar arrival stream to compare all service models."""

import json
import shutil
import time
from pathlib import Path

import numpy as np
from scipy.stats import norm

from delay_stationary import stationary_queue
from explore_delay_state import displacement_trace, sample_phases
from validate_service_revision import residual_queues, write_rows


def main():
    output = Path('runs/rr-delay-independent-20261031')
    output.mkdir(parents=True, exist_ok=False)
    for name in ['compare_delay_models.py', 'service_dynamics.py', 'delay_stationary.py',
                 'explore_delay_state.py', 'validate_service_revision.py']:
        shutil.copy2(name, output / name)
    start = time.perf_counter()
    kernel = np.load('runs/rr-delay-validation-20261031/central_kernel.npz')
    states, transition, frequencies = kernel['states'], kernel['transition'], kernel['frequencies']
    h, capacity, rate, vehicles, burn = 4.8, 34, 0.97 * 34 / 4.8, 80000, 10000
    sigma = 2 * np.sqrt(40 * 0.2)
    positive_mean = h * norm.cdf(h / sigma) + sigma * norm.pdf(h / sigma)
    prediction = stationary_queue(states, transition, h, rate, capacity, 1024)
    np.savez_compressed(output / 'stationary_joint.npz', joint=prediction.pop('joint'),
                        phase=prediction.pop('phase'))
    (output / 'stationary_prediction.json').write_text(json.dumps(prediction, indent=2) + '\n')
    shutil.copy2('runs/rr-delay-validation-20261031/central_kernel.npz', output / 'central_kernel.npz')
    rows = []
    for replicate in range(20):
        seed = 2026103300 + replicate
        rng = np.random.default_rng(seed + 100000)
        displacement = displacement_trace(seed, vehicles, burn, h)
        quantized = np.minimum(np.floor(displacement / 0.4 + 1e-8) * 0.4, states[-1])
        generated = sample_phases(states, transition, frequencies, vehicles + burn + 1, rng)[burn:]
        phase_models = {'propagated_fifo': displacement, 'quantized_fifo': quantized,
                        'delay_state_kernel': generated}
        intervals = {name: h + np.diff(phases) for name, phases in phase_models.items()}
        assert all(values.min() >= -1e-8 for values in intervals.values())
        intervals = {name: np.maximum(values, 0) for name, values in intervals.items()}
        intervals['fifo_permuted'] = rng.permutation(intervals['propagated_fifo'])
        intervals['unscaled_normal'] = np.maximum(0, h + sigma * rng.standard_normal(vehicles))
        intervals['rate_matched_normal'] = intervals['unscaled_normal'] * h / positive_mean
        clocks = {name: np.cumsum(values) for name, values in intervals.items()}
        end = max(clock[-1] for clock in clocks.values())
        arrivals = np.sort(rng.uniform(0, end, rng.poisson(rate * end)))
        for name, clock in clocks.items():
            cumulative = np.searchsorted(arrivals, clock, side='right')
            per_service = np.diff(np.r_[0, cumulative])
            residual = residual_queues(per_service, capacity)
            queue = np.r_[0, residual[:-1]] + per_service
            sample = slice(vehicles // 2, vehicles)
            row = {'replicate': replicate, 'seed': seed, 'model': name,
                   'observation': 'finite horizon only' if name == 'unscaled_normal' else 'stable-case moments',
                   'mean_headway': float(clock[-1] / vehicles),
                   'mean_residual': float(residual[sample].mean()),
                   'variance_residual': float(residual[sample].var(ddof=1)),
                   'mean_queue': float(queue[sample].mean()),
                   'variance_queue': float(queue[sample].var(ddof=1)),
                   'growth_per_time': float((residual[-1] - residual[vehicles // 2 - 1])
                                            / (clock[-1] - clock[vehicles // 2 - 1])),
                   'mean_wait': float('nan'), 'variance_wait': float('nan')}
            if name != 'unscaled_normal':
                boarded = cumulative - residual
                assert np.all(np.diff(np.r_[0, boarded]) >= 0)
                service_indices = np.searchsorted(boarded, np.arange(1, boarded[-1] + 1))
                waits = clock[service_indices] - arrivals[:boarded[-1]]
                assert waits.min() >= -1e-8
                waits = waits[service_indices >= vehicles // 2]
                row.update(mean_wait=float(waits.mean()), variance_wait=float(waits.var(ddof=1)))
            rows.append(row)
        print('matched evaluation', replicate + 1, flush=True)
    write_rows(output / 'evaluation_replications.csv', rows)
    protocol = {'h': h, 'available_capacity': capacity, 'arrival_rate': rate, 'load_ratio': 0.97,
                'station': 8, 'vehicles': vehicles, 'burn_in_service_vehicles': burn,
                'measurement_window': 'last 40000 service intervals', 'evaluation_replications': 20,
                'evaluation_seeds': list(range(2026103300, 2026103320)),
                'shared_input': 'one calendar-time Poisson arrival trace per replication for all six clocks',
                'calibration': 'independent 400000-vehicle service-only trace, seed 2026103200; no queue fitting',
                'phase_grid': 0.4, 'phase_tail_probability': 0.0005, 'queue_cutoff': 1024,
                'normal_positive_part_mean': positive_mean,
                'scope': 'isolated station with deterministic available capacity, zero dwell, exogenous dispatch; does not establish coupled-route or dwell-feedback accuracy',
                'pilot_reuse': 'No pilot or development evaluation seed is used in this final evaluation; kernel and hyperparameters were frozen beforehand.',
                'predeclared_gate': '10 percent held-out residual mean and waiting mean error; all four paper metrics and their uncertainty are also reported',
                'elapsed_seconds': time.perf_counter() - start}
    (output / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    print(json.dumps(protocol), flush=True)


if __name__ == '__main__':
    main()
