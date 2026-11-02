"""Independent upstream calibration, marked-phase queue and paired order controls."""

import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np

from delay_stationary import stationary_queue
from route_validation import (CalendarPassengers, incident_delays,
                              simulate_route)
from run_route_suite import demand
from validate_service_revision import residual_queues, write_rows


def marked_kernel(displacements, capacities, step=.8, tail=.0005, physical_capacity=50):
    upper = int(np.floor(np.quantile(displacements, 1 - tail) / step))
    labels = np.minimum(np.floor(displacements / step + 1e-8).astype(int), upper)
    bins, encoded = np.unique(labels, return_inverse=True)
    counts = np.zeros((physical_capacity + 1, len(bins), len(bins)), dtype=np.int64)
    np.add.at(counts, (capacities[1:], encoded[:-1], encoded[1:]), 1)
    totals = counts.sum(axis=(0, 2))
    if np.any(totals == 0):
        raise ValueError('An observed phase has no outgoing calibration transitions.')
    marks = counts / totals[None, :, None]
    transition = marks.sum(axis=0)
    return bins * step, transition, marks, upper * step, counts


def queue_metrics(clock, capacities, stream):
    stream.through(clock[-1])
    cumulative = np.searchsorted(stream.times, clock, side='right')
    counts = np.diff(np.r_[0, cumulative])
    residual = residual_queues(counts, capacities)
    queue = np.r_[0, residual[:-1]] + counts
    begin, end = len(clock) // 2, 3 * len(clock) // 4
    first, stop = cumulative[begin - 1], cumulative[end - 1]
    boarded = cumulative - residual
    served_stop = max(first, min(stop, int(boarded[-1])))
    indices = np.arange(first, served_stop)
    service = np.searchsorted(boarded, indices + 1, side='left')
    waits = clock[service] - stream.times[indices]
    return {'mean_residual': float(residual[begin:].mean()),
            'mean_queue': float(queue[begin:].mean()), 'variance_queue': float(queue[begin:].var()),
            'mean_wait': float(waits.mean()), 'variance_wait': float(waits.var()),
            'cohort_passengers': int(stop - first), 'cohort_served': len(waits),
            'censored_fraction': float(1 - len(waits) / (stop - first)),
            'mean_headway': float(np.diff(clock)[begin - 1:].mean())}, residual


def sample_marked(states, transition, marks, phase, number, rng):
    cumulative = transition.cumsum(axis=1)
    cumulative[:, -1] = 1.
    capacities = np.empty(number, dtype=int)
    displacement = np.empty(number + 1)
    current = rng.choice(len(states), p=phase)
    displacement[0] = states[current]
    for k in range(number):
        following = np.searchsorted(cumulative[current], rng.random(), side='right')
        cdf = marks[:, current, following].cumsum() / transition[current, following]
        cdf[-1] = 1.
        capacities[k] = np.searchsorted(cdf, rng.random(), side='right')
        displacement[k + 1] = states[following]
        current = following
    return displacement, capacities


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--route-run', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    for name in ['validate_route_phase.py', 'delay_stationary.py', 'route_validation.py',
                 'run_route_suite.py', 'validate_service_revision.py', 'service_dynamics.py']:
        shutil.copy2(name, args.output / name)
    protocol = {'status': 'frozen before calibration and independent evaluation',
        'calibration_incident_seed': 2026103800, 'calibration_passenger_seed': 2026403800,
        'calibration_vehicles': 400000, 'calibration_burn': 100000,
        'fit_inputs': 'only exogenous displacement and available capacity at station9; no local queue fitting',
        'central_grid': .8, 'central_tail': .0005, 'queue_cutoffs': [1024, 2048],
        'grid_refinement': .4, 'range_refinement_tail': .00005,
        'route_run': str(args.route_run), 'evaluation_seeds': list(range(2026103600, 2026103620)),
        'headway': 4.8, 'physical_capacity': 50, 'load': .97, 'station': 9,
        'control_burn': 10000, 'measurement': 'last half; arrival cohort middle quarter followed to final service',
        'controls': ['chronological', 'jointly_permuted', 'capacity_permuted', 'headway_permuted',
                     'independently_permuted', 'quantized_chronological', 'marked_phase_simulation'],
        'shared_arrivals': 'the same calendar-time local Poisson stream as the full physical route',
        'replay_check': 'chronological residuals must equal the physical route pathwise',
        'scope': 'empirical first-order displacement closure with edge capacity marks; no exact full-route projection claim',
        'accuracy_gate': 'report all four paper metrics; 10 percent relative-error gate at this stated operating point'}
    (args.output / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    start = time.perf_counter()
    rates, p = demand(.97)
    calibration = simulate_route(rates, p, incident_delays(2026103800, 400000, 10), 2026403800)
    displacement = calibration.starts[100000:, 8] - calibration.origin[100000:] - 45.
    available = calibration.capacities[100000:, 8]
    np.savez_compressed(args.output / 'calibration_inputs.npz', displacement=displacement,
                        available_capacity=available)
    del calibration
    predictions = []
    for name, step, tail, cutoff in [('central', .8, .0005, 1024),
        ('queue_refinement', .8, .0005, 2048), ('grid_refinement', .4, .0005, 1024),
        ('range_refinement', .8, .00005, 1024)]:
        states, transition, marks, upper, counts = marked_kernel(displacement, available, step, tail)
        result = stationary_queue(states, transition, 4.8, rates[8], 50, cutoff,
                                   capacity_kernel=marks)
        joint, phase = result.pop('joint'), result.pop('phase')
        result.update(model=name, grid=step, tail=tail, cutoff=cutoff, upper_displacement=upper,
                      phases=len(states), capped_fraction=float(np.mean(displacement > upper)))
        predictions.append(result)
        if name == 'central':
            central = states, transition, marks, phase, upper
            np.savez_compressed(args.output / 'central_kernel.npz', states=states,
                transition=transition, marks=marks, counts=counts, joint=joint, phase=phase)
        (args.output / 'predictions.json').write_text(json.dumps(predictions, indent=2) + '\n')
        print('stationary prediction', name, result['mean_residual'], flush=True)
    states, transition, marks, phase, upper = central
    rows = []
    for rep in range(20):
        trace = np.load(args.route_run / 'traces' / f'zero-load097-rep{rep:02d}-service.npz')
        clock = trace['starts'][:, 8]
        capacity = trace['capacities'][:, 8]
        length, burn = len(clock), 10000
        delta = clock - np.arange(length) * 4.8 - 45.
        rng = np.random.default_rng(2026603600 + rep)
        intervals = np.diff(np.r_[0., clock])
        suffix = intervals[burn:]
        pair_order = rng.permutation(len(suffix))
        independent_order = rng.permutation(len(suffix))
        models = {'chronological': (clock, capacity),
            'jointly_permuted': (np.r_[clock[:burn], clock[burn - 1] + suffix[pair_order].cumsum()],
                                np.r_[capacity[:burn], capacity[burn:][pair_order]]),
            'capacity_permuted': (clock, np.r_[capacity[:burn], capacity[burn:][pair_order]]),
            'headway_permuted': (np.r_[clock[:burn], clock[burn - 1] + suffix[pair_order].cumsum()], capacity),
            'independently_permuted': (np.r_[clock[:burn], clock[burn - 1] + suffix[pair_order].cumsum()],
                                      np.r_[capacity[:burn], capacity[burn:][independent_order]]),
            'quantized_chronological': (np.arange(length) * 4.8 + 45.
                + np.minimum(np.floor(delta / .8 + 1e-8) * .8, upper), capacity)}
        sampled, sampled_capacity = sample_marked(states, transition, marks, phase, length - burn, rng)
        phase_clock = clock[burn - 1] + np.cumsum(4.8 + np.diff(sampled))
        models['marked_phase_simulation'] = (np.r_[clock[:burn], phase_clock],
                                            np.r_[capacity[:burn], sampled_capacity])
        stream = CalendarPassengers(rates[8], p[8], 2026103600 + rep + 300000 + 8000)
        stream.through(max(t[-1] for t, _ in models.values()))
        for name, (service_clock, batches) in models.items():
            assert np.all(np.diff(service_clock) >= -1e-8)
            metrics, residual = queue_metrics(service_clock, batches, stream)
            if name == 'chronological':
                np.testing.assert_array_equal(residual, trace['residuals'][:, 8])
            metrics.update(model=name, replicate=rep, seed=2026103600 + rep)
            rows.append(metrics)
        write_rows(args.output / 'evaluation_replications.csv', rows)
        print('paired capacity evaluation', rep + 1, flush=True)
    (args.output / 'status.json').write_text(json.dumps({'status': 'completed',
        'evaluation_replications': 20, 'elapsed_seconds': time.perf_counter() - start}, indent=2) + '\n')


if __name__ == '__main__':
    main()
