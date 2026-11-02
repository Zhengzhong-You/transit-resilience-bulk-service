"""Frozen paired full-route, dwell, fleet, destination and incident validation."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import time

import numpy as np

from route_validation import (cut_loads, destination_law, incident_delays,
                              matched_destination_law, simulate_route, arrival_cohort_summaries)
from validate_service_revision import write_rows


def demand(load, headway=4.8, capacity=50):
    p = destination_law()
    rates = np.r_[np.ones(9), 0.]
    rates *= load * capacity / cut_loads(rates, p, headway).max()
    return rates, p


def one_task(task):
    output = Path(task['output'])
    case, rep = task['case'], task['replicate']
    seed = task['seed_base'] + rep
    rates, p = demand(case['load'])
    if case.get('od'):
        p = matched_destination_law(rates, p)
    count = task['vehicles']
    delays = incident_delays(seed, count, 10, case.get('gamma', .2),
                             case.get('duration', 'exponential'))
    kwargs = {}
    if case.get('fleet'):
        kwargs.update(fleet=case['fleet'], return_delays=incident_delays(seed + 100000, count, 10))
    if case.get('dwell') == 'endogenous':
        kwargs['dwell_coefficients'] = (.1, .02, .01)
    elif case.get('dwell') in ['constant', 'exogenous']:
        calibration = np.load(output / f"dwell_calibration_load{int(case['load'] * 100):03d}.npz")['dwell']
        if case['dwell'] == 'constant':
            prescribed = np.broadcast_to(calibration.mean(axis=0), (count, 10))
        else:
            rng = np.random.default_rng(seed + 200000)
            prescribed = np.column_stack([rng.choice(calibration[:, n], count) for n in range(10)])
        kwargs['prescribed_dwell'] = prescribed
    initial = np.where(rates > 0, case.get('initial', 0), 0)
    start = time.perf_counter()
    result = simulate_route(rates, p, delays, seed + 300000,
                            initial_backlog=initial, passing=case.get('passing', False), **kwargs)
    rows = []
    horizons = ([count // 16, count // 4, count] if case['group'] == 'boundary' else [count])
    # Prefix views retain the actual initial state; no reinitialization at shorter horizons.
    for horizon in sorted(set(horizons)):
        from route_validation import RouteResult
        prefix = RouteResult(**{name: value[:horizon] for name, value in vars(result).items()})
        cohorts = arrival_cohort_summaries(prefix, rates, p, seed + 300000, initial)
        for row, cohort in zip(prefix.summaries(4.8), cohorts):
            row.update(cohort)
            n = row['station'] - 1
            row.update(case=case['name'], group=case['group'], load=case['load'],
                       replicate=rep, incident_seed=seed, passenger_seed=seed + 300000,
                       vehicles=horizon, initial_backlog=case.get('initial', 0),
                       nominal_cut_load=float(cut_loads(rates, p, 4.8)[n] / 50),
                       arrival_rate=float(rates[n]))
            rows.append(row)
    name = f"{case['name']}-rep{rep:02d}"
    write_rows(output / 'replications' / (name + '.csv'), rows)
    indices = np.unique(np.r_[0, np.linspace(0, count - 1, 1001).astype(int)])
    np.savez_compressed(output / 'traces' / (name + '.npz'), vehicle=indices,
        residual=result.residuals[indices], origin=result.origin[indices],
        starts=result.starts[indices], returned=result.returned[indices])
    if case['name'] == 'zero-load097':
        np.savez_compressed(output / 'traces' / (name + '-service.npz'),
            starts=result.starts, capacities=result.capacities,
            residuals=result.residuals, rates=rates, destinations=p)
    elapsed = time.perf_counter() - start
    (output / 'replications' / (name + '.json')).write_text(json.dumps(
        {'case': case, 'replicate': rep, 'elapsed_seconds': elapsed,
         'status': 'completed', 'boarded_equals_alighted': True}, indent=2) + '\n')
    return name, elapsed


def cases():
    result = []
    for load in [.80, .90, .97, 1., 1.01, 1.02]:
        for initial in [0, 1000]:
            result.append({'name': f'boundary-{load:.2f}-initial{initial}',
                           'group': 'boundary', 'load': load, 'initial': initial})
    for load in [.8, .97]:
        for dwell in ['zero', 'constant', 'exogenous', 'endogenous']:
            result.append({'name': f'{dwell}-load{int(load * 100):03d}',
                           'group': 'dwell', 'load': load, 'dwell': dwell})
    for fleet in [25, 30, 40]:
        for dwell in ['zero', 'endogenous']:
            result.append({'name': f'fleet{fleet}-{dwell}', 'group': 'fleet',
                           'load': .97, 'fleet': fleet, 'dwell': dwell})
    for gamma in [0., .02, .2]:
        for duration in ['deterministic', 'exponential', 'lognormal']:
            # At gamma=0 all laws are exactly the same deterministic experiment.
            if gamma == 0 and duration != 'exponential':
                continue
            result.append({'name': f'incident-{gamma:.2f}-{duration}', 'group': 'incidents',
                           'load': .97, 'gamma': gamma, 'duration': duration})
    result.extend([{'name': 'origin-dependent-od', 'group': 'destinations', 'load': .97, 'od': True},
                   {'name': 'passing-allowed', 'group': 'passing', 'load': .97, 'passing': True}])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--vehicles', type=int, default=80000)
    parser.add_argument('--replications', type=int, default=20)
    parser.add_argument('--workers', type=int, default=24)
    parser.add_argument('--pilot', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    for directory in ['replications', 'traces', 'source']:
        (output / directory).mkdir()
    for name in ['route_validation.py', 'run_route_suite.py', 'validate_service_revision.py',
                 'service_dynamics.py', 'test_route_validation.py']:
        shutil.copy2(name, output / 'source' / name)
    start = time.perf_counter()
    selected = cases()
    if args.pilot:
        selected = [c for c in selected if c['name'] in
                    ['zero-load097', 'endogenous-load097', 'fleet25-endogenous', 'passing-allowed']]
    protocol = {'status': 'frozen before calibration/evaluation',
        'python': platform.python_version(), 'numpy': np.__version__, 'platform': platform.platform(),
        'host': platform.node(), 'pid': os.getpid(), 'headway': 4.8, 'capacity': 50,
        'stations': 10, 'normal_segment_minutes': 5., 'calibration_vehicles_per_load': 40000,
        'calibration_incident_seeds': [2026103500, 2026103501],
        'calibration_passenger_seed_offset': 300000,
        'evaluation_incident_seeds': list(range(2026103450 if args.pilot else 2026103600,
                                               (2026103450 if args.pilot else 2026103600) + args.replications)),
        'evaluation_passenger_seed_offset': 300000,
        'replications': args.replications, 'vehicles': args.vehicles, 'workers': args.workers,
        'measurement': 'last half of each stated horizon; waiting is to gated boarding start',
        'initialization': 'empty unless boundary case specifies 1000 passengers at every positive-demand station',
        'incident_clock': 'normal travel only, independent across segment/run; equal mean duration1',
        'lognormal_sigma': float(np.sqrt(np.log(2))),
        'dwell': 'gated service start=max(upstream departure+travel+incident, predecessor berth release); .1+.02*board+.01*alight minutes',
        'dwell_controls': 'constant station means or iid empirical station draws from separate endogenous calibration at each corresponding load',
        'fleet': 'round-robin vehicle reuse only after explicit ten-segment FIFO empty return trip; no terminal layover',
        'od': '2x2 exchanges preserve origin rates, destination rates and all nominal cut loads',
        'passing': 'zero-dwell free trajectories, boarding in chronological station arrival order',
        'waiting_scope': 'both last-half boarded observations and arrivals in the middle calendar quarter, followed through the final horizon; unserved cohort members explicitly censored; no stationary moments claimed in unstable cases',
        'uncertainty': 'independent replication unit; paired differences use common seeds',
        'cases': selected, 'pilot': args.pilot,
        'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (output / 'source').iterdir()}}
    (output / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    for load, calibration_seed in zip([.8, .97], [2026103500, 2026103501]):
        rates, p = demand(load)
        calibration = simulate_route(rates, p, incident_delays(calibration_seed, 40000, 10),
                                     calibration_seed + 300000,
                                     dwell_coefficients=(.1, .02, .01))
        np.savez_compressed(output / f'dwell_calibration_load{int(load * 100):03d}.npz',
                            dwell=calibration.dwell[20000:])
        del calibration
    tasks = [{'case': case, 'replicate': rep, 'vehicles': args.vehicles, 'output': str(output),
              'seed_base': 2026103450 if args.pilot else 2026103600}
             for case in selected for rep in range(args.replications)]
    print('frozen tasks', len(tasks), flush=True)
    completed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(one_task, task) for task in tasks]
        for future in as_completed(futures):
            name, elapsed = future.result()
            completed += 1
            state = {'status': 'running', 'completed': completed, 'total': len(tasks),
                     'last_completed': name, 'elapsed_seconds': time.perf_counter() - start}
            (output / 'status.json').write_text(json.dumps(state, indent=2) + '\n')
            print(completed, '/', len(tasks), name, round(elapsed, 2), flush=True)
    state['status'] = 'completed'
    (output / 'status.json').write_text(json.dumps(state, indent=2) + '\n')
    print(json.dumps(state), flush=True)


if __name__ == '__main__':
    main()
