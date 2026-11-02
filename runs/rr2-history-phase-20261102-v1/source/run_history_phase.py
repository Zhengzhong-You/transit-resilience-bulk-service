"""History-augmented phase for every validation cell, from the frozen calibration runs.

Each calibration run is re-simulated from its recorded seeds to recover the upstream residual
queues; the stored displacement and capacity arrays remain the fitting data and must be reproduced.
All kernels are fitted and frozen before the fitted-chain checks, which reuse the evaluation seeds,
vehicle numbers and calendar windows of the validation sample. No physical validation output enters
the fit.
"""

import os
for _thread_option in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[_thread_option] = '1'

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import platform
import shutil
import time

import numpy as np
import pandas as pd
from scipy.sparse import load_npz, save_npz

from evaluate_frozen_route_phase import sample_joint
from history_phase import (BLOCKS, CLASS_LADDER, GRID_DIVISOR, STATE_LIMIT, choose_resolution,
                           fit_history_phase, net_input_calibration, net_input_chain, upstream_backlog)
from route_validation import CalendarPassengers
from run_revision_scan import METRICS, save_json, simulate
from sparse_stationary import stationary_queue_sparse
from validate_route_phase import queue_metrics


def recover(task):
    case, index, scan, directory = task
    directory = Path(directory)
    directory.mkdir(parents=True)
    stored = np.load(Path(scan) / 'calibration' / case['name'] / 'calibration_inputs.npz')
    start = time.perf_counter()
    route = simulate(case, 2026110200 + index, 400000)
    burn = 100000
    displacement = route.starts[burn:, :9] - route.origin[burn:, None] - 5 * np.arange(1, 10)
    step = case['headway'] / GRID_DIVISOR
    check = dict(capacities_equal=bool(np.array_equal(route.capacities[burn:, :9], stored['available_capacity'])),
                 maximum_displacement_difference=float(np.abs(displacement - stored['displacement']).max()),
                 grid_classes_equal=bool(np.array_equal(np.floor(displacement / step + 1e-8),
                                                        np.floor(stored['displacement'] / step + 1e-8))))
    if not (check['capacities_equal'] and check['grid_classes_equal']
            and check['maximum_displacement_difference'] < 1e-8):
        raise ArithmeticError(f'The calibration re-simulation differs for {case["name"]}.')
    np.savez_compressed(directory / 'upstream_residuals.npz',
                        residuals=route.residuals[burn:, :9].astype(np.int32))
    save_json(directory / 'recovery.json', dict(case=case['name'], index=index,
        incident_seed=2026110200 + index, seconds=time.perf_counter() - start, **check))
    return case['name']


def fit_cell(task):
    case, scan, directory, n = task
    directory = Path(directory)
    stored = np.load(Path(scan) / 'calibration' / case['name'] / 'calibration_inputs.npz')
    displacement, capacity = stored['displacement'][:, n - 1], stored['available_capacity'][:, n - 1]
    backlog = upstream_backlog(np.load(directory / 'upstream_residuals.npz')['residuals'],
                               case['destinations'], n)
    rate, h = case['rates'][n - 1], case['headway']
    row, history = dict(case=case['name'], station=n, status='failed'), []
    start = time.perf_counter()
    try:
        step, classes, _ = choose_resolution(displacement, capacity, backlog, h)
        fitted = fit_history_phase(displacement, capacity, backlog, step, classes)
        fitting_seconds = time.perf_counter() - start
        delta, seats, transition = fitted['displacement'], fitted['capacity'], fitted['transition']
        previous, joint = None, None
        for cutoff in [512, 1024, 2048, 4096, 8192]:
            initial = None if joint is None else np.vstack([joint, np.zeros((cutoff + 1 - len(joint), len(delta)))])
            result = stationary_queue_sparse(delta, transition, h, rate, seats, cutoff,
                                             tolerance=1e-10, initial=initial)
            values = {k: result[k] for k in METRICS}
            change = (max(abs(values[k] - previous[k]) / max(1., abs(values[k])) for k in METRICS)
                      if previous is not None else None)
            history.append(dict(cutoff=cutoff, change=change, **values,
                boundary=result['upper_boundary_probability'], balance=result['balance_residual_l1'],
                iterations=result['iterations'], seconds=result['elapsed_seconds']))
            if previous is not None and change < .005 and result['upper_boundary_probability'] < 1e-10:
                break
            previous, joint = values, result['joint']
        else:
            raise RuntimeError('Queue-cutoff refinement gate was not met.')
        phase = result.pop('phase')
        result.pop('joint')
        save_npz(directory / f'station{n}-transition.npz', transition)
        np.savez_compressed(directory / f'station{n}-phase.npz', displacement=delta, capacity=seats,
                            backlog_class=fitted['backlog_class'], phase=phase)
        # Calibration-only diagnostic: per-vehicle net-input variance over vehicle blocks.
        target = net_input_calibration(displacement, capacity, rate, h)
        refined = net_input_chain(delta, seats, transition, phase, rate, h)
        old = Path(scan) / 'calibration' / case['name']
        z = np.load(old / f'station{n}-phase.npz')
        first = net_input_chain(z['displacement'], z['capacity'], load_npz(old / f'station{n}-transition.npz').tocsr(),
                                z['phase'], rate, h)
        row.update(result, status='completed', grid=step, backlog_classes=classes, states=len(delta),
            edges=transition.nnz, removed_transition_fraction=fitted['removed_transition_fraction'],
            upper_displacement=fitted['upper_displacement'], cutoff=cutoff,
            fitting_seconds=fitting_seconds, refinement=history,
            solve_seconds_total=sum(x['seconds'] for x in history),
            net_input_ratio_history={str(m): refined[m] / target[m] for m in BLOCKS},
            net_input_ratio_first_order={str(m): first[m] / target[m] for m in BLOCKS})
    except (ValueError, RuntimeError, ArithmeticError, np.linalg.LinAlgError) as error:
        row['error'] = str(error)
        row['refinement'] = history
    row['total_fit_solve_seconds'] = time.perf_counter() - start
    save_json(directory / f'station{n}-prediction.json', row)
    return f'{case["name"]}-station{n}'


def check_chain(task):
    case, index, rep, n, directory, protocol, vehicles = task
    directory = Path(directory)
    seed = protocol['evaluation_seed_base'] + rep
    z = np.load(directory / f'station{n}-phase.npz')
    transition = load_npz(directory / f'station{n}-transition.npz').tocsr()
    delta, seats = sample_joint(z['displacement'], z['capacity'], transition, z['phase'], vehicles,
                                np.random.default_rng(seed + 200000000 + 1000 * index + n))
    h = case['headway']
    stream = CalendarPassengers(case['rates'][n - 1], np.array(case['destinations'][n - 1]),
                                seed + 300000 + (n - 1) * 1000)
    values, _ = queue_metrics(np.arange(vehicles) * h + 5 * n + delta, seats, stream, scheduled_headway=h,
                              observation_window=protocol['observation_window'],
                              cohort_window=protocol['cohort_window'])
    return dict(case=case['name'], station=n, replicate=rep, model='history_phase_simulation',
                seed=seed, vehicles=vehicles, **values)


def execute_pool(function, tasks, workers, output, stage):
    results, completed = [], 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(function, task) for task in tasks]):
            results.append(future.result())
            completed += 1
            save_json(output / 'status.json', dict(stage=stage, status='running', completed=completed,
                                                   total=len(tasks)))
            print(stage, completed, '/', len(tasks), flush=True)
    return results


def physical_vehicles(scan, repair):
    """Vehicle numbers of the physical validation replications, with the repaired profile."""
    rows = pd.concat([pd.read_csv(p) for p in (scan / 'replications').glob('*.csv')], ignore_index=True)
    rows = rows[rows.model.eq('physical') & rows.phase_validation & ~rows.case.eq('original-demand1')]
    repaired = pd.concat([pd.read_csv(p) for p in (repair / 'replications').glob('*.csv')], ignore_index=True)
    rows = pd.concat([rows, repaired[repaired.model.eq('physical')]], ignore_index=True)
    vehicles = rows.groupby(['case', 'replicate']).vehicles.agg(['min', 'max'])
    assert (vehicles['min'] == vehicles['max']).all() and len(vehicles) == 19 * 20
    return vehicles['min'].to_dict()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scan', type=Path, default=Path('runs/rr2-scan-20261101-v1'))
    parser.add_argument('--repair', type=Path, default=Path('runs/rr2-scan-repair-20261101-v1'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--profiles', nargs='*', help='restrict to these validation profiles (smoke tests)')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    for name in ['source', 'calibration']:
        (output / name).mkdir()
    names = ['run_history_phase.py', 'history_phase.py', 'sparse_stationary.py', 'sparse_phase_queue.py',
             'run_revision_scan.py', 'route_validation.py', 'validate_route_phase.py',
             'evaluate_frozen_route_phase.py', 'revision_inputs.py']
    for name in names:
        shutil.copy2(name, output / 'source' / name)
    scan = args.scan.resolve()
    scan_protocol = json.loads((scan / 'protocol.json').read_text())
    profiles = [(i, c) for i, c in enumerate(scan_protocol['cases']) if c['phase_validation']
                and (not args.profiles or c['name'] in args.profiles)]
    protocol = dict(status='frozen before fitting and fitted-chain checks', scan=str(args.scan),
        repair=str(args.repair), profiles=[c['name'] for _, c in profiles],
        phase='displacement class, available capacity and upstream backlog class',
        backlog='expected passengers left at upstream stations after the vehicle whose destination lies beyond the station',
        backlog_classes='class 0 for an empty backlog; positive values split at equal-frequency calibration quantiles',
        grid_divisor=GRID_DIVISOR, class_ladder=list(CLASS_LADDER), state_limit=STATE_LIMIT,
        resolution_rule='grid h/16 and the most backlog classes in the ladder with at most state_limit phases',
        kernel='empirical transitions restricted to the largest strongly connected class',
        numerical_refinement='successive queue cutoffs 512,1024,2048,4096,8192 with warm starts; four-moment change below 0.5 percent and boundary mass below 1e-10',
        fitted_chain_check='evaluation seeds, vehicle numbers and calendar windows of the validation sample',
        python=platform.python_version(), numpy=np.__version__, platform=platform.platform(),
        source_sha256={name: hashlib.sha256((output / 'source' / name).read_bytes()).hexdigest() for name in names})
    save_json(output / 'protocol.json', protocol)
    start = time.perf_counter()
    execute_pool(recover, [(c, i, str(scan), str(output / 'calibration' / c['name'])) for i, c in profiles],
                 args.workers, output, 'recovery')
    # Incident-rich downstream cells have the largest phases and start first to balance the pool.
    cells = sorted([(c, n) for _, c in profiles for n in range(1, 10)], key=lambda x: (-x[0]['gamma'], -x[1]))
    execute_pool(fit_cell, [(c, str(scan), str(output / 'calibration' / c['name']), n) for c, n in cells],
                 args.workers, output, 'fit')
    for _, c in profiles:
        directory = output / 'calibration' / c['name']
        rows = [json.loads((directory / f'station{n}-prediction.json').read_text()) for n in range(1, 10)]
        save_json(directory / 'predictions.json', rows)
        for n in range(1, 10):
            (directory / f'station{n}-prediction.json').unlink()
    freeze = {str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted((output / 'calibration').rglob('*')) if p.is_file()}
    save_json(output / 'frozen-kernels.json', freeze)
    print('ALL history kernels frozen; starting fitted-chain checks', flush=True)
    vehicles = physical_vehicles(scan, args.repair.resolve())
    tasks = [(c, i, rep, n, str(output / 'calibration' / c['name']), scan_protocol, int(vehicles[(c['name'], rep)]))
             for i, c in profiles for n in range(1, 10) for rep in range(scan_protocol['replications'])
             if json.loads((output / 'calibration' / c['name'] / 'predictions.json').read_text())[n - 1]['status'] == 'completed']
    rows = execute_pool(check_chain, tasks, args.workers, output, 'fitted-chain')
    pd.DataFrame(rows).sort_values(['case', 'station', 'replicate']).to_csv(output / 'fitted-chain-checks.csv', index=False)
    for name, digest in freeze.items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest
    predictions = [r for _, c in profiles for r in json.loads((output / 'calibration' / c['name'] / 'predictions.json').read_text())]
    save_json(output / 'status.json', dict(status='completed', elapsed_seconds=time.perf_counter() - start,
        cells=len(predictions), completed=sum(r['status'] == 'completed' for r in predictions),
        fitted_chain_replications=len(rows)))


if __name__ == '__main__':
    main()
