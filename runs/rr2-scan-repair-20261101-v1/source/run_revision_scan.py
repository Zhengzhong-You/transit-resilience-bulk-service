"""Independent all-station validation and restored heterogeneous-route policies."""

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
from scipy.sparse import load_npz, save_npz

from delay_stationary import stationary_queue
from evaluate_frozen_route_phase import sample_joint
from revision_inputs import all_scenarios, validation_profiles
from route_validation import CalendarPassengers, incident_delays, simulate_route
from sparse_phase_queue import fit_joint_phase, sparse_phase_transition_operator
from validate_route_phase import queue_metrics
from validate_service_revision import write_rows


METRICS = ['mean_queue', 'variance_queue', 'mean_wait', 'variance_wait']


def save_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def simulate(case, seed, number):
    incidents = incident_delays(seed, number, 10, case['gamma'],
                                duration_seed=seed + 100000000) * case['mean_duration']
    return simulate_route(np.array(case['rates']), np.array(case['destinations']),
        incidents, seed + 300000, headway=case['headway'], capacity=case['capacity'])


def calibrate(task):
    case, index, directory, count, burn = task
    directory = Path(directory)
    directory.mkdir()
    seed = 2026110200 + index
    start = time.perf_counter()
    physical = simulate(case, seed, count)
    elapsed = time.perf_counter() - start
    displacement = physical.starts[burn:, :9] - physical.origin[burn:, None] - 5 * np.arange(1, 10)
    np.savez_compressed(directory / 'calibration_inputs.npz', displacement=displacement,
                        available_capacity=physical.capacities[burn:, :9])
    save_json(directory / 'calibration.json', dict(case=case, incident_seed=seed,
        duration_seed=seed + 100000000, passenger_seed=seed + 300000,
        vehicles=count, burn=burn, calibration_seconds=elapsed,
        fitted_fields=['displacement', 'available_capacity'], local_queue_fitting=False))
    return case['name']


def fit_profile(task):
    case, directory, stations, pilot = task
    directory = Path(directory)
    data = np.load(directory / 'calibration_inputs.npz')
    rows = []
    for n in stations:
        start = time.perf_counter()
        row = dict(case=case['name'], station=n, status='failed')
        history = []
        try:
            delta, capacity, sparse, upper = fit_joint_phase(data['displacement'][:, n - 1],
                data['available_capacity'][:, n - 1], step=case['headway'] / 4, tail=.0005)
            fitting_seconds = time.perf_counter() - start
            transition = sparse.toarray()
            previous, result = None, None
            cutoffs = [256, 512, 1024] if pilot else [512, 1024, 2048, 4096, 8192]
            for cutoff in cutoffs:
                result = stationary_queue(delta, transition, case['headway'], case['rates'][n - 1],
                    capacity, cutoff, tolerance=1e-7 if pilot else 1e-10,
                    operator_factory=sparse_phase_transition_operator)
                values = {k: result[k] for k in METRICS}
                change = (max(abs(values[k] - previous[k]) / max(1., abs(values[k]))
                              for k in METRICS) if previous is not None else None)
                history.append(dict(cutoff=cutoff, change=change, **values,
                    boundary=result['upper_boundary_probability'], balance=result['balance_residual_l1'],
                    iterations=result['iterations'], seconds=result['elapsed_seconds']))
                if previous is not None and change < .005 and result['upper_boundary_probability'] < 1e-10:
                    break
                previous = values
            else:
                raise RuntimeError('Queue-cutoff refinement gate was not met.')
            phase, joint = result.pop('phase'), result.pop('joint')
            save_npz(directory / f'station{n}-transition.npz', sparse)
            np.savez_compressed(directory / f'station{n}-phase.npz', displacement=delta,
                                capacity=capacity, phase=phase)
            # A direct stationary array is retained for the two main reference cells.
            if (case['name'], n) in [('uniform-e097-g20', 9), ('heterogeneous-e097', 3)]:
                np.savez_compressed(directory / f'station{n}-stationary.npz', joint=joint)
            row.update(result, status='completed', grid=case['headway'] / 4,
                tail=.0005, upper_displacement=upper, states=len(delta), edges=sparse.nnz,
                cutoff=cutoff, fitting_seconds=fitting_seconds, refinement=history,
                solve_seconds_total=sum(x['seconds'] for x in history))
        except (ValueError, RuntimeError, ArithmeticError, np.linalg.LinAlgError) as error:
            row['error'] = str(error)
            row['refinement'] = history
        row['total_fit_solve_seconds'] = time.perf_counter() - start
        rows.append(row)
        save_json(directory / 'predictions.json', rows)
    return case['name']


def evaluate(task):
    case, index, rep, output, protocol = task
    output = Path(output)
    seed = protocol['evaluation_seed_base'] + rep
    h = case['headway']
    obs, cohort = protocol['observation_window'], protocol['cohort_window']
    count = int(np.ceil(obs[1] / h)) + 2
    history = []
    while True:
        start = time.perf_counter()
        physical = simulate(case, seed, count)
        simulation_seconds = time.perf_counter() - start
        members, censored = [], []
        for n in protocol['stations']:
            stream = CalendarPassengers(case['rates'][n - 1], np.array(case['destinations'][n - 1]),
                                        seed + 300000 + (n - 1) * 1000)
            m, residual = queue_metrics(physical.starts[:, n - 1], physical.capacities[:, n - 1],
                stream, scheduled_headway=h, observation_window=obs, cohort_window=cohort)
            np.testing.assert_array_equal(residual, physical.residuals[:, n - 1])
            members.append(m['cohort_passengers'])
            censored.append(m['censored_fraction'])
        history.append(dict(vehicles=count, simulation_seconds=simulation_seconds,
                            maximum_censored_fraction=max(censored)))
        if max(censored) == 0:
            break
        if len(history) >= 4:
            raise RuntimeError(f'Uncompleted common cohort in {case["name"]}.')
        # Separate incident streams guarantee this extends the same input path.
        count *= 2
    prediction_dir = output / 'calibration' / case['name']
    predictions = ({x['station']: x for x in json.loads((prediction_dir / 'predictions.json').read_text())}
                   if case['phase_validation'] else {})
    rows = []
    for n in protocol['stations']:
        clock, seats = physical.starts[:, n - 1], physical.capacities[:, n - 1]
        stream = CalendarPassengers(case['rates'][n - 1], np.array(case['destinations'][n - 1]),
                                    seed + 300000 + (n - 1) * 1000)
        models = {'physical': (clock, seats)}
        prediction = predictions.get(n)
        if case['phase_validation']:
            burn = min(10000, len(clock) // 8)
            intervals = np.diff(np.r_[0., clock])
            rng = np.random.default_rng(seed + 210000000 + 1000 * index + n)
            order, separate = rng.permutation(len(clock) - burn), rng.permutation(len(clock) - burn)
            shuffled = np.r_[clock[:burn], clock[burn - 1] + intervals[burn:][order].cumsum()]
            shuffled_seats = np.r_[seats[:burn], seats[burn:][order]]
            models.update(pairs_permuted=(shuffled, shuffled_seats),
                capacity_permuted=(clock, shuffled_seats), headways_permuted=(shuffled, seats),
                separately_permuted=(shuffled, np.r_[seats[:burn], seats[burn:][separate]]))
            if prediction['status'] == 'completed':
                z = np.load(prediction_dir / f'station{n}-phase.npz')
                transition = load_npz(prediction_dir / f'station{n}-transition.npz').tocsr()
                delta, cap = sample_joint(z['displacement'], z['capacity'], transition,
                    z['phase'], count, np.random.default_rng(seed + 200000000 + 1000 * index + n))
                models['phase_simulation'] = (np.arange(count) * h + 5 * n + delta, cap)
        stream.through(max(v[0][-1] for v in models.values()))
        for model, (service, capacity) in models.items():
            start = time.perf_counter()
            values, residual = queue_metrics(service, capacity, stream, scheduled_headway=h,
                                             observation_window=obs, cohort_window=cohort)
            elapsed = time.perf_counter() - start
            if values['cohort_passengers'] != members[protocol['stations'].index(n)]:
                raise ArithmeticError('The calendar arrival cohort changed across models.')
            sample = (service > obs[0]) & (service <= obs[1])
            chosen = np.flatnonzero(sample)
            row = dict(case=case['name'], aliases='|'.join(case['aliases']),
                phase_validation=case['phase_validation'], group=case['group'],
                maximum_cut_load=case['maximum_cut_load'], station=n, replicate=rep,
                model=model, seed=seed, vehicles=count, **values,
                local_replay_seconds=elapsed,
                queue_drift_per_minute=float((residual[chosen[-1]] - residual[chosen[0]])
                                             / (service[chosen[-1]] - service[chosen[0]])),
                mean_available_capacity=float(capacity[sample].mean()),
                variance_headway=float(np.diff(service)[sample[1:]].var()),
                mean_boarded_actual=float('nan'), mean_boarded_factorized=float('nan'),
                correlation_queue_capacity=float('nan'), covariance_residual_next_headway=float('nan'))
            if model == 'physical':
                q = physical.queues[:, n - 1][sample]
                s = seats[sample]
                grid = np.arange(1, case['capacity'] + 1)
                row['mean_boarded_actual'] = float(physical.boarded[:, n - 1][sample].mean())
                row['mean_boarded_factorized'] = float(np.sum((q[:, None] >= grid).mean(axis=0)
                                                            * (s[:, None] >= grid).mean(axis=0)))
                if q.std() > 0 and s.std() > 0:
                    row['correlation_queue_capacity'] = float(np.corrcoef(q, s)[0, 1])
                after = chosen[chosen + 1 < len(service)]
                row['covariance_residual_next_headway'] = float(np.mean(
                    (residual[after] - residual[after].mean())
                    * (np.diff(service)[after] - np.diff(service)[after].mean())))
            rows.append(row)
    stem = f'{case["name"]}-rep{rep:02d}'
    write_rows(output / 'replications' / (stem + '.csv'), rows)
    save_json(output / 'replications' / (stem + '.json'), dict(case=case,
        replicate=rep, seed=seed, followup=history, all_cohorts_completed=True,
        chronology_replay_equal=True, final_simulation_seconds=history[-1]['simulation_seconds']))
    if rep == 0:
        np.savez_compressed(output / 'traces' / (stem + '.npz'),
            starts=physical.starts, capacities=physical.capacities,
            residuals=physical.residuals, queues=physical.queues)
    return stem


def execute_pool(function, tasks, workers, output, stage):
    completed = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(function, task) for task in tasks]):
            name = future.result()
            completed += 1
            save_json(output / 'status.json', dict(stage=stage, status='running',
                completed=completed, total=len(tasks), last_completed=name))
            print(stage, completed, '/', len(tasks), name, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=24)
    parser.add_argument('--pilot', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    for name in ['source', 'calibration', 'replications', 'traces']:
        (output / name).mkdir()
    names = ['run_revision_scan.py', 'revision_inputs.py', 'route_validation.py',
             'validate_route_phase.py', 'delay_stationary.py', 'sparse_phase_queue.py',
             'evaluate_frozen_route_phase.py', 'validate_service_revision.py',
             'service_dynamics.py', 'run_route_suite.py']
    for name in names:
        shutil.copy2(name, output / 'source' / name)
    cases = all_scenarios()
    if args.pilot:
        wanted = ['uniform-e097-g20', 'heterogeneous-e097', 'original-reference']
        cases = [c for c in cases if c['name'] in wanted]
    count = 12000 if args.pilot else 400000
    evaluation_runs = 6000 if args.pilot else 80000
    protocol = dict(status='frozen before calibration and evaluation', pilot=args.pilot,
        cases=cases, calibration_vehicles=count, calibration_burn=count // 4,
        stations=[3, 9] if args.pilot else list(range(1, 10)),
        replications=2 if args.pilot else 20, evaluation_seed_base=2026111000 if args.pilot else 2026111100,
        calibration_seed_base=2026110200, duration_seed_offset=100000000,
        passenger_seed_offset=300000, workers=args.workers,
        observation_window=[.5 * evaluation_runs * 4.8, evaluation_runs * 4.8],
        cohort_window=[.5 * evaluation_runs * 4.8, .75 * evaluation_runs * 4.8],
        fitting='only independent calibration displacement/capacity pairs; grid=h/4,upperquantile0.9995',
        numerical_refinement='successivequeuecutoffs512,1024,2048,4096,8192;max4momentchange<0.5percent,boundarymass<1e-10; all failures retained',
        primary_accuracy='report all4relativeerrors against physical reference; all4<=10percent is descriptive gate, not universalguarantee',
        permutations='first10000servicesretained (pilot:oneeighth), remaining alignedpairs/capacities/headways/separatesequencespermuted; not a physicalwhole-routepolicy',
        followup='fixedcalendarcohort; vehiclenumberdoublesifcensored, atmost4attempts; counts/durationsseparatestreams preserveprefix',
        trace_scope='full physicaltraceforrep0percase; allreplicationmetricsandinputseeds retained; allchronologyreplaysexactlychecked',
        python=platform.python_version(), numpy=np.__version__, platform=platform.platform(),
        source_sha256={name: hashlib.sha256((output / 'source' / name).read_bytes()).hexdigest() for name in names})
    save_json(output / 'protocol.json', protocol)
    start = time.perf_counter()
    profiles = [(i, c) for i, c in enumerate(cases) if c['phase_validation']]
    execute_pool(calibrate, [(c, i, str(output / 'calibration' / c['name']), count, count // 4)
                            for i, c in profiles], args.workers, output, 'calibration')
    execute_pool(fit_profile, [(c, str(output / 'calibration' / c['name']), protocol['stations'], args.pilot)
                             for _, c in profiles], args.workers, output, 'fit')
    freeze = {str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted((output / 'calibration').rglob('*')) if p.is_file()}
    save_json(output / 'prefinal-frozen-inputs.json', freeze)
    print('ALL phase inputs frozen; starting independent evaluation', flush=True)
    execute_pool(evaluate, [(c, i, rep, str(output), protocol) for i, c in enumerate(cases)
                           for rep in range(protocol['replications'])], args.workers, output, 'evaluation')
    for name, digest in freeze.items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest
    save_json(output / 'status.json', dict(status='completed', elapsed_seconds=time.perf_counter() - start,
        physical_cases=len(cases), physical_replications=len(cases) * protocol['replications'],
        phase_validation_profiles=len(profiles), phase_validation_cells=len(profiles) * len(protocol['stations'])))


if __name__ == '__main__':
    main()
