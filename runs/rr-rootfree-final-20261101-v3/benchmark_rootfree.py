"""Matched mean/variance and timing comparison of roots and contour integration."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

from root_solver import certified_interpolation_roots
from rootfree_bulk import contour_moments, effective_arrival_pgf, moments_from_roots
from validate_service_revision import write_rows


def root_worker(case):
    c, rho = case['capacity'], case['load']
    _, derivatives, characteristic = effective_arrival_pgf(c, rho, case['availability'], sigma=case['sigma'])
    start = time.perf_counter()
    row = {'roots_status': 'completed', 'roots_mean': float('nan'),
           'roots_variance': float('nan'), 'roots_residual': float('nan')}
    try:
        roots, diagnostics = certified_interpolation_roots(characteristic, c, rho)
        mean, variance = moments_from_roots(roots, c, derivatives)
        if max(abs(mean.imag), abs(variance.imag)) > 1e-5 or min(mean.real, variance.real) < 0:
            raise ArithmeticError('Nonphysical or complex moments from root extraction.')
        row.update(roots_mean=float(mean.real), roots_variance=float(variance.real),
                   roots_residual=diagnostics['largest_residual'])
    except (RuntimeError, ArithmeticError) as error:
        row['roots_status'] = str(error)
    row['roots_seconds'] = time.perf_counter() - start
    return row


def compare_case(case, limit):
    values, _, _ = effective_arrival_pgf(case['capacity'], case['load'], case['availability'], sigma=case['sigma'])
    rows = []
    for repetition in range(3):
        contour = contour_moments(case['capacity'], values)
        row = dict(case, repetition=repetition, contour_mean=contour['mean_residual'],
                   contour_variance=contour['variance_residual'], contour_seconds=contour['elapsed_seconds'],
                   contour_points=contour['points'], contour_refinement=contour['relative_refinement_change'],
                   mean_relative_error=float('nan'), variance_relative_error=float('nan'))
        try:
            completed = subprocess.run([sys.executable, __file__, '--worker-case', json.dumps(case)],
                                       capture_output=True, text=True, timeout=limit, check=True)
            root = json.loads(completed.stdout)
        except subprocess.TimeoutExpired:
            root = {'roots_status': f'time limit {limit:g} seconds including child startup',
                    'roots_seconds': limit, 'roots_mean': float('nan'),
                    'roots_variance': float('nan'), 'roots_residual': float('nan')}
        row.update(root)
        if root['roots_status'] == 'completed':
            row['mean_relative_error'] = abs(root['roots_mean'] - contour['mean_residual']) / max(1e-12, abs(contour['mean_residual']))
            row['variance_relative_error'] = abs(root['roots_variance'] - contour['variance_residual']) / max(1e-12, abs(contour['variance_residual']))
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--root-limit', type=float, default=30.)
    parser.add_argument('--worker-case')
    args = parser.parse_args()
    if args.worker_case:
        print(json.dumps(root_worker(json.loads(args.worker_case))))
        return
    if args.output is None:
        parser.error('--output is required for the benchmark')
    args.output.mkdir(parents=True, exist_ok=False)
    for name in ['root_solver.py', 'rootfree_bulk.py', 'benchmark_rootfree.py',
                 'test_root_verification.py', 'test_rootfree_bulk.py']:
        shutil.copy2(name, args.output / name)
    cases = [{'capacity': c, 'load': rho, 'availability': q, 'sigma': float(sigma)}
             for c in [1, 8, 34, 50] for rho in [.5, .9, .97, .99]
             for q, sigma in [(1., 0.), (.7, 2 * np.sqrt(8.))]]
    protocol = {'cases': cases, 'timing_repetitions': 3, 'numerical_tolerance': 1e-8,
        'parallel_cases': args.workers, 'time_limit_per_root_repetition': args.root_limit,
        'timeout_boundary': 'includes child interpreter startup; successful numerical timings exclude startup',
        'root_method': 'checked-in Powell/interpolation least-squares solver; close multi-zero clusters now rejected',
        'root_free_method': 'Oblakova et al.2019 Eq20 mean and differentiated logarithmic-factorization variance',
        'same_model': 'iid positive-normal (or deterministic) headways; independent binomial available capacity; A(z)=E[z^(Y+C-S)]',
        'comparison_boundary': 'characteristic and its derivatives already constructed; root search or contour quadrature plus moment extraction timed',
        'interpretation': 'numerical refinement is not an exact certificate; iid model only',
        'source': 'https://link.springer.com/article/10.1007/s11134-019-09614-1'}
    (args.output / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(compare_case, case, args.root_limit) for case in cases]
        for future in as_completed(futures):
            records = future.result()
            rows.extend(records)
            write_rows(args.output / 'replications.csv', rows)
            print({key: records[0][key] for key in ['capacity', 'load', 'availability', 'sigma']},
                  records[-1]['roots_status'], flush=True)
    (args.output / 'status.json').write_text(json.dumps({'status': 'completed', 'cases': len(cases),
                                                       'repetitions': len(rows)}, indent=2) + '\n')


if __name__ == '__main__':
    main()
