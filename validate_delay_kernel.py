"""Independent trace validation and numerical refinement of the displacement closure."""

import json
import shutil
import time
from pathlib import Path

import numpy as np

from delay_stationary import stationary_queue
from explore_delay_state import displacement_trace, fit_kernel, sample_phases
from validate_service_revision import residual_queues, write_rows


def main():
    output = Path('runs/rr-delay-validation-20261031')
    output.mkdir(parents=True, exist_ok=False)
    for name in ['service_dynamics.py', 'delay_stationary.py', 'explore_delay_state.py',
                 'validate_delay_kernel.py', 'validate_service_revision.py']:
        shutil.copy2(name, output / name)
    start = time.perf_counter()
    h, capacity, rate, vehicles, burn = 4.8, 34, 0.97 * 34 / 4.8, 80000, 10000
    calibration = displacement_trace(2026103200, 400000, burn, h)
    refinements = []
    cases = [('queue_cutoff', 0.4, 0.0005, k) for k in [128, 256, 512, 1024]]
    cases += [('phase_grid', step, 0.0005, 512) for step in [0.8, 0.2]]
    cases += [('phase_tail', 0.4, tail, 512) for tail in [0.005, 0.00005]]
    for kind, step, tail, cutoff in cases:
        states, transition, frequencies, upper, counts = fit_kernel(calibration, step, tail)
        result = stationary_queue(states, transition, h, rate, capacity, cutoff)
        np.savez_compressed(output / f'{kind}-{step}-{tail}-{cutoff}.npz',
                            joint=result.pop('joint'), phase=result.pop('phase'))
        refinements.append({'check': kind, 'step': step, 'tail_probability': tail,
                            'cutoff': cutoff, 'states': len(states), 'upper': upper, **result})
        print(kind, step, tail, cutoff, result['mean_residual'], flush=True)
    independent_fits = []
    for seed in range(2026103201, 2026103206):
        trace = displacement_trace(seed, 200000, burn, h)
        states, transition, frequencies, upper, counts = fit_kernel(trace, 0.4, 0.0005)
        result = stationary_queue(states, transition, h, rate, capacity, 512)
        result.pop('joint')
        result.pop('phase')
        independent_fits.append({'calibration_seed': seed, 'vehicles': 200000,
                                 'states': len(states), 'upper': upper, **result})
        print('independent calibration', seed, result['mean_residual'], flush=True)
    states, transition, frequencies, upper, counts = fit_kernel(calibration, 0.4, 0.0005)
    np.savez_compressed(output / 'central_kernel.npz', states=states, transition=transition,
                        frequencies=frequencies, counts=counts)
    evaluations = []
    for replicate in range(20):
        seed = 2026103100 + replicate
        rng = np.random.default_rng(seed + 100000)
        displacement = displacement_trace(seed, vehicles, burn, h)
        quantized = np.minimum(np.floor(displacement / 0.4 + 1e-8) * 0.4, upper)
        generated = sample_phases(states, transition, frequencies, vehicles + burn + 1, rng)[burn:]
        models = {'propagated_fifo': displacement, 'quantized_fifo': quantized,
                  'delay_state_kernel': generated}
        clocks = {name: np.cumsum(np.maximum(0, h + np.diff(phases)))
                  for name, phases in models.items()}
        end = max(clock[-1] for clock in clocks.values())
        arrivals = np.sort(rng.uniform(0, end, rng.poisson(rate * end)))
        for name, clock in clocks.items():
            cumulative = np.searchsorted(arrivals, clock, side='right')
            per_service = np.diff(np.r_[0, cumulative])
            residual = residual_queues(per_service, capacity)
            boarded = cumulative - residual
            assert np.all(np.diff(np.r_[0, boarded]) >= 0)
            departure_indices = np.searchsorted(boarded, np.arange(1, boarded[-1] + 1))
            waits = clock[departure_indices] - arrivals[:boarded[-1]]
            assert waits.min() >= -1e-8
            measured_waits = waits[departure_indices >= vehicles // 2]
            headways = np.diff(np.r_[0, clock])
            previous_residual = np.r_[0, residual[:-1]]
            sample = residual[vehicles // 2:]
            evaluations.append({'replicate': replicate, 'seed': seed, 'model': name,
                                'mean_residual': float(sample.mean()),
                                'variance_residual': float(sample.var(ddof=1)),
                                'mean_wait': float(measured_waits.mean()),
                                'variance_wait': float(measured_waits.var(ddof=1)),
                                'pr_residual_over_100': float(np.mean(sample > 100)),
                                'pr_residual_over_200': float(np.mean(sample > 200)),
                                'residual_times_next_headway': float(np.mean(
                                    previous_residual[vehicles // 2:] * headways[vehicles // 2:]))})
        print('independent evaluation', replicate + 1, flush=True)
    write_rows(output / 'numerical_refinements.csv', refinements)
    write_rows(output / 'independent_calibrations.csv', independent_fits)
    write_rows(output / 'evaluation_replications.csv', evaluations)
    protocol = {'h': h, 'available_capacity': capacity, 'arrival_rate': rate, 'load_ratio': 0.97,
                'vehicles': vehicles, 'burn_in_service_vehicles': burn,
                'queue_measurement_window': 'last half of evaluation vehicle intervals',
                'central_calibration_seed': 2026103200, 'central_calibration_vehicles': 400000,
                'other_calibration_seeds': list(range(2026103201, 2026103206)),
                'evaluation_seeds': list(range(2026103100, 2026103120)),
                'calibration_uses': 'service displacements only; no passenger queue fitting',
                'predeclared_gate': '10 percent error for held-out residual mean and waiting mean; report variance/tail errors and phase/grid/cutoff sensitivity separately',
                'scope': 'One station with fixed available capacity, zero dwell, exogenous dispatch. Empirical first-order phase closure, not an exact coupled-route stationary distribution.',
                'numerical_status': 'Finite capped-chain consistency and refinement checks, not a rigorous truncation-error certificate.',
                'elapsed_seconds': time.perf_counter() - start}
    (output / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    print(json.dumps(protocol), flush=True)


if __name__ == '__main__':
    main()
