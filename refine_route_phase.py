"""Retain observed available capacity in the calibrated departure phase."""

import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np
from scipy.sparse import save_npz

from delay_stationary import stationary_queue
from run_route_suite import demand
from sparse_phase_queue import fit_joint_phase, sparse_phase_transition_operator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    for name in ['refine_route_phase.py', 'sparse_phase_queue.py', 'delay_stationary.py',
                 'test_sparse_phase_queue.py']:
        shutil.copy2(name, args.output / name)
    protocol = {'motivation': 'displacement-only phase with edge capacity emissions missed the 10 percent variance gate',
        'state': '(quantized departure displacement, observed available capacity)',
        'calibration': str(args.calibration), 'queue_fitting': False,
        'settings': [['central', 1.2, .0005, 1024], ['queue_refinement', 1.2, .0005, 2048],
                     ['grid_refinement', .8, .0005, 1024], ['range_refinement', 1.2, .00005, 1024]],
        'evaluation_status': 'development refinement; final independent evaluation required before an accuracy claim',
        'existing_evaluations': '2026103600..3619 are development after inspecting first closure; final seeds must be new'}
    (args.output / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    calibration = np.load(args.calibration)
    displacement, available = calibration['displacement'], calibration['available_capacity']
    rate = demand(.97)[0][8]
    start = time.perf_counter()
    rows = []
    for name, step, tail, cutoff in protocol['settings']:
        states, capacity, sparse, upper = fit_joint_phase(displacement, available, step, tail)
        print('solving', name, 'phases', len(states), 'edges', sparse.nnz, flush=True)
        transition = sparse.toarray()
        result = stationary_queue(states, transition, 4.8, rate, capacity, cutoff,
            tolerance=1e-10, operator_factory=sparse_phase_transition_operator)
        joint, phase = result.pop('joint'), result.pop('phase')
        result.update(model=name, grid=step, tail=tail, cutoff=cutoff, phases=len(states),
                      edges=sparse.nnz, upper_displacement=upper)
        rows.append(result)
        if name == 'central':
            save_npz(args.output / 'phase_transition.npz', sparse)
            np.savez_compressed(args.output / 'central_joint.npz', displacement=states,
                                capacity=capacity, phase=phase, joint=joint)
        (args.output / 'predictions.json').write_text(json.dumps(rows, indent=2) + '\n')
        print(name, result['mean_residual'], result['variance_wait'], flush=True)
    (args.output / 'status.json').write_text(json.dumps({'status': 'completed',
        'elapsed_seconds': time.perf_counter() - start}, indent=2) + '\n')


if __name__ == '__main__':
    main()
