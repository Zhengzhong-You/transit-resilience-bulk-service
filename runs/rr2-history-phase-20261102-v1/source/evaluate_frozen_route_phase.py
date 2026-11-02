"""Evaluate frozen departure/capacity kernels on the final untouched route seeds."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
from scipy.sparse import load_npz

from route_validation import CalendarPassengers
from run_route_suite import demand
from validate_route_phase import queue_metrics
from validate_service_revision import write_rows


def normalized_hash_paths(hashes):
    return {str(path).replace('\\', '/'): value for path,value in hashes.items()}


def sample_joint(displacement, capacity, transition, stationary, number, rng):
    cumulative = transition.data.copy()
    for i in range(transition.shape[0]):
        a, b = transition.indptr[i:i+2]
        cumulative[a:b] = np.cumsum(cumulative[a:b])
        cumulative[b-1] = 1.
    indices = np.empty(number, dtype=int)
    indices[0] = rng.choice(len(stationary), p=stationary)
    uniforms = rng.random(number-1)
    for k, uniform in enumerate(uniforms, 1):
        a, b = transition.indptr[indices[k-1]:indices[k-1]+2]
        offset = np.searchsorted(cumulative[a:b], uniform, side='right')
        indices[k] = transition.indices[a+offset]
    return displacement[indices], capacity[indices]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--route-run', type=Path, required=True)
    parser.add_argument('--phase-run', type=Path, required=True)
    parser.add_argument('--edge-phase-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    for name in ['evaluate_frozen_route_phase.py','validate_route_phase.py','route_validation.py',
                 'validate_service_revision.py','service_dynamics.py','run_route_suite.py']:
        shutil.copy2(name,args.output/name)
    route_protocol=json.loads((args.route_run/'protocol.json').read_text())
    kernels=[args.phase_run/'central_joint.npz',args.phase_run/'phase_transition.npz',
             args.edge_phase_run/'central_kernel.npz']
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in kernels}
    assert normalized_hash_paths(hashes)==normalized_hash_paths(route_protocol['frozen_phase_kernel_sha256'])
    assert route_protocol['evaluation_incident_seeds']==list(range(2026104200,2026104220))
    assert json.loads((args.route_run/'status.json').read_text())['status']=='completed'
    protocol={'kernel_sha256_verified_against_prefinal_freeze':hashes,
        'evaluation_seeds':list(range(2026104200,2026104220)),
        'phase_simulation_seeds':list(range(2026704200,2026704220)),
        'permutation_seeds':list(range(2026804200,2026804220)),
        'arrival_cohort':'same local calendar arrivals in (.5*L*4.8,.75*L*4.8] for all models',
        'queue_measurement':'last half of80000 vehicle service opportunities',
        'controls':'suffix after first10000 opportunities permuted; full chronological replay must reproduce physical residuals',
        'calibration':'independent seed3800 upstream displacement/available-capacity trace; no local queue fitting',
        'predeclared_gate':'10 percent relative error in each of mean/variance of queue and waiting time; report all metrics',
        'scope':'conditional first-order joint phase closure, one binding-capacity route profile; not field calibration or full-route exact moments'}
    (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    kernel=np.load(kernels[0]);delta,c=kernel['displacement'],kernel['capacity']
    phase=kernel['phase'];transition=load_npz(kernels[1]).tocsr()
    rates,p=demand(.97)
    start=time.perf_counter();rows=[]
    for rep in range(20):
        seed=2026104200+rep
        trace=np.load(args.route_run/'traces'/f'zero-load097-rep{rep:02d}-service.npz')
        clock=trace['starts'][:,8];capacity=trace['capacities'][:,8]
        length,burn=len(clock),10000
        intervals=np.diff(np.r_[0.,clock]);suffix=intervals[burn:]
        rng=np.random.default_rng(2026804200+rep)
        together=rng.permutation(len(suffix));separate=rng.permutation(len(suffix))
        permuted_clock=np.r_[clock[:burn],clock[burn-1]+suffix[together].cumsum()]
        sampled,sampled_capacity=sample_joint(delta,c,transition,phase,length,
                                              np.random.default_rng(2026704200+rep))
        physical_delta=clock-np.arange(length)*4.8-45.
        upper=float(delta.max())
        models={'chronological':(clock,capacity),
            'jointly_permuted':(permuted_clock,np.r_[capacity[:burn],capacity[burn:][together]]),
            'capacity_permuted':(clock,np.r_[capacity[:burn],capacity[burn:][together]]),
            'headway_permuted':(permuted_clock,capacity),
            'independently_permuted':(permuted_clock,np.r_[capacity[:burn],capacity[burn:][separate]]),
            'quantized_chronological':(np.arange(length)*4.8+45.
                +np.minimum(np.floor(physical_delta/1.2+1e-8)*1.2,upper),capacity),
            'joint_phase_simulation':(np.arange(length)*4.8+45.+sampled,sampled_capacity)}
        stream=CalendarPassengers(rates[8],p[8],seed+300000+8000)
        stream.through(max(t[-1] for t,_ in models.values()))
        cohort_members=None
        for name,(service,batches) in models.items():
            assert np.all(np.diff(service)>=-1e-8)
            metrics,residual=queue_metrics(service,batches,stream)
            if name=='chronological':
                np.testing.assert_array_equal(residual,trace['residuals'][:,8])
                cohort_members=metrics['cohort_passengers']
            assert metrics['cohort_passengers']==cohort_members
            metrics.update(model=name,replicate=rep,seed=seed)
            rows.append(metrics)
        write_rows(args.output/'replications.csv',rows)
        print('fresh paired evaluation',rep+1,flush=True)
    reference={k:float(np.mean([row[k] for row in rows if row['model']=='chronological']))
               for k in ['mean_queue','variance_queue','mean_wait','variance_wait','mean_residual']}
    predictions=json.loads((args.phase_run/'predictions.json').read_text())
    central=next(row for row in predictions if row['model']=='central')
    edge=json.loads((args.edge_phase_run/'predictions.json').read_text())[0]
    errors={name:{metric:100*abs(pred[metric]-reference[metric])/abs(reference[metric])
                  for metric in ['mean_queue','variance_queue','mean_wait','variance_wait']}
            for name,pred in [('joint_phase',central),('displacement_only',edge)]}
    (args.output/'relative_errors.json').write_text(json.dumps({'reference':reference,
        'relative_errors_percent':errors,'joint_phase_gate_passed':all(e<10 for e in errors['joint_phase'].values())},indent=2)+'\n')
    for name in ['predictions.json','protocol.json']:
        shutil.copy2(args.phase_run/name,args.output/('frozen_phase_'+name))
    (args.output/'status.json').write_text(json.dumps({'status':'completed',
        'replications':20,'elapsed_seconds':time.perf_counter()-start},indent=2)+'\n')


if __name__=='__main__':
    main()
