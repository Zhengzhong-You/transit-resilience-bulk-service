"""Final fresh-seed route validation with identical calendar arrival cohorts."""

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

from route_validation import incident_delays, simulate_route
from run_route_suite import cases, demand, one_task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    for name in ['source', 'traces', 'replications']:
        (output / name).mkdir()
    for name in ['run_revision_final_suite.py', 'run_route_suite.py', 'route_validation.py',
                 'service_dynamics.py', 'validate_service_revision.py']:
        shutil.copy2(name, output / 'source' / name)
    settings = cases()
    settings += [{'name': f'slow-{dwell}-load{int(load * 100):03d}', 'group': 'dwell_sensitivity',
                  'load': load, 'dwell': dwell, 'dwell_scale': 2., 'donor_suffix': '-slow'}
                 for load in [.8, .97] for dwell in ['constant', 'exogenous', 'endogenous']]
    protocol = {'status': 'frozen before fresh-seed evaluation', 'cases': settings,
        'vehicles': 80000, 'replications': 20, 'workers': 24,
        'evaluation_incident_seeds': list(range(2026104200, 2026104220)),
        'passenger_seed_offset': 300000, 'scheduled_headway': 4.8, 'capacity': 50,
        'stations': 10, 'normal_segment_minutes': 5.,
        'queue_measurement': 'last half of each stated vehicle horizon',
        'waiting_cohort': 'the SAME local calendar arrivals in (.5*L*4.8,.75*L*4.8] under every variant and initial state, followed until final service, explicit censoring',
        'horizon_scope': 'L counts retained scheduled runs; all missed departures remain in the timetable and may depart late',
        'headway_definition': 'service-gate interval; departure-headway variance also recorded',
        'dwell_laws_minutes': {'fast':[.1,.02,.01], 'slow':[.1,.04,.02]},
        'donor_calibration': {'vehicles':40000,'measurement':'last20000',
            'incident_seeds':[2026103500,2026103501,2026103900,2026103901],
            'passenger_seed_offset':300000},
        'fleet': 'cyclic vehicle assignment, explicit ten-segment FIFO empty return, no terminal turnaround; exogenous incident clock during normal travel only',
        'incident_laws': 'deterministic/exponential mean1 minute; lognormal mean1 variance1, sigma=sqrt(log2)',
        'destinations': 'origin-dependent 2x2 exchanges preserve origin/destination totals and all nominal cut loads',
        'passing': 'free zero-dwell trajectories, vehicle identity retained, station service processed chronologically',
        'phase_validation': 'joint displacement/capacity model and settings frozen using independent seed3800 calibration and development seeds3600..3619, before accessing these final seeds',
        'development_exclusion': '3450 pilots,3600..3619 former route/phase studies are retained but never pooled into this final cohort',
        'python':platform.python_version(),'numpy':np.__version__,'host':platform.node(),'pid':os.getpid(),
        'source_sha256': {p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (output / 'source').iterdir()}}
    frozen_phase_files = [Path('runs/rr-route-phase-refinement-20261101-v1/central_joint.npz'),
                          Path('runs/rr-route-phase-refinement-20261101-v1/phase_transition.npz'),
                          Path('runs/rr-route-phase-20261101-v1/central_kernel.npz')]
    assert all(p.is_file() for p in frozen_phase_files)
    protocol['frozen_phase_kernel_sha256'] = {str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                                             for p in frozen_phase_files}
    (output / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    start=time.perf_counter()
    for load,seed,scale,suffix in [(.8,2026103500,1.,''),(.97,2026103501,1.,''),
                                   (.8,2026103900,2.,'-slow'),(.97,2026103901,2.,'-slow')]:
        rates,p=demand(load)
        calibration=simulate_route(rates,p,incident_delays(seed,40000,10),seed+300000,
                                    dwell_coefficients=(.1,.02*scale,.01*scale))
        np.savez_compressed(output/f'dwell_calibration_load{int(load*100):03d}{suffix}.npz',
                            dwell=calibration.dwell[20000:])
    tasks=[{'case':case,'replicate':rep,'vehicles':80000,'output':str(output),
            'seed_base':2026104200} for case in settings for rep in range(20)]
    print('frozen final tasks',len(tasks),flush=True)
    with ProcessPoolExecutor(max_workers=24) as executor:
        futures=[executor.submit(one_task,task) for task in tasks]
        for completed,future in enumerate(as_completed(futures),1):
            name,elapsed=future.result()
            state={'status':'running','completed':completed,'total':len(tasks),
                   'last_completed':name,'elapsed_seconds':time.perf_counter()-start}
            (output/'status.json').write_text(json.dumps(state,indent=2)+'\n')
            print(completed,'/',len(tasks),name,round(elapsed,2),flush=True)
    state['status']='completed'
    (output/'status.json').write_text(json.dumps(state,indent=2)+'\n')


if __name__=='__main__':
    main()
