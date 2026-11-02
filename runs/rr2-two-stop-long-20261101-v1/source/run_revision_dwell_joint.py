"""Reconstruct previous dwell runs and independently validate a direct joint kernel."""
import os
for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[name]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib,json,shutil,time
from pathlib import Path
import numpy as np
import pandas as pd
from route_validation import incident_delays,simulate_route,CalendarPassengers
from run_route_suite import demand
from validate_route_phase import queue_metrics
from validate_service_revision import write_rows


def dwell_task(task):
    case,rep,old,out=task
    old,out=Path(old),Path(out)
    seed=2026104200+rep; L=80000
    rates,p=demand(case['load'])
    incidents=incident_delays(seed,L,10)
    kwargs={}
    if case['dwell']=='endogenous':
        scale=case.get('dwell_scale',1.)
        kwargs['dwell_coefficients']=(.1,.02*scale,.01*scale)
    elif case['dwell'] in ['constant','exogenous']:
        donor=np.load(old/f"dwell_calibration_load{int(case['load']*100):03d}{case.get('donor_suffix','')}.npz")['dwell']
        if case['dwell']=='constant':
            prescribed=np.broadcast_to(donor.mean(axis=0),(L,10))
        else:
            rng=np.random.default_rng(seed+200000)
            prescribed=np.column_stack([rng.choice(donor[:,n],L) for n in range(10)])
        kwargs['prescribed_dwell']=prescribed
    result=simulate_route(rates,p,incidents,seed+300000,**kwargs)
    stem=f"{case['name']}-rep{rep:02d}"
    previous=pd.read_csv(old/'replications'/(stem+'.csv'))
    errors=[];wait_variance_errors=[]
    for row in result.summaries(4.8):
        ref=previous[previous.station.eq(row['station'])].iloc[0]
        for key,value in row.items():
            if np.isfinite(value) and np.isfinite(ref[key]):
                (wait_variance_errors if key=='variance_wait_boarded' else errors).append(abs(value-float(ref[key])))
    trace=np.load(old/'traces'/(stem+'.npz')); indices=trace['vehicle']
    trace_equal=all(np.array_equal(trace[k],getattr(result,attr)[indices]) for k,attr in
        [('starts','starts'),('residual','residuals'),('origin','origin'),('returned','returned')])
    if not trace_equal or max(errors)>1e-7 or max(wait_variance_errors)>1e-4:
        raise ArithmeticError(f'Previous dwell evidence not reproduced: {stem}, {max(errors)}, {trace_equal}')
    rows=[]
    for n in range(10):
        hu=np.diff(result.starts[:,n])[39999:]
        hd=np.diff(result.departures[:,n])[39999:]
        current=result.dwell[40000:,n]; prior=result.dwell[39999:-1,n]; dd=current-prior
        cov=lambda a,b:float(np.mean((a-a.mean())*(b-b.mean())))
        upstream=result.origin if n==0 else result.departures[:,n-1]
        ready=upstream+5+incidents[:,n]
        block=(result.starts[:,n]-ready)[40000:]
        identity=float(np.max(np.abs(hd-hu-dd)))
        variance_identity=abs(float(hd.var()-hu.var()-dd.var()-2*cov(hu,dd)))
        if identity>1e-7 or variance_identity>1e-7 or np.min(hu-prior)<-1e-7 or np.min(hd-current)<-1e-7:
            raise ArithmeticError('Gate/departure/variance identity failed.')
        q,s,beta=result.queues[40000:,n],result.capacities[40000:,n],result.boarded[40000:,n]
        row=dict(case=case['name'],load=case['load'],station=n+1,replicate=rep,
            mean_gate_headway=float(hu.mean()),variance_gate_headway=float(hu.var()),
            mean_departure_headway=float(hd.mean()),variance_departure_headway=float(hd.var()),
            mean_dwell=float(current.mean()),variance_dwell=float(current.var()),
            variance_dwell_difference=float(dd.var()),covariance_gate_dwell_difference=cov(hu,dd),
            covariance_gate_current_dwell=cov(hu,current),covariance_gate_previous_dwell=cov(hu,prior),
            departure_short_fraction=float(np.mean(hd<.5)),gate_short_fraction=float(np.mean(hu<.5)),
            departure_zero_fraction=float(np.mean(hd<1e-8)),berth_blocked_fraction=float(np.mean(block>1e-8)),
            mean_berth_block=float(block.mean()),denied_fraction=float(np.mean(q>s)),
            boards_full_capacity_fraction=float(np.mean(beta==50)),
            mean_available_capacity=float(s.mean()),mean_boarded=float(beta.mean()),
            chronology_identity_max_error=identity,variance_identity_error=variance_identity,
            old_summary_max_error=max(errors),old_wait_variance_max_error=max(wait_variance_errors),old_trace_exact=trace_equal)
        for quantile in [.01,.1,.5,.9,.99]:
            row[f'departure_quantile_{quantile}']=float(np.quantile(hd,quantile))
        rows.append(row)
    write_rows(out/'replications'/(stem+'.csv'),rows)
    return stem


def joint_task(task):
    name,a,b,rep,out,direct,L=task
    out,direct=Path(out),Path(direct);seed=2026112100+rep
    rates=np.array([a,b,0.]);p=np.array([[0.,.5,.5],[0.,0.,1.],[0.,0.,0.]])
    result=simulate_route(rates,p,np.zeros((L,3)),seed,headway=1.,capacity=1,segment_time=.25)
    rows=[]
    truth=np.load(direct/(name+'-joint.npz'))['joint'];cutoff=len(truth)-1
    chosen=result.residuals[L//2:,:2]
    overflow=int(np.sum(np.any(chosen>cutoff,axis=1)))
    if overflow: raise ArithmeticError('Physical joint sample exceeds direct-kernel cap; retain a wider comparison.')
    hist=np.bincount((chosen[:,0]*(cutoff+1)+chosen[:,1]),minlength=(cutoff+1)**2).reshape(truth.shape)/len(chosen)
    joint_tv=float(.5*np.abs(hist-truth).sum())
    corr=float(np.corrcoef(chosen.T)[0,1])
    for n in range(2):
        stream=CalendarPassengers(rates[n],p[n],seed+n*1000)
        metrics,residual=queue_metrics(result.starts[:,n],result.capacities[:,n],stream,
            scheduled_headway=1.,observation_window=(L/2,L),cohort_window=(L/2,3*L/4))
        np.testing.assert_array_equal(residual,result.residuals[:,n])
        if metrics['censored_fraction']!=0:raise ArithmeticError('Incomplete toy validation cohort.')
        rows.append(dict(metrics,case=name,replicate=rep,station=n+1,seed=seed,
            mean_residual=float(chosen[:,n].mean()),variance_residual=float(chosen[:,n].var()),
            joint_empirical_target_tv=joint_tv,residual_correlation=corr))
    write_rows(out/'joint'/(f'{name}-rep{rep:02d}.csv'),rows)
    if rep==0:np.savez_compressed(out/'joint'/(name+'-physical-residual.npz'),residual=chosen)
    return f'joint-{name}-{rep:02d}'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous',type=Path,required=True)
    parser.add_argument('--direct',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=24)
    parser.add_argument('--pilot',action='store_true')
    parser.add_argument('--skip-dwell',action='store_true')
    parser.add_argument('--toy-vehicles',type=int,default=120000)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    for name in ['source','replications','joint']:(args.output/name).mkdir()
    for name in ['run_revision_dwell_joint.py','route_validation.py','run_route_suite.py','validate_route_phase.py','validate_service_revision.py']:
        shutil.copy2(Path(__file__).parent/name,args.output/'source'/name)
    cases=[x for x in json.loads((args.previous/'protocol.json').read_text())['cases'] if x['group'] in ['dwell','dwell_sensitivity']]
    if args.skip_dwell:cases=[]
    reps=1 if args.pilot else 20
    tasks=[(c,r,str(args.previous.resolve()),str(args.output.resolve())) for c in cases for r in range(reps)]
    toy=[(name,a,b,r,str(args.output.resolve()),str(args.direct.resolve()),args.toy_vehicles) for name,a,b in [('light',.35,.25),('binding',.75,.55)] for r in range(reps)]
    protocol=dict(previous_run=str(args.previous),dwell_reconstructions=len(tasks),fresh_joint_runs=len(toy),
        dwell_seeds='original 2026104200..4219; identical 80000-run arrays and donor marginals',
        reconstruction_not_new_independent_evidence=True,toy_vehicles=args.toy_vehicles,toy_seed_base=2026112100,
        toy_observation=[args.toy_vehicles/2,args.toy_vehicles],toy_waiting_cohort=[args.toy_vehicles/2,3*args.toy_vehicles/4],
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.output/'source').iterdir()})
    (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2))
    start=time.perf_counter();done=0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(dwell_task,t) for t in tasks]+[pool.submit(joint_task,t) for t in toy]
        for future in as_completed(futures):
            name=future.result();done+=1
            state=dict(status='running',completed=done,total=len(futures),last=name,elapsed_seconds=time.perf_counter()-start)
            (args.output/'status.json').write_text(json.dumps(state,indent=2));print(done,'/',len(futures),name,flush=True)
    state['status']='completed';(args.output/'status.json').write_text(json.dumps(state,indent=2))

if __name__=='__main__':main()
