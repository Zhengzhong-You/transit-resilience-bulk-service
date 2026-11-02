"""Evaluate joint dependence from the final complete physical route trajectories."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from scipy.stats import t

from route_validation import CalendarPassengers
from run_route_suite import demand


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--route-run',type=Path,required=True)
    parser.add_argument('--phase-run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    rates,p=demand(.97);records=[];correlations=[]
    for rep in range(20):
        seed=2026104200+rep
        trace=np.load(args.route_run/'traces'/f'zero-load097-rep{rep:02d}-service.npz')
        clocks,available,residual=trace['starts'],trace['capacities'],trace['residuals']
        length=len(clocks);begin=length//2
        rng=np.random.default_rng(2026904200+rep)
        inspection=np.sort(rng.uniform(.5*length*4.8,.75*length*4.8,20000))
        calendar=[]
        for n in range(9):
            stream=CalendarPassengers(rates[n],p[n],seed+300000+n*1000)
            stream.through(clocks[-1,n])
            cumulative=np.searchsorted(stream.times,clocks[:,n],side='right')
            arrivals=np.diff(np.r_[0,cumulative])
            queue=np.r_[0,residual[:-1,n]]+arrivals
            boarded=cumulative-residual[:,n]
            batch=np.diff(np.r_[0,boarded])
            assert np.all(batch>=0) and np.all(batch<=available[:,n])
            last=np.searchsorted(clocks[:,n],inspection,side='right')-1
            assert np.all(last>=0)
            at_time=np.searchsorted(stream.times,inspection,side='right')-boarded[last]
            assert np.all(at_time>=0)
            calendar.append(at_time)
            if n==8:
                reference=pd.read_csv(args.route_run/'replications'/f'zero-load097-rep{rep:02d}.csv')
                row=reference[reference.station.eq(9)].iloc[0]
                np.testing.assert_allclose(queue[begin:].mean(),row.mean_queue,rtol=1e-12)
                g=50-available[begin:,n]
                rr=residual[begin:,n]
                h=np.diff(clocks[:,n])[begin:]
                records.append({'replicate':rep,'seed':seed,
                    'corr_prequeue_remaining_load':float(np.corrcoef(queue[begin:],g)[0,1]),
                    'corr_residual_available_capacity':float(np.corrcoef(rr,available[begin:,n])[0,1]),
                    'cov_residual_next_headway':float(np.cov(residual[begin:-1,n],h,ddof=0)[0,1]),
                    'mean_calendar_queue':float(at_time.mean()),
                    'variance_calendar_queue':float(at_time.var())})
        correlations.append(np.corrcoef(calendar))
    frame=pd.DataFrame(records);frame.to_csv(args.output/'replications.csv',index=False)
    summaries={}
    for key in frame.columns.difference(['replicate','seed']):
        values=frame[key];half=t.ppf(.975,19)*values.std(ddof=1)/np.sqrt(20)
        summaries[key]={'mean':float(values.mean()),'lower95':float(values.mean()-half),
                        'upper95':float(values.mean()+half)}
    kernel=np.load(args.phase_run/'central_joint.npz');joint=kernel['joint']
    phase,c,delta=kernel['phase'],kernel['capacity'],kernel['displacement']
    transition=load_npz(args.phase_run/'phase_transition.npz').tocsr()
    prediction=json.loads((args.phase_run/'predictions.json').read_text())[0]
    first=np.arange(len(joint))@joint;g=50-c;pg=transition@g
    mean_g=phase@g;variance_g=phase@(g*g)-mean_g**2
    e_hg=4.8*mean_g+phase@(delta*g)-(phase*delta)@pg
    e_qg=first@pg+rates[8]*e_hg
    fitted={'corr_prequeue_remaining_load':float((e_qg-prediction['mean_queue']*mean_g)
                /np.sqrt(prediction['variance_queue']*variance_g)),
            'corr_residual_available_capacity':float((first@c-prediction['mean_residual']*(phase@c))
                /np.sqrt(prediction['variance_residual']*variance_g)),
            'cov_residual_next_headway':float(prediction['residual_times_next_headway']
                -4.8*prediction['mean_residual'])}
    average=np.mean(correlations,axis=0)
    pd.DataFrame(average,index=np.arange(1,10),columns=np.arange(1,10)).to_csv(args.output/'calendar_queue_correlations.csv')
    result={'physical':summaries,'fitted_joint_phase':fitted,
            'protocol':{'scope':'final zero-dwell route at maximum cut.97,station9;20independent seeds4200..4219',
                'event_window':'last40000services','calendar_times':'20000independentuniformpoints in(.5Lh,.75Lh] perrep,shared acrossstations',
                'inspection_seeds':list(range(2026904200,2026904220)),
                'interpretation':'correlation diagnostics, not causal attribution or a universal accuracy guarantee; covarianceR,H uses NEXT headway',
                'reconstruction_check':'pre-service mean matches archived full-route output; all reconstructed boarding batches valid'}}
    (args.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    plt.rcParams.update({'font.size':9,'pdf.fonttype':42})
    fig,ax=plt.subplots(figsize=(4.2,3.5));im=ax.imshow(average,vmin=-1,vmax=1,cmap='RdBu_r')
    ax.set(xticks=np.arange(9),xticklabels=np.arange(1,10),yticks=np.arange(9),yticklabels=np.arange(1,10),
           xlabel='Station',ylabel='Station')
    fig.colorbar(im,ax=ax,label='Queue correlation at common calendar times')
    fig.tight_layout();fig.savefig(args.output/'joint_calendar_queues.pdf');plt.close(fig)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
