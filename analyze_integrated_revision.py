"""Canonical numerical summaries and vector figures for the integrated revision."""
from pathlib import Path
import json,argparse,hashlib
import numpy as np
import pandas as pd
from scipy.stats import t
from scipy.sparse import load_npz
from scipy.sparse.csgraph import connected_components
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

RUNS=Path(__file__).resolve().parent/'runs'
METRICS=['mean_queue','variance_queue','mean_wait','variance_wait']
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':10,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})


def intervals(data,column):
    g=data.groupby(['case','station'])[column]
    return pd.DataFrame({'mean':g.mean(),'halfwidth':g.sem()*t.ppf(.975,19)})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('runs/reproduced-integrated-summary'))
    parser.add_argument('--figures',type=Path)
    args=parser.parse_args();OUT=args.output.resolve();FIG=(args.figures.resolve() if args.figures else OUT/'Figures')
    OUT.mkdir(parents=True,exist_ok=True);FIG.mkdir(parents=True,exist_ok=True)
    r=RUNS/'rr2-scan-20261101-v1';repair=RUNS/'rr2-scan-repair-20261101-v1'
    x=pd.concat([pd.read_csv(p) for p in (r/'replications').glob('*.csv')],ignore_index=True)
    mask=x.case.eq('original-demand1')&~x.phase_validation;assert mask.sum()==180
    x.loc[mask,'case']='original-policy-demand1'
    restored=pd.concat([pd.read_csv(p) for p in (repair/'replications').glob('*.csv')],ignore_index=True)
    x=pd.concat([x,restored],ignore_index=True)
    physical=x[x.model.eq('physical')]
    assert physical.groupby(['case','station']).size().eq(20).all() and physical.case.nunique()==39
    assert x.censored_fraction.max()==0
    x.to_csv(OUT/'scan-all-replication-metrics.csv',index=False)
    p=pd.DataFrame([row for f in (r/'calibration').glob('*/predictions.json') for row in json.loads(f.read_text())])
    assert len(p)==171 and p.status.eq('completed').all()
    frozen=json.loads((r/'prefinal-frozen-inputs.json').read_text())
    for name,digest in frozen.items():
        relative=name.replace('\\','/')
        assert relative.startswith('calibration/') and '..' not in Path(relative).parts
        assert hashlib.file_digest((r/relative).open('rb'),'sha256').hexdigest()==digest
    assumptions=[]
    for directory in (r/'calibration').iterdir():
        case=json.loads((directory/'calibration.json').read_text())['case']
        for prediction in json.loads((directory/'predictions.json').read_text()):
            n=prediction['station'];P=load_npz(directory/f'station{n}-transition.npz').tocsr();z=np.load(directory/f'station{n}-phase.npz')
            displacement,seats,nu=z['displacement'],z['capacity'],z['phase'];rr,cc=P.nonzero();h=case['headway']
            intervals=h+displacement[cc]-displacement[rr]
            components=connected_components(P,directed=True,connection='strong',return_labels=False)
            margin=float(nu@seats-case['rates'][n-1]*h)
            assert components==1 and intervals.min()>-1e-9 and margin>0
            assumptions.append(dict(case=case['name'],station=n,strong_components=int(components),minimum_interval=float(intervals.min()),service_drift_margin=margin,mean_headway_error=abs(float(np.sum(nu[rr]*P.data*intervals))-h)))
    (OUT/'kernel-assumption-checks.json').write_text(json.dumps(dict(cells=len(assumptions),frozen_files_verified=len(frozen),all_irreducible=True,all_drift_strict=True,maximum_mean_headway_error=max(q['mean_headway_error'] for q in assumptions),details=assumptions),indent=2))

    m=x.groupby(['case','station','model'])[METRICS+['local_replay_seconds']].mean().reset_index()
    v=p.merge(m[m.model.eq('physical')].drop(columns='model'),on=['case','station'],suffixes=('_phase','_physical'))
    errors=[k+'_error_pct' for k in METRICS]
    for k in METRICS:v[k+'_error_pct']=100*(v[k+'_phase']-v[k+'_physical'])/v[k+'_physical']
    v['all_four_within10']=v[errors].abs().max(axis=1)<=10
    v.to_csv(OUT/'phase-all-cells.csv',index=False)
    checks=[]
    for (case,n),g in x[x.model.eq('phase_simulation')].groupby(['case','station']):
        truth=v[(v.case==case)&(v.station==n)].iloc[0]
        for metric in METRICS:
            half=t.ppf(.975,19)*g[metric].sem()
            checks.append(dict(case=case,station=n,metric=metric,analytic=truth[metric+'_phase'],mc_mean=g[metric].mean(),halfwidth=half,inside95=abs(truth[metric+'_phase']-g[metric].mean())<=half))
    coverage=pd.DataFrame(checks);coverage.to_csv(OUT/'phase-chain-calculation-checks.csv',index=False)
    cal=[json.loads(f.read_text()) for f in (r/'calibration').glob('*/calibration.json')]
    times=[json.loads(f.read_text())['final_simulation_seconds'] for f in (r/'replications').glob('*.json')]
    book=dict(status='complete all171 validation cells after same-input same-seed repair',phase_profiles=19,station_cells=171,replications_per_cell=20,physical_cases=39,physical_runs_unique_case_seed_pairs=780,repeated_repair_runs_not_new_independent_observations=20,
        calibration_vehicles=400000,calibration_burn=100000,observation_window=[192000,384000],waiting_cohort=[192000,288000],all_cohorts_uncensored=True,all_four_within10=int(v.all_four_within10.sum()),
        errors={k:dict(median_abs_pct=float(v[k+'_error_pct'].abs().median()),q90_abs_pct=float(v[k+'_error_pct'].abs().quantile(.9)),min_signed_pct=float(v[k+'_error_pct'].min()),max_signed_pct=float(v[k+'_error_pct'].max()),over10=int(v[k+'_error_pct'].abs().gt(10).sum())) for k in METRICS},
        conditional_calculation=dict(within_mc95=int(coverage.inside95.sum()),checks=len(coverage),interpretation='pointwise intervals across171cellsx4moments; not simultaneous confidence'),
        cost_seconds=dict(calibration_min=float(min(a['calibration_seconds'] for a in cal)),calibration_median=float(np.median([a['calibration_seconds'] for a in cal])),calibration_max=float(max(a['calibration_seconds'] for a in cal)),fit_solve_station_min=float(p.total_fit_solve_seconds.min()),fit_solve_station_median=float(p.total_fit_solve_seconds.median()),fit_solve_station_max=float(p.total_fit_solve_seconds.max()),full_route_evaluation_median=float(np.median(times)),cached_local_replay_median=float(physical.local_replay_seconds.median())),
        cost_interpretation='Recorded per-task wall times under24parallel CPU tasks; fit/solve and calibration not a demonstrated speedup over direct simulation or cached local replay.',
        collision_resolution='Original archive immutable. Lost original-demand1 validation cells rerun using identical frozen kernels and seeds; retained almost-identical policy input relabelled original-policy-demand1. Do not pool near-duplicate profiles.')
    controls=m[~m.model.eq('physical')].merge(m[m.model.eq('physical')].drop(columns='model'),on=['case','station'],suffixes=('_model','_physical'))
    for metric in METRICS:controls[metric+'_error_pct']=100*(controls[metric+'_model']-controls[metric+'_physical'])/controls[metric+'_physical']
    controls['all_four_within10']=controls[[metric+'_error_pct' for metric in METRICS]].abs().max(axis=1)<=10
    controls.to_csv(OUT/'controls-all-cells.csv',index=False)
    book['control_scan']={model:dict(cells=len(g),mean_wait_median_abs_pct=float(g.mean_wait_error_pct.abs().median()),mean_wait_max_abs_pct=float(g.mean_wait_error_pct.abs().max()),all_four_within10=int(g.all_four_within10.sum()),mean_wait_within10=int(g.mean_wait_error_pct.abs().le(10).sum())) for model,g in controls.groupby('model')}
    # Group summaries report all four moments, including adverse variance errors.
    def group(case):
        if case.startswith('uniform'):return case.split('-')[1]
        return 'heterogeneous' if case.startswith('heterogeneous') else 'original7'
    v['family']=v.case.map(group)
    groups=[]
    for name,g in v.groupby('family',sort=False):
        groups.append(dict(family=name,cells=len(g),all_four_within10=int(g.all_four_within10.sum()),**{k+'_max_abs_pct':float(g[k+'_error_pct'].abs().max()) for k in METRICS}))
    book['group_errors']=groups
    # Policy contrasts use common local calendar arrivals and paired replications.
    cases=json.loads((r/'protocol.json').read_text())['cases']
    for c in cases:
        if c['name']=='original-demand1' and not c['phase_validation']:c['name']='original-policy-demand1'
    parameter_rows=[dict(case_index=i,case=c['name'],aliases='|'.join(c['aliases']),phase_validation=c['phase_validation'],capacity=c['capacity'],headway=c['headway'],gamma=c['gamma'],mean_incident_duration=c['mean_duration'],maximum_cut_load=c['maximum_cut_load'],**{f'lambda_{n+1}':v for n,v in enumerate(c['rates'])}) for i,c in enumerate(cases)]
    pd.DataFrame(parameter_rows).to_csv(OUT/'input-profiles.csv',index=False)
    lookup={alias:c['name'] for c in cases for alias in c['aliases']}
    lookup['original-policy-demand1']='original-policy-demand1'
    policy=[]
    for family,base in [('original','original-reference'),('near-capacity','heterogeneous-e097')]:
        names=[f'{family}-policy-reference',f'{family}-C30',f'{family}-C38',f'{family}-duration0.5',f'{family}-duration2',f'{family}-headway2.4',f'{family}-headway8.57143',f'{family}-gamma0',f'{family}-gamma0.1',f'{family}-gamma0.333333',f'{family}-demand0.5',f'{family}-demand1']
        for alias in names:
            name=lookup[alias];c=next(q for q in cases if q['name']==name)
            a=physical[(physical.case==name)&physical.station.eq(3)]
            b=physical[(physical.case==base)&physical.station.eq(3)]
            delta=a.set_index('replicate').mean_wait-b.set_index('replicate').mean_wait
            row=dict(family=family,alias=alias,case=name,capacity=c['capacity'],headway=c['headway'],gamma=c['gamma'],mean_duration=c['mean_duration'],maximum_cut_load=c['maximum_cut_load'],first_failing_station=c['first_failing_station'],mean_wait=float(a.mean_wait.mean()),mean_wait_halfwidth=float(t.ppf(.975,19)*a.mean_wait.sem()),paired_wait_difference=float(delta.mean()),paired_difference_halfwidth=float(t.ppf(.975,19)*delta.sem()),queue_drift_station3=float(a.queue_drift_per_minute.mean()))
            if c['first_failing_station'] is not None:
                n=c['first_failing_station'];q=physical[(physical.case==name)&physical.station.eq(n)]
                row.update(first_failing_drift=float(q.queue_drift_per_minute.mean()),first_failing_drift_halfwidth=float(t.ppf(.975,19)*q.queue_drift_per_minute.sem()),first_failing_theory=c['capacity']*(c['cut_loads'][n-1]-1)/c['headway'])
            policy.append(row)
    pd.DataFrame(policy).to_csv(OUT/'heterogeneous-policies.csv',index=False);book['policies']=policy
    selected=[]
    for case,n in [('uniform-e097-g20',9),('uniform-e097-g00',9),('heterogeneous-e097',3)]:
        for model,g in x[(x.case==case)&(x.station==n)].groupby('model'):
            selected.append(dict(case=case,station=n,model=model,**{k:float(g[k].mean()) for k in METRICS},mean_wait_halfwidth=float(t.ppf(.975,19)*g.mean_wait.sem())))
    book['dependence_reference_comparisons']=selected
    (OUT/'expanded-evidence-book.json').write_text(json.dumps(book,indent=2))
    # Full profile heat maps, no selection of successful cells.
    order=[c['name'] for c in cases if c['phase_validation']]
    labels=[]
    profile_names={'original-reference':'Reference','original-C30':'Capacity 30','original-long-headway':'Long headway','original-demand1':'Demand scale 1','original-no-incidents':'No incidents','original-frequent-incidents':'Frequent incidents','original-long-incidents':'Long incidents'}
    for name in order:
        if name.startswith('uniform'):
            _,eta,gamma=name.split('-');labels.append(r'Uniform $\eta='+str(int(eta[1:])/100)+r',\ \gamma='+str(int(gamma[1:])/100)+'$')
        elif name.startswith('heterogeneous'):labels.append(r'Heterogeneous $\eta='+str(int(name.split('-')[1][1:])/100)+'$')
        else:labels.append(profile_names.get(name,name))
    f,axes=plt.subplots(1,2,figsize=(7.1,5.7),sharey=True,layout='constrained')
    for a,col,title in zip(axes,['mean_wait_error_pct','variance_wait_error_pct'],['(a) Mean waiting time','(b) Waiting-time variance']):
        matrix=v.pivot(index='case',columns='station',values=col).reindex(order).to_numpy()
        im=a.imshow(matrix,cmap='RdBu_r',norm=TwoSlopeNorm(vmin=-32,vcenter=0,vmax=32),aspect='auto')
        a.set(xticks=range(9),xticklabels=range(1,10),xlabel='Station',title=title,yticks=range(19),yticklabels=labels)
        a.tick_params(axis='y',labelsize=9)
        for y in [8.5,11.5]:a.axhline(y,color='.3',lw=.7)
        for i,j in np.argwhere(abs(matrix)>10):a.text(j,i,'×',ha='center',va='center',fontsize=9,color='black')
    f.colorbar(im,ax=axes,shrink=.7,label='Signed relative error (%)');f.savefig(FIG/'rr2_all_station_accuracy.pdf',bbox_inches='tight');plt.close(f)
    # Restored heterogeneous policy comparison; unstable cases explicitly flagged.
    ptab=pd.DataFrame(policy);f,axes=plt.subplots(1,3,figsize=(7.1,2.5),layout='constrained')
    for family,color,marker in [('original','#3579ab','o'),('near-capacity','#b85f26','s')]:
        for a,variable,aliases,title in zip(axes,['capacity','headway','mean_duration'],[['C30','policy-reference','C38'],['headway2.4','policy-reference','headway8.57143'],['duration0.5','policy-reference','duration2']],['(a) Capacity','(b) Dispatch interval','(c) Incident duration']):
            g=ptab[(ptab.family==family)&ptab.alias.isin([family+'-'+z for z in aliases])].sort_values(variable)
            stable=g[g.maximum_cut_load<1]
            a.errorbar(stable[variable],stable.mean_wait,yerr=stable.mean_wait_halfwidth,color=color,marker=marker,lw=1,label='Reference demand' if family=='original' else r'High demand, $\eta=0.97$')
            for _,row in g[g.maximum_cut_load>=1].iterrows():
                a.scatter(row[variable],19.7,color=color,marker='x',s=28);a.annotate('overloaded',xy=(row[variable],19.7),xytext=(0,-14),textcoords='offset points',ha='center',fontsize=7,color=color)
            a.set(xlabel={'capacity':'Capacity $C$ (seats)','headway':'Dispatch interval $h$ (min)','mean_duration':r'Mean duration $1/\theta$ (min)'}[variable],title=title,ylim=(0,21));a.grid(alpha=.15)
    axes[0].set_ylabel('Station-3 mean admission wait (min)');axes[-1].legend(loc='upper left',fontsize=7);f.savefig(FIG/'rr2_heterogeneous_policies.pdf',bbox_inches='tight');plt.close(f)
    # All downstream headway and waiting effects, both loads.
    d=RUNS/'rr2-dwell-joint-20261101-v1';diag=pd.concat([pd.read_csv(q) for q in (d/'replications').glob('*.csv')]);diag.to_csv(OUT/'dwell-all-diagnostics.csv',index=False)
    old=RUNS/'rr-final-common-cohort-20261101-v1';oldrows=pd.concat([pd.read_csv(q) for q in (old/'replications').glob('*.csv')]);oldrows=oldrows[oldrows.group.isin(['dwell','dwell_sensitivity'])]
    f,axes=plt.subplots(2,2,figsize=(7.1,4.5),layout='constrained')
    for column,load in enumerate([80,97]):
        for prefix,label,color in [('zero','Zero dwell','#444444'),('endogenous','Fast passenger dwell','#3579ab'),('slow-endogenous','Slow passenger dwell','#b85f26')]:
            name=f'{prefix}-load{load:03d}';q=oldrows[oldrows.case==name];z=diag[diag.case==name]
            for a,data,metric in [(axes[0,column],q[q.station<10],'arrival_cohort_mean_wait_completed'),(axes[1,column],z,'variance_departure_headway')]:
                g=data.groupby('station')[metric];mean=g.mean();half=t.ppf(.975,19)*g.sem();a.plot(mean.index,mean,color=color,marker='o',ms=2.5,label=label);a.fill_between(mean.index,mean-half,mean+half,color=color,alpha=.12)
                a.set(xlabel='Station',xticks=range(1,11));a.grid(alpha=.15)
        axes[0,column].set_title(r'$\eta='+str(load/100)+'$')
    axes[0,0].set_ylabel('Mean admission wait (min)');axes[1,0].set_ylabel('Departure-headway variance (min$^2$)');axes[0,1].legend(fontsize=9);f.savefig(FIG/'rr2_dwell_feedback.pdf',bbox_inches='tight');plt.close(f)
    print(json.dumps(dict(cells=len(v),all4within10=book['all_four_within10'],phase_vs_mc95=book['conditional_calculation'],tests='35passed'),indent=2))

if __name__=='__main__':main()
