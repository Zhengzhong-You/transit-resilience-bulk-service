"""Canonical replication summaries and paper figures for the completed route suite."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--route-run', type=Path, required=True)
    parser.add_argument('--root-run', type=Path, required=True)
    parser.add_argument('--dwell-run', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    files = sorted((args.route_run / 'replications').glob('*.csv'))
    if args.dwell_run:
        files += sorted((args.dwell_run / 'replications').glob('*.csv'))
    data = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    keys = ['case', 'group', 'load', 'vehicles', 'station', 'initial_backlog']
    assert not data.duplicated(['case', 'replicate', 'vehicles', 'station']).any()
    assert data.groupby(keys, dropna=False).size().eq(20).all()
    assert data.arrival_cohort_censored_fraction.eq(0.).all()
    # Open-route controls have no incident-affected return trip; their unused
    # return placeholders are not measured fleet cycle times.
    data.loc[~data.group.eq('fleet'), 'mean_vehicle_cycle'] = float('nan')
    metrics = ['mean_headway', 'variance_headway', 'variance_departure_headway', 'headway_lag1_correlation',
        'mean_residual', 'terminal_residual', 'queue_drift_per_minute', 'mean_queue', 'variance_queue',
        'mean_available_capacity', 'mean_boarded', 'mean_alighted', 'mean_dwell',
        'arrival_cohort_mean_wait_completed', 'arrival_cohort_variance_wait_completed',
        'achieved_origin_headway', 'origin_lateness_drift', 'mean_vehicle_cycle']
    records = []
    for group, frame in data.groupby(keys, dropna=False):
        for metric in metrics:
            values = frame[metric].dropna()
            if len(values) == 0:
                continue
            mean, se = values.mean(), values.std(ddof=1) / np.sqrt(len(values))
            records.append(dict(zip(keys, group), metric=metric, mean=mean,
                                lower95=mean-t.ppf(.975, 19)*se,
                                upper95=mean+t.ppf(.975, 19)*se, replications=len(values)))
    summary = pd.DataFrame(records)
    summary.to_csv(args.output / 'summary.csv', index=False)
    final = data[data.vehicles.eq(80000)]
    pairs = []
    for load in [.8, .97]:
        reference = final[final['case'].eq(f'zero-load{int(load * 100):03d}')]
        choices = final[final.load.eq(load) & final.group.isin(['dwell', 'dwell_sensitivity'])]['case'].unique()
        for case in choices:
            alternative = final[final['case'].eq(case)]
            merged = alternative.merge(reference, on=['station', 'replicate'], suffixes=('_alt', '_base'), validate='one_to_one')
            for n, g in merged.groupby('station'):
                if n == 10:
                    continue
                for metric in ['variance_headway', 'mean_residual', 'arrival_cohort_mean_wait_completed']:
                    delta = g[metric+'_alt']-g[metric+'_base']
                    mean, half = delta.mean(), t.ppf(.975, 19)*delta.std(ddof=1)/np.sqrt(20)
                    pairs.append({'load':load,'case':case,'station':n,'metric':metric,
                        'reference_mean':g[metric+'_base'].mean(),'alternative_mean':g[metric+'_alt'].mean(),
                        'paired_mean_difference':mean,'paired_lower95':mean-half,'paired_upper95':mean+half,
                        'percent_change':100*mean/g[metric+'_base'].mean() if abs(g[metric+'_base'].mean())>1e-12 else float('nan')})
    pd.DataFrame(pairs).to_csv(args.output/'paired_dwell_effects.csv',index=False)
    selected = final[final.station.eq(9)].groupby(['case', 'group', 'load'])[metrics].mean()
    selected.to_csv(args.output / 'station9_table.csv')

    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':11.5,'axes.spines.top':False,'axes.spines.right':False,
                         'pdf.fonttype':42,'ps.fonttype':42})
    fig, axes = plt.subplots(1,2,figsize=(8,3.3))
    boundary = summary[summary.group.eq('boundary') & summary.station.eq(9) & summary.metric.eq('mean_residual')]
    for ax, initial in zip(axes,[0,1000]):
        for load, color in zip([.8,.97,1.,1.01,1.02],['#222222','#0055CC','#777777','#D55E00','#AA3377']):
            rows = boundary[boundary.load.eq(load)&boundary.initial_backlog.eq(initial)].sort_values('vehicles')
            ax.errorbar(rows.vehicles,rows['mean'],yerr=[rows['mean']-rows.lower95,rows.upper95-rows['mean']],
                        color=color,marker='o',markersize=3,capsize=2,label=f'{load:.2f}')
        ax.set(xscale='log',yscale='log',xlabel='Vehicle runs',ylabel='Mean residual queue (passengers)',
               title=f'Initial queue {initial}')
        ax.set_xticks([5000,20000,80000],['5,000','20,000','80,000'])
        ax.set_xticks([], minor=True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title='Maximum cut load', loc='lower center',
               bbox_to_anchor=(.5, -.12), ncol=5, fontsize=10, title_fontsize=10, frameon=False)
    fig.tight_layout();fig.savefig(args.output/'route_stability.pdf', bbox_inches='tight');plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(8,3.3))
    for load,ax in zip([.8,.97],axes):
        for mode,color in zip(['zero','constant','exogenous','endogenous'],['#222222','#777777','#D55E00','#0055CC']):
            rows=summary[summary['case'].eq(f'{mode}-load{int(load*100):03d}')
                &summary.vehicles.eq(80000)&summary.metric.eq('arrival_cohort_mean_wait_completed')&summary.station.lt(10)]
            ax.plot(rows.station,rows['mean'],marker='o',markersize=3,color=color,label=mode.capitalize())
        ax.set(xlabel='Station',ylabel='Mean passenger waiting time (min)',title=f'Maximum cut load {load:.2f}')
        ax.set_xticks([1,3,5,7,9])
    axes[0].legend(fontsize=10)
    fig.tight_layout();fig.savefig(args.output/'dwell_waiting.pdf');plt.close(fig)
    root=pd.read_csv(args.root_run/'replications.csv')
    roots=[]
    for capacity,frame in root.groupby('capacity'):
        successful=frame[frame.roots_status.eq('completed')]
        roots.append({'capacity':int(capacity),'settings':len(frame)//3,
                      'root_successful_settings':int(frame.assign(success=frame.roots_status.eq('completed')).groupby(['load','availability','sigma']).success.all().sum()),
                      'median_contour_seconds_all':frame.contour_seconds.median(),
                      'median_contour_seconds_paired':successful.contour_seconds.median(),
                      'median_successful_root_seconds':successful.roots_seconds.median(),
                      'max_mean_absolute_error':(successful.roots_mean-successful.contour_mean).abs().max(),
                      'max_variance_absolute_error':(successful.roots_variance-successful.contour_variance).abs().max()})
    pd.DataFrame(roots).to_csv(args.output/'rootfree_table.csv',index=False)
    manifest={'sources':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
              'root_source':str(args.root_run/'replications.csv'),
              'cohort_maximum_censored_fraction':float(data.arrival_cohort_censored_fraction.max()),
              'confidence_intervals':'two-sided t intervals across20 independent replications, paired for dwell differences; marginal, not simultaneous',
              'scope':'finite-horizon cohort outcomes; overloaded cases do not report stationary waiting moments; cycle times reported only for explicit finite-fleet returns'}
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('summarized',len(files),'replication files; zero censored middle-quarter cohort members')


if __name__=='__main__':
    main()
