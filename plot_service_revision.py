"""Summarize the matched service-clock validation without changing raw outputs."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
import numpy as np
import pandas as pd
from scipy.stats import t


def summarize(data, groups, metrics):
    summary = data.groupby(groups)[metrics].agg(['mean', 'std', 'count'])
    for metric in metrics:
        n = summary[(metric, 'count')]
        summary[(metric, 'ci_half_width')] = (
            t.ppf(0.975, n - 1) * summary[(metric, 'std')] / np.sqrt(n)
        )
    summary.columns = ['_'.join(column) for column in summary.columns]
    return summary.reset_index()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', type=Path, help='directory for summaries and figures; defaults to the run directory')
    args = parser.parse_args()
    out = args.output or args.run
    out.mkdir(parents=True, exist_ok=True)
    headways = pd.read_csv(args.run / 'headway_replications.csv')
    queues = pd.read_csv(args.run / 'stability_replications.csv')
    h = summarize(headways, ['model', 'station'],
                  ['mean_headway', 'sd_headway', 'lag1_correlation', 'zero_fraction'])
    q = summarize(queues, ['model', 'station', 'load_ratio', 'initial_backlog', 'vehicles'],
                  ['backlog_per_time', 'second_half_mean_backlog', 'second_half_growth_per_time'])
    h.to_csv(out / 'headway_summary.csv', index=False)
    q.to_csv(out / 'stability_summary.csv', index=False)
    protocol = json.loads((args.run / 'protocol.json').read_text())
    names = {'propagated_fifo': 'Physical FIFO', 'fifo_permuted': 'Permuted FIFO intervals',
             'rate_matched_normal': 'Mean-matched iid', 'unscaled_normal': 'Unscaled iid'}
    colors = {'propagated_fifo': '#222222', 'fifo_permuted': '#009E73',
              'rate_matched_normal': '#0072B2', 'unscaled_normal': '#D55E00'}
    markers = {'propagated_fifo': 'o', 'fifo_permuted': 's',
               'rate_matched_normal': '^', 'unscaled_normal': 'D'}
    plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 12, 'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.4), layout='constrained')
    for model in ['propagated_fifo', 'rate_matched_normal', 'unscaled_normal']:
        data = h[h.model == model]
        axes[0].errorbar(data.station, data.mean_headway_mean,
                         yerr=data.mean_headway_ci_half_width, label=names[model],
                         marker=markers[model], color=colors[model], ms=3, capsize=2)
    axes[0].axhline(protocol['h'], color='#777777', lw=0.8, ls=':')
    axes[0].set(xlabel='Station', ylabel='Mean headway (min)', title='(a) Service clock')
    axes[0].legend(fontsize=11.0, frameon=False, loc='upper left')
    for model in names:
        data = q[(q.model == model) & (q.initial_backlog == 0) & (q.load_ratio == 0.97)]
        axes[1].errorbar(data.vehicles, data.second_half_mean_backlog_mean,
                         yerr=data.second_half_mean_backlog_ci_half_width,
                         marker=markers[model], color=colors[model], ms=4, capsize=2,
                         label=names[model])
        data = q[(q.model == model) & (q.initial_backlog == 0)
                 & (q.vehicles == protocol['vehicles'])]
        axes[2].errorbar(data.load_ratio, data.second_half_growth_per_time_mean,
                         yerr=data.second_half_growth_per_time_ci_half_width,
                         marker=markers[model], color=colors[model], ms=4, capsize=2)
    axes[1].set(xscale='log', yscale='log', xlabel='Vehicle intervals',
                ylabel='Mean residual queue (passengers)', title='(b) Nominal load 0.97')
    axes[1].set_xticks(protocol['horizons'], [f'{n:,}' for n in protocol['horizons']])
    axes[1].xaxis.set_minor_locator(NullLocator())
    axes[1].legend(fontsize=11.0, frameon=False)
    axes[2].axhline(0, color='#777777', lw=0.8)
    axes[2].axvline(1, color='#777777', lw=0.8, ls=':')
    axes[2].set(xlabel=r'Nominal load $\lambda h/C$',
                ylabel='Residual growth (passengers/min)', title='(c) Capacity boundary')
    fig.savefig(out / 'service_diagnostics.pdf', bbox_inches='tight')
    fig.savefig(out / 'service_diagnostics.png', dpi=180, bbox_inches='tight')
    print(h[(h.station == 8)][['model', 'mean_headway_mean', 'lag1_correlation_mean']].to_string(index=False))
    print(q[(q.vehicles == protocol['vehicles']) & (q.initial_backlog == 0)
            & (q.load_ratio == 0.97)][['model', 'second_half_mean_backlog_mean',
                                     'second_half_mean_backlog_ci_half_width']].to_string(index=False))
    print(out / 'service_diagnostics.png')


if __name__ == '__main__':
    main()
