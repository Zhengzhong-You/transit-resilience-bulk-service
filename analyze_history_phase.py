"""Accuracy of the history-augmented phase against the unchanged physical validation sample."""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
import pandas as pd
from scipy.stats import t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

RUNS = Path(__file__).resolve().parent / 'runs'
METRICS = ['mean_queue', 'variance_queue', 'mean_wait', 'variance_wait']
plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 10, 'pdf.fonttype': 42, 'ps.fonttype': 42,
                     'axes.spines.top': False, 'axes.spines.right': False})


def family(case):
    if case.startswith('uniform'):
        return case.split('-')[1]
    return 'heterogeneous' if case.startswith('heterogeneous') else 'original7'


def replications(scan, repair):
    """All validation-sample replication rows, with the repaired profile as in the integrated analysis."""
    x = pd.concat([pd.read_csv(p) for p in (scan / 'replications').glob('*.csv')], ignore_index=True)
    mask = x.case.eq('original-demand1') & ~x.phase_validation
    assert mask.sum() == 180
    x.loc[mask, 'case'] = 'original-policy-demand1'
    restored = pd.concat([pd.read_csv(p) for p in (repair / 'replications').glob('*.csv')], ignore_index=True)
    return pd.concat([x, restored], ignore_index=True)


def predictions(run):
    return pd.DataFrame([row for f in (run / 'calibration').glob('*/predictions.json') for row in json.loads(f.read_text())])


def errors(prediction, physical):
    v = prediction.merge(physical, on=['case', 'station'], suffixes=('_phase', '_physical'))
    for k in METRICS:
        v[k + '_error_pct'] = 100 * (v[k + '_phase'] - v[k + '_physical']) / v[k + '_physical']
    v['all_four_within10'] = v[[k + '_error_pct' for k in METRICS]].abs().max(axis=1) <= 10
    v['family'] = v.case.map(family)
    return v


def summary(v):
    return dict(cells=len(v), all_four_within10=int(v.all_four_within10.sum()),
                **{k: dict(median_abs_pct=float(v[k + '_error_pct'].abs().median()),
                           max_abs_pct=float(v[k + '_error_pct'].abs().max()),
                           over10=int((v[k + '_error_pct'].abs() > 10).sum())) for k in METRICS})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--history', type=Path, default=RUNS / 'rr2-history-phase-20261102-v1')
    parser.add_argument('--scan', type=Path, default=RUNS / 'rr2-scan-20261101-v1')
    parser.add_argument('--repair', type=Path, default=RUNS / 'rr2-scan-repair-20261101-v1')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--figures', type=Path)
    args = parser.parse_args()
    out = args.output.resolve(); fig_dir = (args.figures or out / 'Figures').resolve()
    out.mkdir(parents=True, exist_ok=True); fig_dir.mkdir(parents=True, exist_ok=True)
    frozen = json.loads((args.history / 'frozen-kernels.json').read_text())
    for name, digest in frozen.items():
        assert hashlib.file_digest((args.history / name).open('rb'), 'sha256').hexdigest() == digest
    x = replications(args.scan, args.repair)
    physical = x[x.model.eq('physical')]
    assert physical[physical.phase_validation].groupby(['case', 'station']).size().eq(20).all()
    means = physical.groupby(['case', 'station'])[METRICS].mean().reset_index()
    history = predictions(args.history)
    first = predictions(args.scan)
    assert len(history) == len(first) == 171 and history.status.eq('completed').all()
    v = errors(history[['case', 'station'] + METRICS], means)
    w = errors(first[['case', 'station'] + METRICS], means)
    v.to_csv(out / 'history-phase-all-cells.csv', index=False)
    both = v.merge(w[['case', 'station'] + [k + '_error_pct' for k in METRICS] + ['all_four_within10']],
                   on=['case', 'station'], suffixes=('', '_first_order'))
    both.to_csv(out / 'history-versus-first-order.csv', index=False)
    groups = []
    for name, g in both.groupby('family', sort=False):
        groups.append(dict(family=name, cells=len(g), all_four_within10=int(g.all_four_within10.sum()),
                           all_four_within10_first_order=int(g.all_four_within10_first_order.sum()),
                           **{k + '_max_abs_pct': float(g[k + '_error_pct'].abs().max()) for k in METRICS},
                           **{k + '_max_abs_pct_first_order': float(g[k + '_error_pct_first_order'].abs().max()) for k in METRICS}))
    # Fitted-chain simulation of the frozen history kernels against their own stationary moments.
    chain = pd.read_csv(args.history / 'fitted-chain-checks.csv')
    checks = []
    for (case, n), g in chain.groupby(['case', 'station']):
        truth = history[(history.case == case) & (history.station == n)].iloc[0]
        for k in METRICS:
            half = t.ppf(.975, len(g) - 1) * g[k].sem()
            checks.append(dict(case=case, station=n, metric=k, analytic=truth[k], mc_mean=g[k].mean(), halfwidth=half,
                               inside95=abs(truth[k] - g[k].mean()) <= half,
                               relative_discrepancy=abs(truth[k] - g[k].mean()) / abs(truth[k])))
    checks = pd.DataFrame(checks); checks.to_csv(out / 'history-chain-checks.csv', index=False)
    # Trace controls are unchanged; compare their mean-wait accuracy with both phase calculations.
    m = x.groupby(['case', 'station', 'model'])[METRICS].mean().reset_index()
    controls = m[~m.model.eq('physical') & ~m.model.eq('phase_simulation')].merge(
        means, on=['case', 'station'], suffixes=('_model', '_physical'))
    for k in METRICS:
        controls[k + '_error_pct'] = 100 * (controls[k + '_model'] - controls[k + '_physical']) / controls[k + '_physical']
    controls['all_four_within10'] = controls[[k + '_error_pct' for k in METRICS]].abs().max(axis=1) <= 10
    scan_table = {model: dict(cells=len(g), mean_wait_median_abs_pct=float(g.mean_wait_error_pct.abs().median()),
                              mean_wait_max_abs_pct=float(g.mean_wait_error_pct.abs().max()),
                              all_four_within10=int(g.all_four_within10.sum()))
                  for model, g in controls.groupby('model')}
    for label, frame in [('history_phase', v), ('first_order_phase', w)]:
        scan_table[label] = dict(cells=len(frame), mean_wait_median_abs_pct=float(frame.mean_wait_error_pct.abs().median()),
                                 mean_wait_max_abs_pct=float(frame.mean_wait_error_pct.abs().max()),
                                 all_four_within10=int(frame.all_four_within10.sum()))
    reference = {}
    for case, n in [('uniform-e097-g20', 9), ('uniform-e097-g00', 9), ('uniform-e097-g02', 9), ('heterogeneous-e097', 3)]:
        row = both[(both.case == case) & (both.station == n)].iloc[0]
        reference[f'{case}-station{n}'] = dict(history={k: float(row[k + '_phase']) for k in METRICS},
            physical={k: float(row[k + '_physical']) for k in METRICS},
            history_error_pct={k: float(row[k + '_error_pct']) for k in METRICS},
            first_order_error_pct={k: float(row[k + '_error_pct_first_order']) for k in METRICS})
    ratios = pd.DataFrame([dict(case=r['case'], station=r['station'],
        history=max(abs(np.log(float(q))) for q in r['net_input_ratio_history'].values()),
        first_order=max(abs(np.log(float(q))) for q in r['net_input_ratio_first_order'].values()))
        for _, r in history.iterrows()])
    book = dict(history=summary(v), first_order=summary(w), group_errors=groups,
        fitted_chain=dict(inside95=int(checks.inside95.sum()), checks=len(checks),
                          median_relative_discrepancy_pct=float(100 * checks.relative_discrepancy.median()),
                          by_metric_median_pct={k: float(100 * g.relative_discrepancy.median()) for k, g in checks.groupby('metric')}),
        control_scan=scan_table, reference_cells=reference,
        resolution=dict(classes=history.backlog_classes.value_counts().sort_index().to_dict(),
                        states_min=int(history.states.min()), states_max=int(history.states.max()),
                        states_median=float(history.states.median()),
                        removed_transition_fraction_max=float(history.removed_transition_fraction.max()),
                        cutoffs=history.cutoff.value_counts().to_dict(),
                        boundary_max=float(history.upper_boundary_probability.max()),
                        balance_max=float(history.balance_residual_l1.max())),
        cost_seconds=dict(fit_solve_min=float((history.fitting_seconds + history.solve_seconds_total).min()),
                          fit_solve_median=float((history.fitting_seconds + history.solve_seconds_total).median()),
                          fit_solve_max=float((history.fitting_seconds + history.solve_seconds_total).max()),
                          note='resolution choice, fitting and all queue cutoffs; excludes the net-input diagnostics'),
        net_input_log_ratio=dict(history_median=float(ratios.history.median()), history_max=float(ratios.history.max()),
                                 first_order_median=float(ratios.first_order.median()), first_order_max=float(ratios.first_order.max()),
                                 history_within5pct=int((ratios.history <= np.log(1.05)).sum()),
                                 first_order_within5pct=int((ratios.first_order <= np.log(1.05)).sum())),
        frozen_files_verified=len(frozen))
    (out / 'history-evidence-book.json').write_text(json.dumps(book, indent=2, default=int))
    # Waiting errors at every station and profile, first-order and history-augmented.
    protocol = json.loads((args.scan / 'protocol.json').read_text())
    order = [c['name'] for c in protocol['cases'] if c['phase_validation']]
    names = {'original-reference': 'Reference', 'original-C30': 'Capacity 30', 'original-long-headway': 'Long headway',
             'original-demand1': 'Demand scale 1', 'original-no-incidents': 'No incidents',
             'original-frequent-incidents': 'Frequent incidents', 'original-long-incidents': 'Long incidents'}
    labels = []
    for name in order:
        if name.startswith('uniform'):
            _, eta, gamma = name.split('-')
            labels.append(r'Uniform $\eta=' + str(int(eta[1:]) / 100) + r',\ \gamma=' + str(int(gamma[1:]) / 100) + '$')
        elif name.startswith('heterogeneous'):
            labels.append(r'Heterogeneous $\eta=' + str(int(name.split('-')[1][1:]) / 100) + '$')
        else:
            labels.append(names.get(name, name))
    f, axes = plt.subplots(1, 3, figsize=(7.1, 5.7), sharey=True, layout='constrained')
    panels = [(both, 'mean_wait_error_pct', '(a) Mean waiting,\nhistory phase'),
              (both, 'variance_wait_error_pct_first_order', '(b) Waiting variance,\nfirst-order phase'),
              (both, 'variance_wait_error_pct', '(c) Waiting variance,\nhistory phase')]
    for a, (frame, column, title) in zip(axes, panels):
        matrix = frame.pivot(index='case', columns='station', values=column).reindex(order).to_numpy()
        im = a.imshow(matrix, cmap='RdBu_r', norm=TwoSlopeNorm(vmin=-32, vcenter=0, vmax=32), aspect='auto')
        a.set(xticks=range(9), xticklabels=range(1, 10), xlabel='Station', yticks=range(19), yticklabels=labels)
        a.set_title(title, fontsize=9)
        a.tick_params(axis='y', labelsize=9); a.tick_params(axis='x', labelsize=8)
        for y in [8.5, 11.5]:
            a.axhline(y, color='.3', lw=.7)
        for i, j in np.argwhere(abs(matrix) > 10):
            a.text(j, i, '×', ha='center', va='center', fontsize=9, color='black')
    f.colorbar(im, ax=axes, shrink=.7, label='Signed relative error (%)')
    f.savefig(fig_dir / 'rr2_history_station_accuracy.pdf', bbox_inches='tight'); plt.close(f)
    print(json.dumps(dict(history=book['history'], first_order_all_four=book['first_order']['all_four_within10'],
                          fitted_chain=book['fitted_chain'], resolution=book['resolution']), indent=2, default=int))


if __name__ == '__main__':
    main()
