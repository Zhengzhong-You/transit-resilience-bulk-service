"""Direct and physical moments of the two-boarding-station example at each simulation horizon."""
from pathlib import Path
import json,argparse
import pandas as pd
from scipy.stats import t

RUNS=Path(__file__).resolve().parent/'runs'
METRICS=['mean_residual','mean_queue','variance_queue','mean_wait','variance_wait']
PHYSICAL=['rr2-dwell-joint-20261101-v1','rr2-two-stop-long-20261101-v1','rr2-two-stop-4800k-20261101-v1']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--direct',type=Path,default=RUNS/'rr2-two-stop-direct-20261101-v1')
    parser.add_argument('--physical',type=Path,nargs='+',default=[RUNS/name for name in PHYSICAL])
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    # The largest numerical queue cap of each case defines its direct value.
    direct={}
    for row in json.loads((args.direct/'results.json').read_text()):
        if row['case'] not in direct or row['cutoff']>direct[row['case']]['cutoff']:direct[row['case']]=row
    rows=[]
    for run in args.physical:
        vehicles=json.loads((run/'protocol.json').read_text())['toy_vehicles']
        x=pd.concat([pd.read_csv(f) for f in sorted((run/'joint').glob('*-rep*.csv'))],ignore_index=True)
        assert x.censored_fraction.max()==0 and x.groupby(['case','station']).size().eq(20).all()
        for (case,station),g in x.groupby(['case','station']):
            for metric in METRICS:
                exact=direct[case][metric][station-1];estimate=float(g[metric].mean());half=float(t.ppf(.975,len(g)-1)*g[metric].sem())
                rows.append(dict(case=case,station=station,metric=metric,vehicles=vehicles,replications=len(g),cutoff=direct[case]['cutoff'],direct=exact,physical=estimate,halfwidth=half,inside95=abs(estimate-exact)<=half))
    table=pd.DataFrame(rows).sort_values(['case','station','metric','vehicles'])
    table.to_csv(args.output/'direct-joint-horizons.csv',index=False)
    print(table.to_string(index=False))
    longest=table[table.vehicles.eq(table.vehicles.max())]
    print('all direct moments inside 95% intervals at',longest.vehicles.iloc[0],'vehicles:',bool(longest.inside95.all()))

if __name__=='__main__':main()
