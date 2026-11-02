"""Fair special-case roots versus contour benchmark, without generic speed claims."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import argparse,json,platform,shutil,time
from pathlib import Path
import numpy as np
from rootfree_bulk import deterministic_bulk_roots,contour_moments,effective_arrival_pgf,moments_from_roots
from validate_service_revision import write_rows


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    for n in ['benchmark_deterministic_bulk.py','rootfree_bulk.py']:shutil.copy2(Path(__file__).parent/n,a.output/n)
    (a.output/'protocol.json').write_text(json.dumps(dict(capacity=[1,8,34,50],load=[.5,.9,.97,.99],headway=4.8,availability=1,repetitions=3,model='deterministic headways, full capacity',timing='serial on same CPU; pgf already constructed; roots and moments versus contour and moments; excludes interpreter startup',python=platform.python_version(),platform=platform.platform()),indent=2))
    rows=[]
    for c in [1,8,34,50]:
        for rho in [.5,.9,.97,.99]:
            values,derivatives,f=effective_arrival_pgf(c,rho)
            for rep in range(3):
                start=time.perf_counter();z=deterministic_bulk_roots(c,rho);mean,var=moments_from_roots(z,c,derivatives);elapsed=time.perf_counter()-start
                contour=contour_moments(c,values)
                error=max(abs(mean.real-contour['mean_residual'])/max(1.,abs(mean.real)),abs(var.real-contour['variance_residual'])/max(1.,abs(var.real)))
                if max(abs(mean.imag),abs(var.imag))>1e-8 or error>1e-8:raise ArithmeticError('Special-case roots disagree with contour.')
                rows.append(dict(capacity=c,load=rho,replicate=rep,root_mean=float(mean.real),root_variance=float(var.real),root_seconds=elapsed,contour_seconds=contour['elapsed_seconds'],relative_error=error,largest_scaled_root_residual=max(abs(f(x))/f.scale(x) for x in z)))
    write_rows(a.output/'replications.csv',rows);(a.output/'status.json').write_text(json.dumps(dict(status='completed',settings=16,repetitions=48),indent=2))

if __name__=='__main__':main()
