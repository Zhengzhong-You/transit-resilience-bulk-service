"""Restore an overwritten named validation cell without changing frozen kernels."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import argparse,json,shutil,hashlib,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
from run_revision_scan import evaluate


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--previous',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    protocol=json.loads((a.previous/'protocol.json').read_text())
    matches=[(i,c) for i,c in enumerate(protocol['cases']) if c['name']=='original-demand1' and c['phase_validation']]
    assert len(matches)==1
    i,c=matches[0];a.output.mkdir(parents=True,exist_ok=False)
    for folder in ['calibration','replications','traces','source']:(a.output/folder).mkdir()
    shutil.copytree(a.previous/'calibration'/c['name'],a.output/'calibration'/c['name'])
    frozen={str(p.relative_to(a.output)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (a.output/'calibration').rglob('*') if p.is_file()}
    for name in ['repair_revision_scan.py','run_revision_scan.py','route_validation.py','validate_route_phase.py']:
        shutil.copy2(Path(__file__).parent/name,a.output/'source'/name)
    (a.output/'protocol.json').write_text(json.dumps(dict(purpose='same frozen-kernel evaluation of overwritten original-demand1 validation profile',previous_protocol=protocol,case=c,index=i,replications=20,all_frozen_hashes=frozen,original_results_unmodified=True),indent=2))
    start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=20) as pool:
        futures=[pool.submit(evaluate,(c,i,r,str(a.output),protocol)) for r in range(20)]
        for k,f in enumerate(as_completed(futures),1):
            print(k,f.result(),flush=True)
    for name,digest in frozen.items():assert hashlib.sha256((a.output/name).read_bytes()).hexdigest()==digest
    (a.output/'status.json').write_text(json.dumps(dict(status='completed',replications=20,elapsed_seconds=time.perf_counter()-start),indent=2))

if __name__=='__main__':main()
