"""Selected frozen physical queries through unchanged Arb, with explicit limits."""
import argparse
import json
from pathlib import Path
import time

from ..models import load
from ..queries import FieldOracle,Limits
from ..study import atomic_json
from .api import MaximumOracle,CertificateChecker


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('outputs/gaussian-b2-max-cert-v1'))
    p.add_argument('--input',type=Path,default=Path('outputs/gaussian-geometry-v1/distribution-v2'));p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    fixture=json.loads((a.root/'frozen-rays.json').read_text());results=[]
    for case in sorted({r['case'] for r in fixture['rays']}):
        rows=[r for r in fixture['rays'] if r['case']==case];chosen={}
        for group,count in [('expensive_MISS',2),('near_threshold_MISS_heuristic',2),('expensive_HIT',2)]:
            selected=sorted([r for r in rows if group in r['strata']],key=lambda r:r['ray_id'])[:count]
            for row in selected:chosen[row['ray_id']]=row
        scene=load(a.input/case/'geometry.json');t=time.perf_counter();arb=FieldOracle(scene,Limits(precision=(128,256,512),max_nodes=32768,max_depth=48,seconds=4.));setup=time.perf_counter()-t
        solver=MaximumOracle(scene,endpoint_probes=False);checker=CertificateChecker(scene)
        for row in chosen.values():
            s,x=row['source_xyz'],row['receiver_xyz'];t=time.perf_counter();decision=arb.segment(s,x);wall=time.perf_counter()-t
            assert not decision.certified or decision.status==row['expected'],('ARBITRARY PRECISION DISAGREEMENT',row,decision)
            result=solver.segment(s,x,proof=True);assert result['status']==row['expected']
            checked=checker.check(result['certificate'],s,x,precision=256) if result['certificate'] else None
            if checked:assert checked['accepted']
            results.append({'ray_id':row['ray_id'],'case':case,'expected':row['expected'],'Arb_status':decision.status,'Arb_reason':decision.reason,'Arb_nodes':decision.nodes,
                            'Arb_seconds':wall,'Arb_geometry_setup_seconds':setup,'new_solver_status':result['status'],'checker_accepted':checked['accepted'] if checked else None})
    atomic_json(a.output,{'queries':results,'limits':{'precision':[128,256,512],'max_nodes':32768,'max_depth':48,'seconds':4.},
                          'no_opposite_physical_labels':True,'unresolved_Arb_results_are_not_physical_rejection':True})
    print(json.dumps({'third_path_queries':len(results),'resolved':sum(r['Arb_status'] in ('MISS','HIT') for r in results),'unresolved':sum(r['Arb_status'] not in ('MISS','HIT') for r in results)}))


if __name__=='__main__':main()
