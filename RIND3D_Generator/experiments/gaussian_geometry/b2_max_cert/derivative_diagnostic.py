"""Independent Arb diagnostics on a bounded DEVELOPMENT-only baseline trace.

These enclosures diagnose opportunities; they do not supply solver labels.
"""
import argparse
import json
from pathlib import Path

from flint import arb,ctx
import numpy as np
from ..models import load,rational
from ..queries import interval,qrange
from ..study import atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('outputs/gaussian-b2-max-cert-v1'))
    p.add_argument('--input',type=Path,default=Path('outputs/gaussian-geometry-v1/distribution-v2'));args=p.parse_args()
    fixture=json.loads((args.root/'frozen-rays.json').read_text());profile=json.loads((args.root/'baseline-characterization.json').read_text())
    costs={r['ray_id']:r['independent_seconds'] for r in profile['results']};rows=[]
    with ctx.workprec(192):
        for case in sorted({r['case'] for r in fixture['rays']}):
            candidates=sorted([r for r in fixture['rays'] if r['case']==case and r['split']=='development' and r['expected']=='MISS'],key=lambda r:(costs[r['ray_id']],r['ray_id']))
            chosen=[candidates[i] for i in sorted(set([0,len(candidates)//3,len(candidates)//2,*range(max(0,len(candidates)-5),len(candidates))]))]
            scene=load(args.input/case/'geometry.json')
            for row in chosen:
                coefficients=[k.coefficients(row['source_xyz'],row['receiver_xyz']) for k in scene.kernels]
                visits=np.load(args.root/f'visits-{row["ray_id"]}.npz')['visits'];visits=visits[(visits[:,3]==0)&(visits[:,2]<0)&(visits[:,1]>visits[:,0])]
                visits=visits[np.linspace(0,len(visits)-1,min(48,len(visits)),dtype=int)] if len(visits) else visits
                global_second=sum(2*rational(k.weight)*coef[0] for k,coef in zip(scene.kernels,coefficients));bounds=[]
                for l,h,_,_ in visits:
                    l,h=rational(l),rational(h);t=interval(l,h);d1=arb(0);d2=arb(0);terms=[]
                    for k,coef in zip(scene.kernels,coefficients):
                        a,b,c=coef;lo,hi,_=qrange(coef,l,h);g=arb(rational(k.weight))*(-interval(lo,hi)/2).exp();dot=arb(a)*t+arb(b)/2
                        z=-dot*g;terms.append(z);d1+=z;d2+=(dot*dot-arb(a))*g
                    bounds.append({'l':float(l),'h':float(h),'stationary_excluded':bool(d1>0 or d1<0),'strictly_concave':bool(d2<0),
                                   'local_curvature_upper_over_global':float(d2.upper()/arb(global_second)) if global_second else 0,
                                   'midpoint_cancellation_diagnostic':sum(abs(float(z.mid())) for z in terms)/max(abs(float(d1.mid())),1e-300)})
                rows.append({'ray_id':row['ray_id'],'case':case,'intervals':bounds})
    all_bounds=[b for r in rows for b in r['intervals']]
    summary={'development_rays':len(rows),'natural_inconclusive_intervals':len(all_bounds),'stationary_excluded_by_Arb_derivative':sum(b['stationary_excluded'] for b in all_bounds),
             'strictly_concave_intervals':sum(b['strictly_concave'] for b in all_bounds),
             'median_local_curvature_upper_over_old_global':float(np.median([b['local_curvature_upper_over_global'] for b in all_bounds])),
             'median_derivative_cancellation_diagnostic':float(np.median([b['midpoint_cancellation_diagnostic'] for b in all_bounds]))}
    atomic_json(args.root/'derivative-diagnostic.json',{'summary':summary,'results':rows,'method':'192-bit Arb with exact canonical rational ray coefficients; no solver labels; development data only'})
    print(json.dumps(summary))


if __name__=='__main__':main()
