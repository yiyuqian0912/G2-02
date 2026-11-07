"""Bounded numerical adversaries, not a new geometry-generator distribution."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from ..models import Kernel,KernelScene,Threshold,record
from ..b2_native.api import NativeOracle
from ..queries import FieldOracle,Limits
from ..study import atomic_json
from .api import MaximumOracle,CertificateChecker,dumps


def fixtures():
    def scene(centers,scale=.125,weights=None,threshold=Threshold('exp',-.25)):
        return KernelScene('gaussian_sum',[Kernel(c,np.eye(3)*scale,(weights or [.5]*len(centers))[i]) for i,c in enumerate(centers)],threshold)
    tangent=scene([[x,.5,.5] for x in (.4375,.5625)])
    rows=[]
    for shift,name,expected in [(0,'exact_transcendental_contact','HIT'),(2**-20,'near_contact_MISS','MISS'),(-2**-20,'near_contact_HIT','HIT'),
                                (np.nextafter(.5625,np.inf)-.5625,'one_ULP_MISS','MISS'),(np.nextafter(.5625,-np.inf)-.5625,'one_ULP_HIT','HIT')]:
        rows.append((name,tangent,[.5,.5625+shift,.03125],[.5,.5625+shift,.96875],expected))
    quartic=scene([[x,.5,.5] for x in (.375,.625)],threshold=Threshold('exp',-.5))
    rows.append(('quartic_degenerate_contact',quartic,[.0625,.5,.5],[.9375,.5,.5],'HIT'))
    kernels=[Kernel([.125+i*7/64,.5,.5],np.diag([1/64,.04,.04]),.8) for i in range(8)]
    many=KernelScene('gaussian_sum',kernels,Threshold('binary64',.85))
    rows.append(('eight_separated_peaks',many,[.01,.5,.5],[.99,.5,.5],'MISS'))
    m=np.array([[2**-18,2**-20,0],[0,1/32,1/512],[0,0,1/16]])
    thin=KernelScene('gaussian_sum',[Kernel([.5]*3,m,.6),Kernel([.5+2**-20,.5,.5],m,.6)],Threshold('binary64',.9))
    rows.extend([('thin_rotated_fused_HIT',thin,[.05,.5,.5],[.95,.5,.5],'HIT'),('thin_rotated_MISS',thin,[.05,.75,.5],[.95,.75,.5],'MISS')])
    shared=scene([[.5]*3,[.5]*3],scale=1,threshold=Threshold('exp',-.5))
    rows.append(('analytic_closed_source_contact',shared,[1.5,.5,.5],[2.,.5,.5],'HIT'))
    tail=scene([[.5]*3],scale=1,weights=[1.],threshold=Threshold('binary64',1e-300))
    rows.append(('positive_underflow_remainder',tail,[1000]*3,[1001]*3,'MISS'))
    rows.append(('underflow_no_binary64_margin',scene([[.5]*3],scale=1,weights=[1.],threshold=Threshold('binary64',np.nextafter(0.,1.))),[1000]*3,[1001]*3,'MISS'))
    rows.append(('zero_length_FREE',many,[0,0,0],[0,0,0],'MISS'))
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--seconds',type=float,default=.1);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    results=[];descriptions=[]
    for name,scene,s,x,expected in fixtures():
        descriptions.append({'name':name,'canonical_scene':record(scene),'source_hex':[float(v).hex() for v in s],'receiver_hex':[float(v).hex() for v in x],
                             'expected_mathematical_label':expected,'pure_numerical_predicate_not_dataset_source_acceptance':True})
        checker=CertificateChecker(scene)
        for method in ('A_primary','A_independent','B_derivative','C_stationary','D_hybrid','D_proof','E_lean','E_lean_proof','D_old_fallback','E_old_fallback'):
            t=time.perf_counter()
            if method.startswith('A_'):engine=NativeOracle(scene,independent=method=='A_independent',max_nodes=1024,query_seconds=a.seconds)
            else:engine=MaximumOracle(scene,method=1 if method=='B_derivative' else 2 if method=='C_stationary' else 3,max_nodes=1024,max_depth=40,seconds=a.seconds,fallback=method.endswith('fallback'),endpoint_probes=not method.startswith('E_'))
            setup=time.perf_counter()-t;t=time.perf_counter();cpu=time.process_time()
            if method.startswith('A_'):
                decision=engine.segment(s,x);r={'status':decision.status,'reason':decision.reason,'nodes':decision.nodes,'depth':decision.depth,'work':dict(engine.work)}
            else:r=engine.segment(s,x,proof=method.endswith('_proof'))
            wall=time.perf_counter()-t;cpu=time.process_time()-cpu;certificate=r.pop('certificate',None)
            assert r['status'] in (expected,'UNKNOWN'),('FALSE PHYSICAL LABEL',name,method,r)
            row={'name':name,'method':method,'expected':expected,'setup_seconds':setup,'wall_seconds':wall,'CPU_seconds':cpu,**r}
            if certificate:
                t=time.perf_counter();encoded=dumps(certificate);row['serialization_seconds']=time.perf_counter()-t
                t=time.perf_counter();checked=checker.check(encoded,s,x,precision=256,seconds=a.seconds);row.update(checker_wall_seconds=time.perf_counter()-t,checker=checked,proof_bytes=len(encoded))
                row.update(proof_intervals=len(certificate['partition']),proof_operations=dict(Counter(r['op'] for r in certificate['partition'])),proof_SHA256=hashlib.sha256(encoded).hexdigest())
                assert checked['accepted'];(a.output/f'{name}-{method}-certificate.json').write_bytes(encoded)
            results.append(row)
        # Third path is deliberately bounded; exhaustion is not physical evidence.
        t=time.perf_counter();arb=FieldOracle(scene,Limits(precision=(128,256,512),max_nodes=8192,max_depth=40,seconds=.3));d=arb.segment(s,x)
        assert d.status not in ('HIT','MISS') or d.status==expected
        results.append({'name':name,'method':'Arb_third','expected':expected,'status':d.status,'reason':d.reason,'nodes':d.nodes,'wall_seconds':time.perf_counter()-t,'work':dict(arb.work)})
    atomic_json(a.output/'summary.json',{'fixtures':descriptions,'results':results,'all_labels_sound':True,'budget_exhaustion_is_unresolved_not_rejection':True,'production_acceptance_unchanged':True})
    print(json.dumps({'adversarial_fixtures':len(descriptions),'records':len(results),'no_opposite_labels':True}))


if __name__=='__main__':main()
