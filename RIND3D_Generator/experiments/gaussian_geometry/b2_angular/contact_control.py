"""Supplemental exact transcendental-contact guard for the new field paths."""
import argparse
from pathlib import Path
import json

import numpy as np
from ..models import Kernel,KernelScene,Threshold
from ..b2_native.api import NativeOracle
from ..study import atomic_json
from .api import AngularOracle


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise ValueError('Use a new versioned guard output')
    # Translation of the frozen nontrivial equality fixture puts the tangent
    # ray on voxel-center coordinates of an 8³ field. Its exact maximum is tau.
    scene=KernelScene('gaussian_sum',[Kernel([x,.5,.5],np.eye(3)*.125,.5) for x in (.5,.625)],Threshold('exp',-.25))
    source=[.5625,.5625,.0625];receiver=[.5625,.5625,.9375];rows=[]
    for independent in (False,True):
        native=NativeOracle(scene,independent=independent,max_nodes=512,query_seconds=.1)
        assert native.source_clearance(source).status=='FREE'
        d=native.segment(source,receiver);assert d.status=='UNKNOWN'
        for method in (0,2,3,4):
            field=AngularOracle(scene,independent=independent,method=method,cover_depth=4,cover_nodes=8192,
                                adaptive_nodes=256,angular_nodes=2048,empty_prefix=True,
                                max_nodes=512,query_seconds=.1).field(source,8,10)
            assert not field['valid'][7,4,4],('New acceleration invented a contact decision',independent,method)
            assert not field['complete']
            rows.append({'backend':'independent' if independent else 'primary','method':method,
                         'original_ray_status':d.status,'target_voxel_valid':False,'complete':False,
                         'reason':field['reason'],'unresolved_voxels':int((~field['valid']).sum()),
                         'work':field['work']})
    atomic_json(args.output,{'status':'PASS','checks':rows,'physical_contact_is_HIT':True,
                            'available_general_predicate_remains_UNKNOWN':True,
                            'invalid_bits_are_not_physical_labels':True,'new_READY_scenes':0})
    print(json.dumps({'exact_contact_guards':len(rows),'all_preserved_UNKNOWN':True}))


if __name__=='__main__':main()
