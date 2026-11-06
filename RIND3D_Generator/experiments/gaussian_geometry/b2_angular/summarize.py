"""Compact audited measurement records; raw fields stay in ignored outputs."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from ..study import atomic_json


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runs',type=Path,nargs='+',required=True)
    p.add_argument('--analysis',type=Path,nargs='*',default=[]);p.add_argument('--controls',type=Path)
    p.add_argument('--numerics',type=Path);p.add_argument('--contact',type=Path);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    runs={};hashes=defaultdict(set);completed=0
    for root in args.runs:
        summary=root/'summary.json';r=json.loads(summary.read_text());records=r['results']
        runs[root.name]={'summary_SHA256':sha(summary),'identity':r['identity'],'records':records}
        for row in records:
            archive=root/f'{row["case"]}-{row["resolution"]}-{row["method"]}-{row["backend"]}.npz'
            assert sha(archive)==row['archive_SHA256'],'Archive changed'
            if row['complete']:
                assert row['valid_voxels']==row['resolution']**3
                hashes[(row['case'],row['resolution'],tuple(row.get('origin_xyz',[0,0,0])),row.get('world_resolution',row['resolution']))].add(row['physical_bits_SHA256']);completed+=1
    assert all(len(v)==1 for v in hashes.values()),'Completed fields disagree'
    result={'experiment':'RIND3D continuous B2 fixed-source angular/depth certificates','base_commit':'9bbd13dc9b46b25940afcba7c36826bc18b131e3',
            'runs':runs,'completed_records':completed,'unique_complete_case_grid_ROI_hash_groups':len(hashes),
            'all_complete_response_and_solid_bits_identical':True,'new_READY_scenes':0,'complete_1024_fields':0,'B2_ready_capable':False,
            'analysis':{x.name:json.loads(x.read_text()) for x in args.analysis},
            'negative_results_preserved':True,'raw_archives_committed':False}
    for name,path in [('controls',args.controls),('existing_Taylor_settings_ablation',args.numerics),('exact_transcendental_contact_guard',args.contact)]:
        if path:result[name]={'SHA256':sha(path),'data':json.loads(path.read_text())}
    atomic_json(args.output,result);print(json.dumps({'complete_records':completed,'physical_hash_groups':len(hashes),'bytes':args.output.stat().st_size}))


if __name__=='__main__':main()
