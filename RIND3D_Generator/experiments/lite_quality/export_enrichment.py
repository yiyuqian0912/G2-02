"""Publish immutable private research artifacts outside frozen READY scenes."""
import json,time
from collections import defaultdict
from .audit import BASE,OUT
from rind3d.enrichment_research import save_artifact,read_artifact
from rind3d.pack import atomic_json

def run():
    folder=OUT/'inverse-v2';a=json.loads((folder/'summary.json').read_text());groups=defaultdict(list)
    for r in a['records']:
        choices=[t for t in r['tests'] if len(t['positions'])>1]
        if not choices:continue
        chosen=max(choices,key=lambda t:(len(t['positions']),-t['report']['elapsed_seconds']))
        groups[r['window']['scene_id']].append({'window_id':r['window']['window_id'],**chosen})
    dest=folder/'enrichment-artifacts';dest.mkdir(exist_ok=True);reports=[]
    for sid,records in groups.items():
        path=dest/(sid+'.private.json');scene=BASE/'scenes'/sid
        if not path.exists():save_artifact(path,scene,records)
        t=time.perf_counter();result=read_artifact(path,scene,physical_recheck=True)
        reports.append({'scene_id':sid,'windows':len(result),'additional_references':sum(len(p)-1 for p in result.values()),'independent_recheck_seconds':time.perf_counter()-t,'artifact':str(path)})
    atomic_json(dest/'summary.json',{'complete':True,'artifacts':reports,'frozen_scenes_mutated':False})

if __name__=='__main__':run()
