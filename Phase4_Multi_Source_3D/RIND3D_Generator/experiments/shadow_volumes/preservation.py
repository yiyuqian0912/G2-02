"""Verify unchanged production/scientific code, siblings and prior artifacts."""
import hashlib,json,time
from pathlib import Path
from rind3d import _native
from rind3d.pack import atomic_json
from rind3d.pilot import source_fingerprint
from .benchmark import ROOT


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def check(items):return {'checked':len(items),'changed':[name for name,value in items.items() if not Path(name).is_file() or sha(name)!=value]}


def main():
    t=time.perf_counter();before=json.loads((ROOT/'before.json').read_text());siblings={}
    for root,inventory in before['protected'].items():
        v=check(inventory['sha']);v['other_changed']=[name for name,value in inventory['other'].items() if not Path(name).is_file() or [Path(name).stat().st_size,Path(name).stat().st_mtime_ns]!=value]
        known=set(inventory['sha'])|set(inventory['other']);v['added']=[str(p) for p in Path(root).rglob('*') if p.is_file() and str(p) not in known];siblings[root]=v
    protected=check({n:v for n,v in before['source_files'].items() if n!='README.md'})
    archived=check(before['old_experiments']);native=sha(_native.__file__)==before['native_sha256'];fingerprint=source_fingerprint()==before['source_fingerprint']
    result={'unchanged_preexisting_source':protected,'prior_experiment_files':archived,'siblings':siblings,'main_native_unchanged':native,'production_source_fingerprint_unchanged':fingerprint,'intentional_existing_file_edits':['README.md'],'elapsed_seconds':time.perf_counter()-t}
    result['passed']=not protected['changed'] and not archived['changed'] and native and fingerprint and all(not(v['changed'] or v['other_changed'] or v['added']) for v in siblings.values())
    atomic_json(ROOT/'preservation.json',result);print(json.dumps(result,indent=2))
    if not result['passed']:raise SystemExit(2)


if __name__=='__main__':main()
