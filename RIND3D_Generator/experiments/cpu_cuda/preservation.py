"""Final immutable-history/sibling audit and purpose-scoped source snapshot."""
import hashlib
import json
from pathlib import Path
import shutil
from rind3d.pack import atomic_json

ROOT=Path('outputs/cpu-cuda-efficiency')


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    before=json.loads((ROOT/'before.json').read_text());changed=[];history=[];sibling=[]
    for p,digest in before['source_sha256'].items():
        if not Path(p).is_file() or sha(p)!=digest:changed.append(p)
    assert changed==['README.md'] or sorted(changed)==['README.md'],changed
    for p,digest in before['prior_artifact_sha256'].items():
        if not Path(p).is_file() or sha(p)!=digest:history.append(p)
    for p,old in before['sibling_stat'].items():
        f=Path('..')/p
        if not f.is_file():sibling.append(p);continue
        s=f.stat()
        if s.st_size!=old['size'] or s.st_mtime_ns!=old['mtime_ns']:sibling.append(p)
    assert not history,history[:20]
    assert not sibling,sibling[:20]
    protected=json.loads(Path('outputs/shadow-volume-experiment/before.json').read_text())['protected']
    scientific={};fail=[]
    for group in protected.values():
        for p,digest in group['sha'].items():
            f=Path(p)
            if any(part in f.parts for part in ('.venv','build','outputs','.git','__pycache__','.pytest_cache')):continue
            if f.suffix not in ('.py','.cpp','.hpp','.h','.cc','.c','.toml','.md','.cmake') and f.name!='CMakeLists.txt':continue
            actual=sha(f) if f.is_file() else None
            scientific[p]=actual
            if actual!=digest:fail.append(p)
    assert not fail,fail[:20]
    paths=[]
    for prefix in ('src','native','canonical','configs','docs','tests','scripts','experiments'):
        for f in Path(prefix).rglob('*'):
            if f.is_file() and '__pycache__' not in f.parts:paths.append(f)
    paths += [Path(p) for p in ('README.md','pyproject.toml','uv.lock','CMakeLists.txt')]
    fingerprints={}
    target=ROOT/'source-snapshot'
    for f in paths:
        out=target/f;out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,out);fingerprints[str(f)]=sha(f)
    report={'original_source_changes':changed,'new_experiment_sources':{p:h for p,h in fingerprints.items() if p.startswith('experiments/cpu_cuda') or p=='tests/test_cpu_cuda_experiment.py'},
            'original_main_geometry_filter_stream_pack_sampler_loader_unchanged':True,
            'prior_artifacts_checked':len(before['prior_artifact_sha256']),'prior_artifact_changes':history,
            'sibling_file_stats_checked':len(before['sibling_stat']),'sibling_stat_changes':sibling,
            'sibling_scientific_sha_checked':len(scientific),'sibling_scientific_sha_changes':fail,
            'sibling_scientific_sha256':scientific,'source_sha256':fingerprints,
            'all_checks_passed':True,'production_ready':False}
    atomic_json(ROOT/'preservation.json',report)
    print({k:report[k] for k in ('original_source_changes','prior_artifacts_checked','sibling_file_stats_checked','sibling_scientific_sha_checked','all_checks_passed')})


if __name__=='__main__':main()
