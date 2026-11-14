"""Additional fail-closed certification for isolated Lite quality/Core pilot."""
from pathlib import Path
import hashlib,json
from .projection_research import implementation_identity

VERSION='lite-quality-core-research-v1'

def identity():
    from . import _research
    pkg=Path(__file__).parent;root=pkg.parents[1]
    names=('counts.py','projection_research.py','stratified_windows.py','fast_geometry.py','core_research.py','core_checks.py','quality_certification.py')
    paths=[pkg/n for n in names]+[root/'native/research.cpp',root/'native/research_projection.hpp',root/'native/research_nonlocal.hpp',Path(_research.__file__)]
    hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    return {'files':hashes,'sha256':hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()}

def validate(record,*,current=True):
    if record.get('version')!=VERSION or not record.get('complete') or not record.get('small_fields_exact') or not record.get('boundary_tests_passed') or not record.get('packed_count_tests_passed'):raise ValueError('Quality/Core fixture gate closed')
    if len(record.get('complete_1024_comparisons',[]))<3 or any(r.get('bricks')!=4096 or not r.get('response_and_solid_exact') for r in record['complete_1024_comparisons']):raise ValueError('Full-field independent comparison gate closed')
    if not record.get('scientific_gates_unchanged') or record.get('source_uniqueness_claimed'):raise ValueError('Unsupported scientific changes')
    if current and (record['identity']!=identity() or record['independent_backend']!=implementation_identity()):raise ValueError('Stale Quality/Core fixture build')
    return record
