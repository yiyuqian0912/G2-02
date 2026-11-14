"""Extra publication checks for the new, explicitly versioned research pilot."""
import json
from pathlib import Path
from .quality_certification import VERSION,validate

def check(path,meta):
    if meta.get('production_revision')!=VERSION:raise ValueError('Unknown Core production revision')
    path=Path(path);fixture=validate(json.loads((path/'quality_certification.json').read_text()),current=False)
    if meta.get('quality_identity')!=fixture['identity']['sha256'] or 'quality_certification.json' not in meta['hashes']:raise ValueError('Core scene/fixture identity mismatch')
    backend=meta['physical_backends'][1]
    if backend not in fixture.get('allowed_independent_backends',[fixture['independent_backend']]):raise ValueError('Uncertified independent verifier build')
    if meta.get('production_mode')!='core' or meta.get('optional_search_budget_seconds')!=0:raise ValueError('Core cannot hide optional search costs')
    if meta.get('candidate_domain')!='world_outside_window':raise ValueError('New Core sources must be outside observations')
    if meta.get('observation_sampling') not in ('natural','stratified','mixed'):raise ValueError('Unknown observation distribution')
    return fixture
