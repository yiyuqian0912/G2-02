"""Hash historical outputs and original scientific modules; read-only siblings."""
import hashlib,json
from pathlib import Path
from rind3d.pack import atomic_json
from rind3d.lite_certification import scientific_identity
from .audit import OUT

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def run():
    before=json.loads((OUT/'before.json').read_text());changed=[];missing=[]
    for name,digest in before['historical_outputs'].items():
        p=Path(name)
        if not p.exists():missing.append(name)
        elif sha(p)!=digest:changed.append(name)
    siblings=[]
    for name,stat in before['siblings'].items():
        p=Path('..')/name
        if not p.exists():siblings.append({'path':name,'reason':'missing'})
        elif [p.stat().st_size,p.stat().st_mtime_ns]!=stat:siblings.append({'path':name,'reason':'size/mtime differs'})
    sibling_roots={Path(name).parts[0] for name in before['siblings']}
    current_siblings={str(p.relative_to(Path('..'))) for root in sibling_roots for p in (Path('..')/root).rglob('*')
                      if p.is_file() and '.git' not in p.parts}
    new_siblings=sorted(current_siblings-set(before['siblings']))
    old=json.loads(Path('outputs/lite-v01-production/scientific-certification.json').read_text())['scientific_identity']
    source_changes=[name for name,digest in before['source'].items() if not Path(name).exists() or sha(Path(name))!=digest]
    record={'historical_files_checked':len(before['historical_outputs']),'historical_changed':changed,'historical_missing':missing,
            'sibling_files_checked':len(before['siblings']),'sibling_changes':siblings,'sibling_unexpected_new_files':new_siblings,'original_scientific_identity_equal':scientific_identity()==old,
            'intentional_common_wrapper_or_documentation_changes':source_changes,'no_mass_production':True}
    atomic_json(OUT/'preservation-audit.json',record)
    if changed or missing or siblings or new_siblings or scientific_identity()!=old:raise AssertionError('Preservation audit failed')
    print(json.dumps({k:v for k,v in record.items() if k!='intentional_common_wrapper_or_documentation_changes'},indent=2))

if __name__=='__main__':run()
