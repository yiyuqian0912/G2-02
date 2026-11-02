"""Create public, reproducible synthetic inputs, never copy private scene labels."""
import json
from pathlib import Path
from rind3d.lite_generators import generate,FAMILIES
from rind3d.lite_acceptance import sample_source,geometry_acceptance
from rind3d.geometry import save_geometry
from rind3d.rendering.context import GeometryContext

def main():
    root=Path(__file__).resolve().parents[1]
    out=root/'experiments/lite_quality/fixtures'
    if out.exists():raise FileExistsError('Preserve existing fixtures; use a fresh checkout for another fixture version')
    out.mkdir(parents=True);rows=[];seed=20261102
    for slot in (0,4,50):
        family=FAMILIES[slot%len(FAMILIES)];bundle,_=generate(seed,slot,family);context=GeometryContext(bundle)
        geometry_acceptance(context,family,seed,slot);source,_=sample_source(context,seed,slot)
        name=f'public-synthetic-{slot}.npz';save_geometry(out/name,bundle)
        rows.append({'slot':slot,'family':family,'geometry_id':context.geometry_id,'geometry_file':name,'source_hex':[float(v).hex() for v in source],'numerical_policy':{}})
    (out/'manifest.json').write_text(json.dumps({'synthetic_fixture_seed':seed,'not_from_published_or_private_dataset':True,'fixtures':rows},indent=2)+'\n')
    print('Prepared three synthetic geometry/source fixtures; no response fields generated')

if __name__=='__main__':main()
