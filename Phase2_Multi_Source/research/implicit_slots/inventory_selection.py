"""Select an explicit count from a completed, immutable complexity census."""
import gzip,hashlib,json
from pathlib import Path
import mask_only_data as labels


def select_catalogue(root, old, options, count):
    root=Path(root)
    report=json.loads((root/'inventory_report.json').read_text())
    protocol=json.loads((root/'inventory_protocol.json').read_text())
    if not report['scan_complete']:raise ValueError('finish the inventory before choosing training size')
    if report['signature']!=protocol['signature']:raise ValueError('inventory protocol mismatch')
    if protocol['source_index_sha256']!=labels._digest(old):raise ValueError('inventory uses another source cache')
    for key in ('seed','min_layers','window_sizes','max_views_per_scene',
                'max_added_per_configuration','max_added_per_geometry'):
        if options[key]!=protocol['selection'][key]:raise ValueError('inventory selection differs: '+key)
    if type(count) is not int or not 1<=count<=report['available_with_current_diversity_caps']:
        raise ValueError(f"requested {count}; inventory has {report['available_with_current_diversity_caps']} capped complex windows")
    paths=sorted((root/'inventory_chunks').glob('chunk-*.json.gz'))
    hashes=[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
    if hashes!=report['catalogue_chunks_sha256']:raise ValueError('inventory catalogue checksum changed')
    selected=[]
    for number,path in enumerate(paths):
        if path.name!=f'chunk-{number:06d}.json.gz':raise ValueError('catalogue sequence gap')
        meta=json.loads(gzip.decompress(path.read_bytes()))
        if meta['signature']!=report['signature'] or labels._digest(meta['records'])!=meta['records_sha256']:
            raise ValueError('invalid catalogue metadata')
        for row in meta['records']:
            if row['within_diversity_caps'] and len(selected)<count:selected.append(row)
    if len(selected)!=count:raise ValueError('inventory availability differs from catalogue')
    return selected,report['signature']
