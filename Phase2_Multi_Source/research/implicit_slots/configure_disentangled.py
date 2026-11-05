"""Create a reproducible run config; no extra data quota is guessed."""
import argparse,copy,json
from pathlib import Path
from train_decomposition import resolve,load_config
from search_cache import immutable_json

PHASE2=Path(__file__).resolve().parents[2]


def configure(additional=0):
    if type(additional) is not int or additional<0:raise ValueError('additional complex count must be nonnegative')
    config=load_config(PHASE2/'configs/masks_disentangled.json')
    if not additional:return PHASE2/'configs/masks_disentangled.json'
    inv=load_config(PHASE2/'configs/masks_count_first.json')
    options=copy.deepcopy(inv['training_expansion']);options.update(mode='append',additional_complex_windows=additional,
        inventory_root=inv['output_root'])
    from inventory_selection import select_catalogue
    old=json.loads((resolve(config['cache_source_root'])/'mask_records.json').read_text())
    select_catalogue(resolve(inv['output_root']),old,options,additional)
    config['training_expansion']=options
    config['output_root']=f'outputs/masks_disentangled_v5_complex{additional}'
    config['name']=f'phase2_disentangled_v5_complex{additional}'
    config['label_preparation']['record_counts']={**old['counts'],'train':old['counts']['train']+additional}
    destination=PHASE2/f'outputs/launch_configs/disentangled_v5_complex{additional}.json'
    immutable_json(destination,config)
    return destination


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--add-complex',type=int,default=0)
    a=p.parse_args();print(configure(a.add_complex))


if __name__=='__main__':main()
