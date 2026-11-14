"""Explicit legacy configuration loading; never validates new publication."""
from pathlib import Path
import yaml
from .window_contract import validate_window_config

def load_config(path):
    path=Path(path)
    if path.parent==Path('configs') and path.name in ('pilot.yaml','pilot-extended.yaml'):path=Path('configs/legacy')/path.name
    cfg=yaml.safe_load(path.read_text())
    if cfg.get('format_version')!=2:raise ValueError('New generation requires format_version: 2; v1 datasets remain readable')
    if cfg.get('max_candidates')!=64 or cfg['production_windows']!=4096 or cfg['resolution']!=1024:raise ValueError('v2 requires max_candidates=64, 4096 windows and resolution=1024')
    if cfg['min_source_separation']!=32/1024:raise ValueError('Source separation must remain 32/1024')
    if cfg.get('include_true_source') is not True:raise ValueError('The true source must be retained')
    if cfg.get('source_precision')!='float64' or cfg.get('response_storage')!='RINDPack-v1':raise ValueError('Explicit float64 source and lossless RINDPack are required')
    if cfg['candidate_domain'] not in ('world_free','world_outside_window'):raise ValueError('Unknown source domain')
    validate_window_config(cfg)
    if cfg['max_window_attempts']<4096 or cfg['global_proposals']<0:raise ValueError('Invalid proposal budgets')
    for key in ('search_batch_size','candidate_pool','verification_seconds','verification_points','window_search_seconds','window_search_points','pilot_source_attempts'):
        if cfg[key]<=0:raise ValueError('Invalid budget: '+key)
    return cfg
