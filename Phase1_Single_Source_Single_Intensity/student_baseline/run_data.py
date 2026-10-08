"""Experiment provenance and record loading, independent of upstream physics."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import torch

from data import collate_records, validate_record_splits
from mock_data import make_mock_records

METADATA_FIELDS = ('dataset_version', 'scene_split_version', 'candidate_search_config',
                   'teacher_cost_config', 'area_weight_convention', 'candidate_count')


def make_mock_splits(config):
    """Cost-only mock targets allow temperature to have its intended effect."""
    splits = {name: make_mock_records(n, config.candidate_count, config.image_size, config.random_seed, name)
              for name, n in [('train', 12), ('validation', 6), ('test', 6)]}
    for records in splits.values():
        for record in records:
            record.pop('teacher_prob')
    return splits


def load_record_file(path, require_metadata=False):
    """Read trusted torch files; experiment files include an upstream metadata envelope."""
    raw = torch.load(Path(path), map_location='cpu', weights_only=False)
    if isinstance(raw, dict) and 'splits' in raw:
        splits, metadata = raw['splits'], raw.get('metadata', {})
    else:
        splits, metadata = raw, {}
    if require_metadata:
        missing = [key for key in METADATA_FIELDS if key not in metadata or metadata[key] in (None, '', {})]
        if missing:
            raise ValueError(f'upstream experiment records require metadata: {missing}')
    return validate_record_splits(splits), metadata


def metadata_for(config):
    values = asdict(config)
    return {key: values[key] for key in METADATA_FIELDS}


def evaluation_fingerprint(records, temperature):
    """Hash the exact ordered normalized observations, candidates, masks and targets.

    Conservative: even reordering an otherwise equivalent evaluation set produces
    a different signature. Raw costs are irrelevant when supplied q is authoritative.
    """
    digest = hashlib.sha256()
    for record in records:
        batch = collate_records([record], temperature)
        digest.update(json.dumps([record['scene_id'], record['view_id']], separators=(',', ':')).encode())
        for key in ('response', 'candidate_xy', 'valid_mask', 'teacher_prob'):
            tensor = batch[key].detach().cpu().contiguous()
            descriptor = json.dumps([key, str(tensor.dtype), list(tensor.shape)]).encode()
            digest.update(len(descriptor).to_bytes(8, 'big'))
            digest.update(descriptor)
            payload = tensor.numpy().tobytes()
            digest.update(len(payload).to_bytes(8, 'big'))
            digest.update(payload)
    return digest.hexdigest()


def split_fingerprints(splits, temperature):
    return {name: evaluation_fingerprint(records, temperature) for name, records in splits.items()}


def comparison_group(evaluation_signature, config):
    # Different teacher/search conventions remain separate even if targets happen
    # to coincide numerically on this particular validation set.
    provenance = {'evaluation': evaluation_signature, 'metadata': metadata_for(config),
                  'teacher_temperature': config.teacher_temperature}
    return hashlib.sha256(json.dumps(provenance, sort_keys=True, allow_nan=False).encode()).hexdigest()
