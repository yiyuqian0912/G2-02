"""Portable record storage; no demonstration dependency."""
import json
import numpy as np

def save(record, path):
    record = dict(record)
    metadata = record.pop('metadata', {})
    np.savez_compressed(path, **record, metadata_json=json.dumps(metadata, allow_nan=False))

