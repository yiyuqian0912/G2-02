"""Freeze one atomically saved checkpoint for all evaluation ranks."""
import hashlib,io,json,os
from pathlib import Path
import torch
import torch.distributed as dist


def sha256_file(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):value.update(block)
    return value.hexdigest()


def create_snapshot(source,root):
    from train_decomposition import write_json
    source=Path(source);raw=source.read_bytes();source_sha=hashlib.sha256(raw).hexdigest()
    saved=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
    directory=Path(root)/'.checkpoint_snapshots'/source_sha;directory.mkdir(parents=True,exist_ok=True)
    model=directory/'model.pt';meta_file=directory/'metadata.json'
    if model.is_file() and meta_file.is_file():
        meta=json.loads(meta_file.read_text())
        if meta['snapshot_sha256']==sha256_file(model):return meta
    state={k:saved[k] for k in ('schema','config','model','epoch','signature')}
    temporary=directory/f'model.{os.getpid()}.tmp.pt';torch.save(state,temporary);temporary.replace(model)
    meta={'source_checkpoint_path':str(source.resolve()),'source_checkpoint_sha256':source_sha,
          'snapshot_path':str(model.resolve()),'snapshot_sha256':sha256_file(model),'epoch':int(saved['epoch']),
          'model_signature':saved['signature'],'optimizer_included':False}
    write_json(meta_file,meta)
    return meta


def load_snapshot(source,root,rank=0,world=1):
    message=[None]
    if rank==0:
        try:message[0]={'snapshot':create_snapshot(source,root)}
        except Exception as error:message[0]={'error':f'{type(error).__name__}: {error}'}
    if world>1:dist.broadcast_object_list(message,src=0)
    if 'error' in message[0]:raise RuntimeError('evaluation snapshot failed: '+message[0]['error'])
    meta=message[0]['snapshot']
    if sha256_file(meta['snapshot_path'])!=meta['snapshot_sha256']:raise ValueError('frozen checkpoint checksum changed')
    state=torch.load(meta['snapshot_path'],map_location='cpu',weights_only=True)
    if state['epoch']!=meta['epoch']:raise ValueError('snapshot epoch mismatch')
    return state,meta
