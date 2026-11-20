"""Reconstitute the CPU audit environment offline from content-checked installed wheel files.
Builds an isolated venv and does not reinstall development-mode import hooks or generator binaries.
"""
import base64
import hashlib
from importlib.metadata import distribution
import json
from pathlib import Path
import shutil
import subprocess
import sys
import venv


def main(target):
    target=target.resolve()
    if target.exists():
        raise FileExistsError(target)
    venv.EnvBuilder(with_pip=False,symlinks=True).create(target)
    python=target/'bin/python'
    libs=Path(subprocess.check_output([python,'-c','import sysconfig; print(sysconfig.get_path("purelib"))'],text=True).strip())
    manifest={}
    for name in ('numpy','scipy','llvmlite','numba','zstandard','packaging','zarr','numcodecs',
                 'donfig','google-crc32c','msgspec','typing-extensions','PyYAML','python-flint','z3-solver','rind3d'):
        d=distribution(name)
        record=d.read_text('RECORD')
        versions=dict(version=d.version,record_sha256=hashlib.sha256(record.encode()).hexdigest(),copied=0)
        for f in d.files:
            parts=Path(f).parts
            if '..' in parts or Path(f).is_absolute() or f.suffix=='.pyc' or '__pycache__' in parts:
                continue
            if name=='rind3d' and not parts[0].endswith('.dist-info'):
                continue # pinned generator Python sources are imported explicitly from detached checkout
            if f.suffix=='.pth':
                raise ValueError(f'unexpected executable import hook {f}')
            src=d.locate_file(f)
            if f.hash is None and f.name!='RECORD':
                raise ValueError(f'unchecked dependency {f}')
            if f.hash:
                with src.open('rb') as stream:
                    actual=hashlib.file_digest(stream,f.hash.mode).digest()
                if base64.urlsafe_b64encode(actual).decode().rstrip('=')!=f.hash.value:
                    raise ValueError(f'installed wheel file differs: {name}/{f}')
            dest=libs/f
            dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(src,dest)
            versions['copied']+=1
        manifest[name]=versions
    print(json.dumps(manifest,sort_keys=True,indent=2))

if __name__=='__main__':
    main(Path(sys.argv[1]))
