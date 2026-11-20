"""Read-only release check. Pin both the input plan and a separately trusted issuance index."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import inputs  # Install the frozen geometry and ledger module paths used by evidence.
import evidence


def read_pinned(path,expected):
    data=path.read_bytes()
    if evidence.digest(data)!=expected:
        raise ValueError(f"unexpected digest: {path}")
    return json.loads(data)


def verify(root,plan,issued,manifest_sha):
    directories={p.name for p in (root/"tasks").iterdir() if p.is_dir()}
    if directories!=set(plan['tasks']) or set(issued)!=directories:
        raise ValueError("release does not cover the complete scientific population")
    for key,ident in plan['tasks'].items():
        folders=sorted((root/'tasks'/key).glob('*/start.json'))
        if not folders: raise ValueError("no recorded attempts: "+key)
        for n,start in enumerate(folders):
            info=json.loads(start.read_bytes())
            rec=json.loads((start.parent/'receipt.json').read_bytes())
            if info['key']!=key or info['manifest_sha']!=manifest_sha:
                raise ValueError("attempt belongs to a different manifest")
            if n==len(folders)-1:
                if rec['status']!='CERTIFIED' or rec['sha256']!=issued[key]:
                    raise ValueError("missing or conflicting issuance")
                evidence.read_issued(start.parent/'label.json',ident,plan['build'],issued[key])
            elif rec['status'] not in ('INCOMPLETE','INTERRUPTED','RESOURCE_INCOMPLETE'):
                raise ValueError("invalid or duplicate historical issuance")
    return len(directories)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('root',type=Path)
    ap.add_argument('manifest_sha')
    ap.add_argument('index',type=Path)
    ap.add_argument('index_sha')
    args=ap.parse_args()
    count=verify(args.root,read_pinned(args.root/'manifest.json',args.manifest_sha),
                 read_pinned(args.index,args.index_sha),args.manifest_sha)
    print(f'{count} issued artifacts cover the complete pinned population')
