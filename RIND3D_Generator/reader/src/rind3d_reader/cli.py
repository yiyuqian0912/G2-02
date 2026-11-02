import argparse,json
from .data import RINDDataset
def main(argv=None):
 p=argparse.ArgumentParser(prog='rind3d-reader');s=p.add_subparsers(dest='command',required=True);q=s.add_parser('verify');q.add_argument('root');q.add_argument('--metadata-only',action='store_true');q=s.add_parser('info');q.add_argument('root')
 q=s.add_parser('pack');q.add_argument('root');q.add_argument('archive')
 q=s.add_parser('extract');q.add_argument('archive');q.add_argument('destination');q.add_argument('--sha256')
 q=s.add_parser('apply-delta');q.add_argument('base');q.add_argument('delta');q.add_argument('output')
 a=p.parse_args(argv)
 from .releases import pack_release,extract_release,apply_delta,verify_release
 if a.command in ('pack','extract','apply-delta'):
  if a.command=='pack':r=pack_release(a.root,a.archive)
  elif a.command=='extract':r=extract_release(a.archive,a.destination,expected_sha256=a.sha256)
  else:r=apply_delta(a.base,a.delta,a.output)
  print(json.dumps(r,indent=2));return 0
 if a.command=='verify':
  from pathlib import Path
  m=json.loads((Path(a.root)/'manifest.json').read_text())
  if 'release_format' in m:
   r=verify_release(a.root,full=not a.metadata_only,crc=not a.metadata_only);print(json.dumps(r,indent=2));return 0
 with RINDDataset(a.root,split=None) as d:
  if a.command=='verify':r=d.verify(full=not a.metadata_only,crc=not a.metadata_only)
  else:
   meta=d.metadata();r={'scenes':len(d.scenes),'windows':len(d),**{k:meta[k] for k in ('name','release_id','kind','dataset_profile','resolution','window_sides','size_quotas','split_counts','family_counts','privacy_policy','license','citation','reader_compatibility') if k in meta}}
 print(json.dumps(r,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
