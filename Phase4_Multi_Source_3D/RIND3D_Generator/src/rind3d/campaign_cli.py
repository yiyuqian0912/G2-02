"""Cumulative campaign CLI; generation limits are per execution, not identities."""
import argparse,json
from . import campaign

def main(argv=None):
 p=argparse.ArgumentParser(prog='rind3d campaign');s=p.add_subparsers(dest='action',required=True)
 q=s.add_parser('create');q.add_argument('--root',required=True);q.add_argument('--config',required=True);q.add_argument('--certificate',required=True);q.add_argument('--target-total',required=True,type=int);q.add_argument('--master-seed');q.add_argument('--diagnostic',action='store_true')
 q=s.add_parser('run');q.add_argument('--root',required=True);q.add_argument('--max-new-scenes',default=100,type=int);q.add_argument('--max-runtime',default=3600,type=float);q.add_argument('--verification-workers',default=4,type=int,choices=(1,2,4));q.add_argument('--resume',action='store_true');q.add_argument('--cpu-only',action='store_true');q.add_argument('--diagnostic',action='store_true');q.add_argument('--diagnostic-fault',choices=('after_scene_rename','before_catalog_commit','after_catalog_commit'))
 for name in ('status','pause'):
  q=s.add_parser(name);q.add_argument('--root',required=True)
 q=s.add_parser('extend');q.add_argument('--root',required=True);q.add_argument('--target-total',required=True,type=int)
 q=s.add_parser('verify');q.add_argument('--root',required=True);q.add_argument('--metadata-only',action='store_true');q.add_argument('--physical',action='store_true');q.add_argument('--recover-catalog',action='store_true')
 q=s.add_parser('export');q.add_argument('--root',required=True);q.add_argument('--output',required=True);q.add_argument('--name',required=True);q.add_argument('--count',type=int);q.add_argument('--parent');q.add_argument('--delta',action='store_true');q.add_argument('--diagnostic',action='store_true')
 a=p.parse_args(argv)
 if a.action=='create':
  from .cli import load_config
  result=campaign.create(a.root,load_config(a.config),a.target_total,a.certificate,master_seed=a.master_seed,diagnostic=a.diagnostic)
 elif a.action=='run':result=campaign.run(a.root,max_new_scenes=a.max_new_scenes,max_runtime=a.max_runtime,workers=a.verification_workers,cpu_only=a.cpu_only,diagnostic=a.diagnostic,fault_phase=a.diagnostic_fault)
 elif a.action=='status':result=campaign.status(a.root)
 elif a.action=='pause':result=campaign.pause(a.root)
 elif a.action=='extend':result=campaign.extend(a.root,a.target_total)
 elif a.action=='verify':result=campaign.verify(a.root,full=not a.metadata_only,physical=a.physical,recover_catalog=a.recover_catalog)
 else:
  from .release_export import export_release
  result=export_release(a.root,a.output,a.name,count=a.count,parent=a.parent,delta=a.delta,diagnostic=a.diagnostic)
 print(json.dumps(result,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
