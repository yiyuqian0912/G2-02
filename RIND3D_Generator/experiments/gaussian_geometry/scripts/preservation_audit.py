"""Read-only preservation and test-evidence audit; never repairs scientific data."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time

from experiments.gaussian_geometry.study import atomic_json


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4<<20),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--before',type=Path,required=True);p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--test-log',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--verify-public-files',action='store_true')
    p.add_argument('--cuda-tests-enabled',action='store_true',help='Record the flag used for the supplied test log; does not execute CUDA')
    args=p.parse_args();began=time.perf_counter()
    before=json.loads(args.before.read_text());old=Path(before['frozen_checkout']);mismatches=[]
    for group in ('tracked','uncommitted_teacher_files'):
        for rel,expected in before[group].items():
            path=old/rel
            if not path.is_file() or digest(path)!=expected:mismatches.append(f'{group}:{rel}')
    manifest=args.dataset/'manifest.json';actual=digest(manifest)
    if actual!=before['dataset_manifest']:mismatches.append('public_dataset_manifest')
    m=json.loads(manifest.read_text());verified=0;size=0
    if args.verify_public_files:
        for rel,evidence in m['files'].items():
            relative=Path(rel)
            if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe immutable manifest path')
            path=args.dataset/relative
            if path.is_symlink() or not path.is_file():mismatches.append(f'public_file_missing:{rel}');continue
            if path.stat().st_size!=evidence['bytes'] or digest(path)!=evidence['sha256']:
                mismatches.append(f'public_file_checksum:{rel}')
            verified+=1;size+=evidence['bytes']
    tracked_diff=subprocess.check_output(['git','diff','--name-only','HEAD'],text=True).splitlines()
    if tracked_diff:mismatches.extend('research_checkout_existing_file_modified:'+p for p in tracked_diff)
    test_text=args.test_log.read_text()
    match=re.search(r'(\d+) passed, (\d+) skipped, (\d+) warnings in ([0-9.]+)s',test_text)
    if not match or ' FAILED ' in test_text:raise ValueError('Successful full test result is absent')
    passed,skipped,warnings,seconds=match.groups()
    unit_log=args.test_log.with_name('unit-final.log');unit_match=re.search(r'(\d+) passed',unit_log.read_text()) if unit_log.exists() else None
    row={'status':'PASS' if not mismatches else 'FAIL','mismatches':mismatches,
         'base_commit':before['base_commit'],'research_branch':subprocess.check_output(['git','branch','--show-current'],text=True).strip(),
         'original_tracked_files_unchanged':len(before['tracked']),
         'original_teacher_WIP_files_unchanged':len(before['uncommitted_teacher_files']),
         'existing_research_checkout_tracked_files_modified':tracked_diff,
         'public_RIND3D_manifest_sha256':actual,'public_scene_count':m['scene_count'],'public_window_count':m['window_count'],
         'public_files_SHA256_verified':verified,'public_files_bytes_verified':size,
         'test_result':{'passed':int(passed),'skipped':int(skipped),'warnings':int(warnings),'seconds':float(seconds),
                        'new_research_tests':int(unit_match[1]) if unit_match else None,
                        'CUDA_regression_enabled':args.cuda_tests_enabled,
                        'skips':'explicit real private/production fixture opt-ins not enabled',
                        'warnings_reason':'existing bounded diagnostic fork after thread initialization'},
         'test_log_sha256':digest(args.test_log),'preservation_before_sha256':digest(args.before),
         'wall_seconds':time.perf_counter()-began,'scientific_production_files_modified':False,
         'new_READY_scenes':0,'new_1024_fields':0,'Drive_release_modified':False}
    atomic_json(args.output,row)
    if mismatches:raise AssertionError('Preservation audit failed; see output')


if __name__=='__main__':main()
