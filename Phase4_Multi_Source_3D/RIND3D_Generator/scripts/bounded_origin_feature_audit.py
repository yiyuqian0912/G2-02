# Legacy four-side diagnostic; archived results/source snapshots are unchanged.
"""Post-acceptance feature audit with a process watchdog on optional clearance.

The original canonical query runs unmodified in a forked, single-threaded child.
Killing a slow optional diagnostic produces UNKNOWN, never a physical rejection.
Composition queries and all previously completed sampling/validators are untouched.
"""
import argparse,json,os,select,signal,time
from pathlib import Path
from collections import Counter
import numpy as np
from rind3d.geometry import Oracle,load_geometry
from rind3d.legacy.config import load_config
from rind3d.legacy.windows import Window
from rind3d.pack import atomic_json
from boundary_experiment_support import plain
from origin_audit_meter import AuditMeter
from origin_distribution_features import (primitive_descriptors,spatial_features,witness_features,
    metadata_geometry_features,measure_composition,measure_center_clearance)


def optional_query(function,seconds):
    """Return payload, terminal status, child CPU, wall; no partial value on timeout."""
    read,write=os.pipe();start=time.perf_counter();pid=os.fork()
    if pid==0:
        os.close(read)
        try:payload={'status':'COMPLETE','value':plain(function())}
        except Exception as exc:payload={'status':'ERROR','error':type(exc).__name__+':'+str(exc)}
        data=json.dumps(payload).encode()
        try:
            while data:
                count=os.write(write,data);data=data[count:]
        finally:os.close(write);os._exit(0)
    os.close(write);chunks=[];timed_out=False
    try:
        while True:
            remaining=seconds-(time.perf_counter()-start)
            if remaining<=0:timed_out=True;break
            ready,_,_=select.select([read],[],[],remaining)
            if not ready:timed_out=True;break
            data=os.read(read,65536)
            if not data:break
            chunks.append(data)
        if timed_out:
            try:os.kill(pid,signal.SIGKILL)
            except ProcessLookupError:pass
        _,_,usage=os.wait4(pid,0)
    finally:os.close(read)
    cpu=usage.ru_utime+usage.ru_stime;wall=time.perf_counter()-start
    if timed_out:return None,'DIAGNOSTIC_TIMEOUT',cpu,wall
    try:payload=json.loads(b''.join(chunks))
    except ValueError:return None,'DIAGNOSTIC_CHILD_FAILURE',cpu,wall
    if payload['status']!='COMPLETE':return None,'DIAGNOSTIC_ERROR:'+payload['error'],cpu,wall
    return payload['value'],'COMPLETE',cpu,wall


class BoundedClearance:
    def __init__(self,seconds):self.seconds=seconds;self.child_cpu=0.;self.timeouts=0
    def measure(self,window,meter,radius):
        before=meter.report()
        payload,status,cpu,wall=optional_query(lambda:(measure_center_clearance(window,meter,radius),meter.report()),self.seconds)
        self.child_cpu+=cpu;meter.timings[meter.stage]['process']+=cpu
        meter.counters[meter.stage]['optional_clearance_jobs']+=1
        if status=='COMPLETE':
            result,after=payload
            for stage,values in after['calls'].items():
                for key,val in values.items():meter.counters[stage][key]+=val-before['calls'].get(stage,{}).get(key,0)
        else:
            self.timeouts+=int(status=='DIAGNOSTIC_TIMEOUT')
            meter.counters[meter.stage]['optional_clearance_unavailable']+=1
            result={'clearance_radius_world':radius,'center_solid':None,'center_near_solid_r32':None,
                    'clearance_reason':status,'clearance_query_valid':False}
        return {**result,'clearance_measurement_status':status,'clearance_watchdog_seconds':self.seconds,
                'clearance_measurement_wall':wall,'clearance_measurement_child_cpu':cpu}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--seed',type=int,required=True);p.add_argument('--scene',type=int,required=True);p.add_argument('--source-rank',type=int,required=True);a=p.parse_args()
    settings=json.loads(Path('configs/experimental-feature-diagnostics.json').read_text());distance=BoundedClearance(settings['optional_clearance_watchdog_seconds'])
    base=Path('outputs/origin-distribution-audit')/f'seed-{a.seed}-scene-{a.scene}';root=base/f'source-{a.source_rank}'
    case=json.loads((root/'case.json').read_text());scene_meta=json.loads((base/'scene.json').read_text())
    if not case['sampling_complete']:raise ValueError('Sampling/validation not complete')
    source=np.asarray(case['source']);cfg=dict(load_config('configs/pilot-extended.yaml'),seed=a.seed)
    start=time.perf_counter();cpu=time.process_time();meter=AuditMeter(Oracle(load_geometry(base/'geometry.npz'),cfg['numerical_policy']))
    descriptors,counts=primitive_descriptors(meter.bundle)
    for method in 'ABC':
        out=root/method;meta=json.loads((out/'method.json').read_text());rows=[]
        windows=[Window(tuple(r['origin']),r['side'],witnesses=tuple(map(tuple,r['witnesses']))) for r in meta['accepted_windows']]
        with (out/'audit-probes.jsonl').open('w') as probes:
            for i,w in enumerate(windows):
                row={'method':method,'seed':a.seed,'scene':a.scene,'families':scene_meta['families'],'source_rank':a.source_rank,'source':plain(source),'window_id':i,
                     **spatial_features(w,source,1024),**witness_features(w,1024),**metadata_geometry_features(w,1024,descriptors,counts)}
                key=[a.seed,a.scene,a.source_rank,w.size,row['pair_id']]
                with meter.timed('diagnostic_composition'):composition,raw=measure_composition(w,source,meter,case['settings'],key)
                row.update(composition)
                with meter.timed('diagnostic_clearance'):row.update(distance.measure(w,meter,case['settings']['clearance_radius_voxels']/1024))
                if raw is not None:probes.write(json.dumps(plain({'window_id':i,**raw}))+'\n');probes.flush()
                rows.append(row);(out/'features.jsonl').write_text(''.join(json.dumps(plain(r))+'\n' for r in rows))
        print(a.seed,a.scene,a.source_rank,method,'feature audit complete',len(windows),flush=True)
    meter.close();case.update(audit_complete=True,audit_wall=time.perf_counter()-start,audit_process=time.process_time()-cpu+distance.child_cpu,
        audit_meter=meter.report(),audit_child_cpu=distance.child_cpu,optional_clearance_timeouts=distance.timeouts,
        optional_clearance_watchdog_seconds=distance.seconds,stored_full_response_available=False)
    atomic_json(root/'case.json',plain(case))


if __name__=='__main__':main()
