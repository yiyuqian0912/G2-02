# Legacy four-side diagnostic; archived results/source snapshots are unchanged.
"""Read-only timing/call accounting and unchanged downstream pilot validators."""
from __future__ import annotations
from collections import Counter, defaultdict
from contextlib import contextmanager
import time
from pathlib import Path
import numpy as np
from rind3d.geometry import Budget, NumericalUnresolved
from rind3d.legacy.search import CandidateBank, search_candidates
from rind3d.legacy.validation import write_candidate_files,validate_records
from rind3d.legacy.window_contract import PILOT_SIDE_COUNTS, require_side_counts
from rind3d.legacy.windows import excludes, placement_summary
from rind3d.pack import atomic_json


def plain(x):
    if isinstance(x, bytes): return x.hex()
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, np.generic): return x.item()
    if isinstance(x, dict): return {str(k): plain(v) for k,v in x.items()}
    if isinstance(x, (tuple,list)): return [plain(v) for v in x]
    return x


class CountReference:
    def __init__(self, original, meter): self.original,self.meter=original,meter
    def __getattr__(self,name):
        method=getattr(self.original,name)
        if name not in ('classify_points','segment_intersect','classify_box'): return method
        def call(*a,**kw):
            part=self.meter.counters[self.meter.stage]
            part[name]+=1
            part['oracle_calls']+=1
            result=method(*a,**kw)
            part['unresolved_results']+=int(np.count_nonzero(~result.valid))
            return result
        return call


class Meter:
    """Transparent instance proxy. No backend method or predicate is replaced."""
    def __init__(self,oracle):
        self.oracle=oracle;self.original_reference=oracle.reference
        self.counters=defaultdict(Counter);self.timings=defaultdict(lambda:Counter(wall=0.,process=0.))
        self.stage='outside'
        oracle.reference=CountReference(self.original_reference,self)
    @property
    def reference(self): return self.oracle.reference
    @reference.setter
    def reference(self,value): self.oracle.reference=value
    def __getattr__(self,name):
        method=getattr(self.oracle,name)
        if name not in ('certificate','contains','response'): return method
        def call(*a,**kw):
            self.counters[self.stage]['api_'+name]+=1
            return method(*a,**kw)
        return call
    def close(self): self.oracle.reference=self.original_reference
    @contextmanager
    def timed(self,stage):
        old=self.stage;self.stage=stage;t=time.perf_counter();c=time.process_time()
        try: yield
        finally:
            self.timings[stage]['wall']+=time.perf_counter()-t
            self.timings[stage]['process']+=time.process_time()-c
            self.stage=old
    def report(self): return plain({'timings':dict(self.timings),'calls':dict(self.counters)})


def independently_check_window(window,source,cfg,oracle):
    """Fresh oracle, no witness cache; sufficient exact informativeness proof.

    Geometry may intersect W. Both witnesses must be FREE, opposite, sampled
    voxel centres, and in the half-open window. No random rediscovery gate.
    """
    origin=np.asarray(window.origin);side=window.size;resolution=cfg['resolution']
    if np.any(origin<0) or np.any(origin+side>resolution): raise ValueError('Window bounds')
    if not excludes(window,source,resolution): raise ValueError('True source inside window')
    coords=np.asarray(window.witnesses)
    if coords.shape!=(2,3) or not np.issubdtype(coords.dtype,np.integer): raise ValueError('Two exact voxel witnesses required')
    if not np.all((coords>=origin)&(coords<origin+side)): raise ValueError('Witness outside half-open window')
    points=(coords+.5)/resolution
    budget=Budget(cfg['verification_seconds'],cfg['verification_points'])
    oracle.validate_source(source,budget)
    mask=oracle.contains(points,budget=budget)
    response=oracle.response(source,points,budget=budget)
    if np.any(mask) or response.tolist()!=[True,False]: raise NumericalUnresolved('Independent free lit/shadow witness disagreement')
    return {'status':'PASS','source_excluded':True,'free':[True,True],'response':response.tolist(),
            'proof':'two independently certified opposite free sampled centres'}


def final_validators(windows,source,pool,cfg,scene,meter,independent,output):
    """Run the existing candidate and record validators, never a READY bypass."""
    output=Path(output);output.mkdir(exist_ok=True)
    checks=[];bank=CandidateBank();ids=[];reports=[]
    for wid,w in enumerate(windows):
        with independent.timed(f'final_validation:{w.size}'):
            info=independently_check_window(w,source,cfg,independent)
        with meter.timed(f'candidate_validation:{w.size}'):
            sources,evidence,report=search_candidates(meter,source,w,pool,cfg,scene,wid)
        if report['status']!='COMPLETE' or not report['all_grid_exact'] or not report['independent_validation_passed']:
            raise NumericalUnresolved('Unchanged downstream validator failed: '+str(report))
        if not 1<=len(sources)<=64 or not np.array_equal(sources[0],source): raise ValueError('Candidate contract')
        ids.append([bank.add(p) for p in sources]);reports.append(report)
        checks.append({'window_id':wid,'origin':list(w.origin),'side':w.size,'witnesses':plain(w.witnesses),
                       'information':info,'sources':plain(sources),'evidence':evidence,'candidate_report':report})
        atomic_json(output/'final-checks.json',plain(checks))
    with meter.timed('record_validation'):
        dtype,_=write_candidate_files(output,windows,bank,ids,reports)
        meta={'resolution':cfg['resolution'],'candidate_id_dtype':dtype,'candidate_storage':'pool-offsets-v2',
              'contract_version':2,'max_candidates':64,'true_source_included':True,
              'min_source_separation':cfg['min_source_separation'],'status':'INCOMPLETE'}
        validate_records(output,meta,strict=False)
        complete=True
        try: require_side_counts((w.size for w in windows),PILOT_SIDE_COUNTS)
        except ValueError: complete=False
        spatial=placement_summary(windows,cfg['resolution'])
        complete=complete and spatial['occupied_octants']>=2
        # The production validator must reject these 32-window diagnostics.
        try: validate_records(output,meta,strict=True)
        except ValueError as exc: ready_rejection=str(exc)
        else: raise AssertionError('Pilot must never pass the production READY gate')
    result={'pilot_quota_complete':complete,'window_count':len(windows),'placement':spatial,
            'all_retained_candidates_valid':True,'candidate_counts':dict(Counter(str(len(r)) for r in ids)),
            'production_ready':False,'production_validator_rejection':ready_rejection,
            'geometry_ready_capable':meter.capability.get('ready_capable',True),
            'publication_restriction_preserved':not meter.capability.get('ready_capable',True)}
    atomic_json(output/'validation.json',plain(result))
    return result
