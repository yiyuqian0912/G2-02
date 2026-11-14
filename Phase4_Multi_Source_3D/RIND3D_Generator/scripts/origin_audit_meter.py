"""Transparent experiment instrumentation, including native-filter evaluations.

Unlike reference-only counts, these counters also count actual evaluations
resolved by the unchanged interval filter. No predicate is replaced or skipped.
"""
from collections import Counter,defaultdict
from contextlib import contextmanager
import time
import numpy as np
from boundary_experiment_support import plain


class ReferenceCounter:
    def __init__(self,reference,meter):self.reference,self.meter=reference,meter
    def __getattr__(self,name):
        method=getattr(self.reference,name)
        if name not in ('classify_points','segment_intersect','classify_box','clearance_decide'):return method
        def call(*a,**kw):
            counters=self.meter.counters[self.meter.stage]
            counters['canonical_'+name]+=1;counters['canonical_queries']+=1
            if name=='clearance_decide':counters['clearance_calls']+=1;counters['oracle_calls']+=1
            result=method(*a,**kw)
            counters['unresolved_'+name]+=int(np.count_nonzero(~result.valid))
            return result
        return call


class AuditMeter:
    def __init__(self,oracle):
        self.oracle=oracle;self.original_reference=oracle.reference;self.stage='outside'
        self.counters=defaultdict(Counter);self.timings=defaultdict(lambda:Counter(wall=0.,process=0.))
        oracle.reference=ReferenceCounter(self.original_reference,self)
    @property
    def reference(self):return self.oracle.reference
    @reference.setter
    def reference(self,value):self.oracle.reference=value
    def __getattr__(self,name):
        method=getattr(self.oracle,name)
        if name not in ('contains','response','certificate','validate_source'):return method
        def call(*a,**kw):
            counters=self.counters[self.stage];counters['api_'+name]+=1
            before_intervals=self.oracle.metrics['interval_points']
            key='canonical_segment_intersect' if name=='response' else 'canonical_classify_points'
            before_reference=counters[key]
            try:return method(*a,**kw)
            finally:
                if name=='certificate':
                    counters['certificate_calls']+=1;counters['oracle_calls']+=1
                else:
                    actual=self.oracle.metrics['interval_points']-before_intervals+counters[key]-before_reference
                    field='response_rays' if name=='response' else 'membership_evaluations'
                    counters[field]+=actual;counters['oracle_calls']+=actual
        return call
    @contextmanager
    def timed(self,stage):
        previous=self.stage;self.stage=stage;t=time.perf_counter();c=time.process_time()
        try:yield
        finally:
            self.timings[stage]['wall']+=time.perf_counter()-t
            self.timings[stage]['process']+=time.process_time()-c
            self.stage=previous
    def close(self):self.oracle.reference=self.original_reference
    def report(self):return plain({'timings':dict(self.timings),'calls':dict(self.counters)})
