"""Experimental exact-rational / rigorous-Arb predicates, never READY authority.

MISS requires an upper enclosure strictly below the closed solid threshold.
Subdivision or precision exhaustion returns an explicit non-physical status.
Gaussian tails are never deleted. Support boxes use the positive-sum theorem:
sum(f_i)>=tau implies at least one f_i>=tau/N.
"""
from collections import Counter
from dataclasses import dataclass
from itertools import product
import math
import threading
import time

from flint import arb,ctx,fmpq
import numpy as np

from .models import KernelScene,SpectralScene,ConvexScene,dot,rational

ARB_LOCK=threading.RLock()


@dataclass(frozen=True)
class Limits:
    precision: tuple = (96,192,384)
    max_nodes: int = 512
    max_depth: int = 40
    seconds: float = .5

    def __post_init__(self):
        if not self.precision or any(n<53 for n in self.precision) or self.max_nodes<0 or self.max_depth<0 or self.seconds<=0:
            raise ValueError('Invalid proof budget')


@dataclass(frozen=True)
class Decision:
    status: str
    reason: str
    nodes: int = 0
    depth: int = 0

    @property
    def certified(self):return self.status in ('FREE','SOLID','HIT','MISS')


def point(p):
    if len(p)!=3:raise ValueError('Expected XYZ point')
    return tuple(rational(v) for v in p)


def ball(q):return arb(q)


def interval(lo,hi):
    if lo>hi:raise ValueError('Reversed interval')
    return arb(lo).union(arb(hi))


def endpoint(value,upper=True):
    m,e=(value.upper() if upper else value.lower()).man_exp();m,e=int(m),int(e)
    return fmpq(m*(1<<e)) if e>=0 else fmpq(m,1<<(-e))


def squares(lo,hi):
    return (fmpq(0) if lo<=0<=hi else min(lo*lo,hi*hi),max(lo*lo,hi*hi))


def qbox(kernel,lo,hi):
    result=[fmpq(0),fmpq(0)]
    for row in kernel.inv:
        low=high=fmpq(0)
        for a,c,l,h in zip(row,kernel.qcenter,lo,hi):
            x,y=a*(l-c),a*(h-c);low+=min(x,y);high+=max(x,y)
        l,h=squares(low,high);result[0]+=l;result[1]+=h
    return tuple(result)


def qrange(coef,l=fmpq(0),r=fmpq(1)):
    a,b,c=coef
    left=(a*l+b)*l+c;right=(a*r+b)*r+c
    t=max(l,min(r,-b/(2*a))) if a else l
    return (a*t+b)*t+c,max(left,right),t


def ellipsoid_hit(kernel,radius2,a,b):
    return qrange(kernel.coefficients(a,b))[0]<=radius2


def boxes_overlap(lo,hi,bounds):
    return all(a<=d and c<=b for a,b,(c,d) in zip(lo,hi,bounds))


def segment_box(a,b,bounds):
    low,high=fmpq(0),fmpq(1)
    for x,y,(lo,hi) in zip(a,b,bounds):
        d=y-x
        if not d:
            if not lo<=x<=hi:return False
        else:
            l,h=(lo-x)/d,(hi-x)/d;low=max(low,min(l,h));high=min(high,max(l,h))
            if low>high:return False
    return True


class FieldOracle:
    ready_capable=False

    def __init__(self,scene,limits=Limits(),*,use_support=True,use_proxies=True):
        self.scene=scene;self.limits=limits;self.work=Counter();self.proxies=[];self.support=[]
        self.use_support=use_support;self.use_proxies=use_proxies
        with ARB_LOCK,ctx.workprec(256):
            if isinstance(scene,KernelScene):
                n=len(scene.kernels)
                for k in scene.kernels:
                    if scene.kind=='ellipsoid_union':outer=inner=fmpq(1)
                    else:
                        outer=endpoint(2*(arb(n)*arb(rational(k.weight))/self.threshold()).log())
                        inner=endpoint(2*(arb(rational(k.weight))/self.threshold()).log(),False)
                    if outer>=0:
                        bounds=[]
                        for c,norm in zip(k.qcenter,k.row_norms):
                            r=(arb(norm)*arb(outer)).sqrt()
                            bounds.append((endpoint(arb(c)-r,False),endpoint(arb(c)+r)))
                        self.support.append(tuple(bounds))
                    if inner>0:self.proxies.append((k,inner))
            elif isinstance(scene,SpectralScene):
                self.support=[tuple((rational(c)-rational(r),rational(c)+rational(r)) for c,r in zip(scene.center,scene.radius))]
            else:raise TypeError('Unsupported implicit model')

    def threshold(self):
        t=self.scene.threshold
        return arb(rational(t.value)) if t.kind=='binary64' else arb(rational(t.value)).exp()

    def _outside(self,lo,hi):return self.use_support and not any(boxes_overlap(lo,hi,b) for b in self.support)

    def _score_ranges(self,ranges):
        total=arb(0)
        for k,(lo,hi) in zip(self.scene.kernels,ranges):
            total+=arb(rational(k.weight))*(-interval(lo,hi)/2).exp()
            self.work['exp_enclosures']+=1
        return total-self.threshold()

    def _spectral_score(self,lo,hi):
        s=self.scene;taper=arb(1)
        for c,r,l,h in zip(s.center,s.radius,lo,hi):
            c,r=rational(c),rational(r)
            qlo,qhi=squares((l-c)/r,(h-c)/r)
            l=max(fmpq(0),1-qhi);h=max(fmpq(0),1-qlo)
            taper*=interval(l*l,h*h)
        total=arb(rational(s.bias))
        for f,a,b in zip(s.frequencies,s.cosine,s.sine):
            angle=sum((arb(rational(w))*interval(l,h) for w,l,h in zip(f,lo,hi)),arb(0))*2
            total+=arb(rational(a))*angle.cos_pi()+arb(rational(b))*angle.sin_pi()
            self.work['trig_enclosures']+=2
        return taper*total-self.threshold()

    def _score_box(self,lo,hi):
        if isinstance(self.scene,SpectralScene):return self._spectral_score(lo,hi)
        return self._score_ranges([qbox(k,lo,hi) for k in self.scene.kernels])

    def membership(self,p):
        self.work['membership_queries']+=1;p=point(p)
        if self._outside(p,p):self.work['support_free_points']+=1;return Decision('FREE','certified_outer_support')
        if self.scene.kind=='ellipsoid_union':
            hit=any(k.quadratic(p)<=1 for k in self.scene.kernels)
            return Decision('SOLID' if hit else 'FREE','exact_rational_ellipsoid')
        if isinstance(self.scene,KernelScene):
            qs=[k.quadratic(p) for k in self.scene.kernels]
            if len(qs)==1 and self.scene.threshold.kind=='exp' and self.scene.kernels[0].weight==1:
                hit=qs[0]<=-2*rational(self.scene.threshold.value)
                return Decision('SOLID' if hit else 'FREE','analytic_single_kernel_reduction')
            if not any(qs) and self.scene.threshold.kind=='binary64':
                hit=sum(map(lambda k:rational(k.weight),self.scene.kernels),fmpq(0))>=rational(self.scene.threshold.value)
                return Decision('SOLID' if hit else 'FREE','exact_common_center_reduction')
        for index,bits in enumerate(self.limits.precision):
            with ARB_LOCK,ctx.workprec(bits):
                value=self._score_box(p,p)
                if index:self.work['precision_escalations']+=1
                if value>=0:return Decision('SOLID','Arb_enclosure')
                if value<0:return Decision('FREE','Arb_enclosure')
        self.work['unresolved_membership']+=1
        return Decision('UNKNOWN','precision_exhausted')

    def classify_box(self,lo,hi):
        self.work['region_queries']+=1;lo,hi=point(lo),point(hi)
        if any(l>h for l,h in zip(lo,hi)):raise ValueError('Reversed box')
        if self._outside(lo,hi):return Decision('FREE','certified_outer_support')
        if self.scene.kind=='ellipsoid_union':
            ranges=[qbox(k,lo,hi) for k in self.scene.kernels]
            if any(h<=1 for l,h in ranges):return Decision('SOLID','ellipsoid_box_bound')
            if all(l>1 for l,h in ranges):return Decision('FREE','ellipsoid_box_bound')
            return Decision('UNKNOWN','mixed_or_unresolved_box')
        with ARB_LOCK,ctx.workprec(self.limits.precision[0]):
            score=self._score_box(lo,hi)
            if score>=0:return Decision('SOLID','Arb_region_enclosure')
            if score<0:return Decision('FREE','Arb_region_enclosure')
        return Decision('UNKNOWN','mixed_or_unresolved_box')

    def segment(self,start,stop):
        self.work['segment_queries']+=1;a,b=point(start),point(stop)
        if self.use_support and not any(segment_box(a,b,box) for box in self.support):
            self.work['support_miss_rays']+=1;return Decision('MISS','certified_outer_support')
        if self.scene.kind=='ellipsoid_union':
            hit=any(ellipsoid_hit(k,fmpq(1),a,b) for k in self.scene.kernels)
            return Decision('HIT' if hit else 'MISS','exact_rational_quadratic')
        coefficients=[k.coefficients(a,b) for k in self.scene.kernels] if isinstance(self.scene,KernelScene) else None
        if coefficients is not None and len(coefficients)==1:
            minimum=qrange(coefficients[0])[0];k=self.scene.kernels[0];t=self.scene.threshold
            if t.kind=='exp' and k.weight==1:
                return Decision('HIT' if minimum<=-2*rational(t.value) else 'MISS','analytic_single_kernel_reduction')
            for bits in self.limits.precision:
                with ARB_LOCK,ctx.workprec(bits):
                    value=arb(rational(k.weight))*(-arb(minimum)/2).exp()-self.threshold()
                    if value>=0:return Decision('HIT','single_kernel_exact_quadratic_plus_Arb')
                    if value<0:return Decision('MISS','single_kernel_exact_quadratic_plus_Arb')
            return Decision('UNKNOWN','single_kernel_numerical_boundary')
        started=time.perf_counter();stack=[(fmpq(0),fmpq(1),0)];nodes=depth=0
        while stack:
            if nodes>=self.limits.max_nodes or time.perf_counter()-started>self.limits.seconds:
                self.work['budget_exhaustions']+=1
                return Decision('BUDGET_EXHAUSTED','node_or_time_limit',nodes,depth)
            l,r,d=stack.pop();nodes+=1;depth=max(depth,d);self.work['ray_nodes']+=1
            with ARB_LOCK,ctx.workprec(self.limits.precision[0]):
                if coefficients is not None:
                    ranges=[qrange(c,l,r) for c in coefficients]
                    score=self._score_ranges([v[:2] for v in ranges])
                else:
                    left=tuple(x+(y-x)*l for x,y in zip(a,b));right=tuple(x+(y-x)*r for x,y in zip(a,b))
                    score=self._score_box(tuple(min(x,y) for x,y in zip(left,right)),tuple(max(x,y) for x,y in zip(left,right)))
                if score<0:continue
                if score>=0:return Decision('HIT','whole_parameter_interval_solid',nodes,depth)
            candidates=[(l+r)/2]
            if nodes==1 and coefficients is not None:
                ranked=sorted(zip(self.scene.kernels,ranges),key=lambda v:float(v[0].weight)*math.exp(-min(1500.,float(v[1][0]))/2),reverse=True)
                candidates=[v[1][2] for v in ranked[:2]]+candidates
            for t in dict.fromkeys(candidates):
                p=tuple(x+(y-x)*t for x,y in zip(a,b));self.work['positive_witness_attempts']+=1
                if self.membership(p).status=='SOLID':return Decision('HIT','certified_point_witness',nodes,depth)
            if d>=self.limits.max_depth:
                self.work['depth_exhaustions']+=1
                return Decision('BUDGET_EXHAUSTED','depth_limit',nodes,depth)
            middle=(l+r)/2;self.work['subdivisions']+=1
            stack.extend(((middle,r,d+1),(l,middle,d+1)))
        self.work['max_query_depth']=max(self.work['max_query_depth'],depth)
        return Decision('MISS','all_closed_parameter_intervals_certified_free',nodes,depth)

    def source_clearance(self,p,radius=2**-26):
        p=point(p);r=rational(radius)
        if any(not 0<x<1 for x in p):return Decision('UNKNOWN','source_outside_open_world')
        box=self.classify_box(tuple(x-r for x in p),tuple(x+r for x in p))
        return Decision('FREE','strict_clearance_cube') if box.status=='FREE' else Decision('UNKNOWN','clearance_not_proved')

    def certificate(self,source,lo,hi):
        """Receiver box certificates; proxies are contained solids, never truth."""
        self.work['visibility_region_queries']+=1
        mask=self.classify_box(lo,hi)
        if mask.status=='SOLID':return 0,1
        hull_lo=tuple(min(x,y) for x,y in zip(point(source),point(lo)))
        hull_hi=tuple(max(x,y) for x,y in zip(point(source),point(hi)))
        if mask.status=='FREE' and self.classify_box(hull_lo,hull_hi).status=='FREE':return 1,0
        if self.use_proxies:
            corners=list(product(*zip(point(lo),point(hi))))
            for kernel,r2 in self.proxies:
                if all(ellipsoid_hit(kernel,r2,point(source),p) for p in corners):
                    self.work['contained_proxy_shadow_certificates']+=1
                    return 0,0 if mask.status=='FREE' else -1
        return -1,0 if mask.status=='FREE' else -1

    def containment(self,margin=.02):
        """Prove the GLOBAL superlevel set avoids all six exterior halfspaces."""
        m=rational(margin)
        if not 0<m<fmpq(1,2):raise ValueError('Invalid world margin')
        if isinstance(self.scene,SpectralScene):
            bounds=self.support[0]
            ok=all(m<lo and hi<1-m for lo,hi in bounds)
            return {'status':'CERTIFIED' if ok else 'UNKNOWN','proof':'exact_compact_taper_support','margin':margin,'ready_capable':False}
        faces=[]
        with ARB_LOCK,ctx.workprec(256):
            for axis in range(3):
                for side in (0,1):
                    ranges=[]
                    for k in self.scene.kernels:
                        distance=k.qcenter[axis]-m if side==0 else 1-m-k.qcenter[axis]
                        minimum=distance*distance/k.row_norms[axis] if distance>0 else fmpq(0)
                        ranges.append(minimum)
                    if self.scene.kind=='ellipsoid_union':proved=all(q>1 for q in ranges)
                    else:
                        value=sum((arb(rational(k.weight))*(-arb(q)/2).exp() for k,q in zip(self.scene.kernels,ranges)),arb(0))
                        proved=value<self.threshold()
                    faces.append({'axis':axis,'side':side,'upper_below_threshold':bool(proved)})
        return {'status':'CERTIFIED' if all(f['upper_below_threshold'] for f in faces) else 'UNKNOWN',
                'proof':'exact_Cauchy_Schwarz_halfspace_minimum_plus_Arb_positive_sum',
                'faces':faces,'margin':margin,'ready_capable':False}


class ConvexOracle:
    """Existing unchanged canonical decoding and independent native A/B."""
    ready_capable=False

    def __init__(self,scene):
        from rind3d.rendering.context import GeometryContext
        from rind3d.lite_acceptance import verify_geometry_proof
        self.scene=scene;self.context=GeometryContext(scene.bundle());self.work=Counter()
        self.proof=self.context.geometry_proof();verify_geometry_proof(self.context.bundle,self.proof)

    def membership(self,p):
        self.work['membership_queries']+=1
        return Decision('SOLID' if self.context.contains(np.array(p,dtype=float)[None])[0] else 'FREE','existing_native_convex')

    def segment(self,a,b):
        self.work['segment_queries']+=1
        y=self.context.bind(a).response(a,np.array(b,dtype=float)[None])[0]
        return Decision('MISS' if y else 'HIT','existing_native_convex')

    def source_clearance(self,p,radius=2**-26):
        from rind3d.lite_acceptance import verify_clearance
        proof=self.context.clearance(p,radius)
        if proof['status']!='GT':return Decision('UNKNOWN','existing_clearance_not_GT')
        verify_clearance(self.context,p,proof);return Decision('FREE','existing_verified_clearance')

    def containment(self,margin=.02):
        from fractions import Fraction
        m=Fraction(float(margin))
        ok=all(min(Fraction(v) for v in r['lower'])>m and max(Fraction(v) for v in r['upper'])<1-m for r in self.proof['components'])
        return {'status':'CERTIFIED' if ok else 'UNKNOWN','proof':'existing_independently_verified_convex_world_duals','margin':margin,'ready_capable':False}


def make_oracle(scene,**kwargs):
    return ConvexOracle(scene) if isinstance(scene,ConvexScene) else FieldOracle(scene,**kwargs)
