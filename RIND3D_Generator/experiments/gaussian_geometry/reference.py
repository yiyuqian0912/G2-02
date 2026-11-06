"""Independent experimental reference: Fraction cofactors and MPFR endpoints.

Does NOT call primary membership, intersection, quadratic extrema, support
boxes or proxies. Matrix inverses use a separate cofactor formula. Gaussian
ray intervals are natural coordinate hulls, not primary quadratic extrema.
Elementary exp/trig comes from MPFR, independently of primary Arb balls.
This is still research evidence, not the original production B authority.
"""
from collections import deque
from fractions import Fraction as Q
import time

import gmpy2 as mp

from .models import KernelScene,SpectralScene
from .queries import Decision


def q(v):
    if isinstance(v,Q):return v
    if isinstance(v,int):return Q(v)
    return Q(float(v))


def invert_cofactors(matrix):
    a,b,c=map(q,matrix[0]);d,e,f=map(q,matrix[1]);g,h,i=map(q,matrix[2])
    det=a*(e*i-f*h)-b*(d*i-f*g)+c*(d*h-e*g)
    if not det:raise ValueError('Singular transform')
    return tuple(tuple(v/det for v in row) for row in
                 ((e*i-f*h,c*h-b*i,b*f-c*e),(f*g-d*i,a*i-c*g,c*d-a*f),(d*h-e*g,b*g-a*h,a*e-b*d)))


def fraction(v):
    a,b=v.as_integer_ratio();return Q(int(a),int(b))


def mpq(v):return mp.mpq(v.numerator,v.denominator)


def exp_interval(lo,hi,bits):
    # For x<=-4096, 0<exp(x)<2^-4096 since e>2. This positive
    # remainder enclosure avoids enormous exact dyadic denominators; it is
    # not zero-tail truncation. Extremely smaller thresholds may stay UNKNOWN.
    with mp.context(precision=bits,round=mp.RoundDown):
        l=mp.mpfr(0) if lo<=-4096 else mp.exp(mp.mpfr(mpq(lo)))
    with mp.context(precision=bits,round=mp.RoundUp):
        h=mp.mpfr(mp.mpq(1,1<<4096)) if hi<=-4096 else mp.exp(mp.mpfr(mpq(hi)))
    return fraction(l),fraction(h)


def mul(a,b):
    values=[x*y for x in a for y in b];return min(values),max(values)


class Reference:
    def __init__(self,scene,bits=256,max_nodes=2048,seconds=2.):
        self.scene=scene;self.bits=bits;self.max_nodes=max_nodes;self.seconds=seconds
        self.transforms=[]
        if isinstance(scene,KernelScene):
            self.transforms=[(tuple(map(q,k.center)),invert_cofactors(k.matrix),q(k.weight)) for k in scene.kernels]
        with mp.context(precision=bits,round=mp.RoundDown):lo=mp.const_pi()
        with mp.context(precision=bits,round=mp.RoundUp):hi=mp.const_pi()
        self.pi=(fraction(lo),fraction(hi))

    def threshold(self):
        t=self.scene.threshold
        return (q(t.value),)*2 if t.kind=='binary64' else exp_interval(q(t.value),q(t.value),self.bits)

    def local_ranges(self,lo,hi):
        ranges=[]
        for center,matrix,weight in self.transforms:
            local=[]
            for row in matrix:
                l=h=Q(0)
                for a,c,x,y in zip(row,center,lo,hi):
                    v,w=a*(x-c),a*(y-c);l+=min(v,w);h+=max(v,w)
                local.append((l,h))
            low=sum((Q(0) if a<=0<=b else min(a*a,b*b) for a,b in local),Q(0))
            high=sum((max(a*a,b*b) for a,b in local),Q(0))
            ranges.append((low,high,weight))
        return ranges

    def trig(self,phase,cosine):
        lo,hi=mul(phase,(2*self.pi[0],2*self.pi[1]));middle=(lo+hi)/2
        with mp.context(precision=self.bits,round=mp.RoundToNearest):m=mp.mpfr(mpq(middle))
        exact=fraction(m);error=max(abs(lo-exact),abs(hi-exact))
        function=mp.cos if cosine else mp.sin
        with mp.context(precision=self.bits,round=mp.RoundDown):low=function(m)
        with mp.context(precision=self.bits,round=mp.RoundUp):high=function(m)
        return max(Q(-1),fraction(low)-error),min(Q(1),fraction(high)+error)

    def bounds(self,lo,hi):
        tlo,thi=self.threshold()
        if isinstance(self.scene,KernelScene):
            ranges=self.local_ranges(lo,hi)
            if self.scene.kind=='ellipsoid_union':
                if any(b<=1 for a,b,w in ranges):return Q(0),Q(1)
                if all(a>1 for a,b,w in ranges):return Q(-1),Q(-1)
                return Q(-1),Q(1)
            # Sum in directed MPFR rather than expanding every extremely
            # small tail into a huge exact dyadic Fraction. Upward summation
            # accounts for every positive tail; none is silently truncated.
            with mp.context(precision=self.bits,round=mp.RoundDown):
                lower=mp.mpfr(0)
                for low,high,weight in ranges:
                    value=mp.mpfr(0) if high>=8192 else mp.exp(mp.mpfr(mpq(-high/2)))
                    lower+=mp.mpfr(mpq(weight))*value
            with mp.context(precision=self.bits,round=mp.RoundUp):
                upper=mp.mpfr(0)
                for low,high,weight in ranges:
                    value=mp.mpfr(mp.mpq(1,1<<4096)) if low>=8192 else mp.exp(mp.mpfr(mpq(-low/2)))
                    upper+=mp.mpfr(mpq(weight))*value
            return fraction(lower)-thi,fraction(upper)-tlo
        s=self.scene;taper=(Q(1),Q(1))
        for c,r,a,b in zip(s.center,s.radius,lo,hi):
            a,b=(a-q(c))/q(r),(b-q(c))/q(r)
            squared=(Q(0) if a<=0<=b else min(a*a,b*b),max(a*a,b*b))
            positive=(max(Q(0),1-squared[1]),max(Q(0),1-squared[0]))
            taper=mul(taper,(positive[0]**2,positive[1]**2))
        total=[q(s.bias),q(s.bias)]
        for vector,a,b in zip(s.frequencies,s.cosine,s.sine):
            low=high=Q(0)
            for w,l,h in zip(vector,lo,hi):
                x,y=q(w)*l,q(w)*h;low+=min(x,y);high+=max(x,y)
            for coefficient,cosine in ((a,True),(b,False)):
                u,v=mul((q(coefficient),)*2,self.trig((low,high),cosine))
                total[0]+=u;total[1]+=v
        l,h=mul(taper,total)
        return l-thi,h-tlo

    def _ellipsoid_ray(self,a,b,radius2=Q(1)):
        for center,matrix,_ in self.transforms:
            u=[sum((v*(x-c) for v,x,c in zip(row,a,center)),Q(0)) for row in matrix]
            v=[sum((z*(y-x) for z,x,y in zip(row,a,b)),Q(0)) for row in matrix]
            aa=sum(z*z for z in v);bb=2*sum(x*y for x,y in zip(u,v));cc=sum(x*x for x in u)-radius2
            # Independent discriminant/sign path, not the primary clamped min.
            if cc<=0 or aa+bb+cc<=0:return True
            if aa>0 and -2*aa<bb<0 and bb*bb-4*aa*cc>=0:return True
        return False

    def membership(self,p):
        p=tuple(map(q,p))
        reduction=isinstance(self.scene,KernelScene) and len(self.transforms)==1 and self.scene.threshold.kind=='exp' and self.transforms[0][2]==1
        if self.scene.kind=='ellipsoid_union' or reduction:
            r2=-2*q(self.scene.threshold.value) if reduction and self.scene.kind!='ellipsoid_union' else Q(1)
            return Decision('SOLID' if self._ellipsoid_ray(p,p,r2) else 'FREE','independent_rational_cofactor')
        lo,hi=self.bounds(p,p)
        if lo>=0:return Decision('SOLID','MPFR_directed_enclosure')
        if hi<0:return Decision('FREE','MPFR_directed_enclosure')
        return Decision('UNKNOWN','MPFR_precision_unresolved')

    def segment(self,start,stop):
        a,b=tuple(map(q,start)),tuple(map(q,stop))
        reduction=isinstance(self.scene,KernelScene) and len(self.transforms)==1 and self.scene.threshold.kind=='exp' and self.transforms[0][2]==1
        if self.scene.kind=='ellipsoid_union' or reduction:
            r2=-2*q(self.scene.threshold.value) if reduction and self.scene.kind!='ellipsoid_union' else Q(1)
            return Decision('HIT' if self._ellipsoid_ray(a,b,r2) else 'MISS','independent_discriminant')
        pending=deque([(Q(0),Q(1),0)]);nodes=depth=0;began=time.perf_counter()
        while pending:
            if nodes>=self.max_nodes or time.perf_counter()-began>self.seconds:
                return Decision('BUDGET_EXHAUSTED','independent_node_or_time_limit',nodes,depth)
            l,r,d=pending.popleft();nodes+=1;depth=max(depth,d)
            left=tuple(x+(y-x)*l for x,y in zip(a,b));right=tuple(x+(y-x)*r for x,y in zip(a,b))
            low,high=self.bounds(tuple(min(x,y) for x,y in zip(left,right)),tuple(max(x,y) for x,y in zip(left,right)))
            if high<0:continue
            if low>=0:return Decision('HIT','MPFR_interval_solid',nodes,depth)
            middle=(l+r)/2;p=tuple(x+(y-x)*middle for x,y in zip(a,b))
            if self.membership(p).status=='SOLID':return Decision('HIT','MPFR_point_witness',nodes,depth)
            pending.extend(((l,middle,d+1),(middle,r,d+1)))
        return Decision('MISS','MPFR_all_intervals_free',nodes,depth)
