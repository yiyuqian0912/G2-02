"""Certified field-only acceleration for a top-level convex/cosine union.

The original Oracle and canonical function remain unchanged. Closed spatial
cells carry strict Arb sign certificates; UNKNOWN cells are not voxel geometry.
Grid point signs use Arb-certified separable trig bases and outward native
arithmetic. Unproved rays/points fall back to the existing exact predicates.
"""
from collections import Counter
from fractions import Fraction
from itertools import product
import math,threading,time
import numpy as np
from flint import arb,ctx,fmpq
from rind.contracts import CSGNode,CSGOpcode,CodecID,Leaf,RootKind
from rind.geometry import Membership,SegmentDecision
from rind.transcendental import CosineFieldPredicate,capture_work
from .geometry import NumericalUnresolved,Unsupported
from ._native import IntervalFilter,CertifiedSpace,CertifiedGrid

VERSION='certified-function-field-v3'
DEFAULTS={'space_depth':6,'space_nodes':100000,'bound_precision':64,'ray_proof_splits':128}


def settings(values=None):
    out={**DEFAULTS,**(values or {})}
    if set(out)!=set(DEFAULTS):raise ValueError('Unknown field-acceleration setting')
    for key,value in out.items():
        if isinstance(value,bool) or not isinstance(value,int):raise ValueError('Integer field-acceleration settings required')
    if not 0<=out['space_depth']<=12 or not 1<=out['space_nodes']<=1_000_000 or out['bound_precision'] not in (64,128,256) or not 0<=out['ray_proof_splits']<=256:raise ValueError('Invalid bounded field-acceleration settings')
    return out


def exact_fraction(value):
    mantissa,exponent=value.man_exp();m,e=int(mantissa),int(exponent)
    return Fraction(m<<e) if e>=0 else Fraction(m,1<<-e)


def enclosure(value):
    """Binary64 endpoints checked against Arb's EXACT dyadic endpoints."""
    low,high=exact_fraction(value.lower()),exact_fraction(value.upper())
    a,b=float(low),float(high)
    if not math.isfinite(a) or not math.isfinite(b):raise Unsupported('Nonfinite acceleration enclosure; exact fallback required')
    while Fraction.from_float(a)>low:a=math.nextafter(a,-math.inf)
    while Fraction.from_float(b)<high:b=math.nextafter(b,math.inf)
    return a,b


def qarb(value):return arb(fmpq(value.numerator,value.denominator))


def extract(bundle):
    """Only declared top unions and positive diagonal affine primitives."""
    refs={x.content_hash:x for x in bundle.references};primitives=[];functions=[]
    def walk(key):
        item=refs[key]
        if isinstance(item,CSGNode):
            if item.opcode!=CSGOpcode.UNION:raise Unsupported('Field accelerator only supports top-level unions')
            for key in item.children:walk(key)
        elif isinstance(item,Leaf) and item.codec==CodecID.COSINE_FIELD:functions.append(item)
        elif isinstance(item,Leaf) and item.codec in (CodecID.AFFINE_BOX,CodecID.AFFINE_BALL):
            c=item.field('center').as_array();m=item.field('matrix').as_array()
            if not np.array_equal(m,np.diag(np.diag(m))) or np.any(np.diag(m)<=0):raise Unsupported('Non-diagonal primitive uses unchanged oracle')
            primitives.append((0 if item.codec==CodecID.AFFINE_BOX else 1,c.tolist(),np.diag(m).tolist()))
        else:raise Unsupported('Undeclared field codec uses unchanged oracle')
    g=bundle.geometries[0]
    if g.root_kind==RootKind.REF:walk(g.root_hash)
    if len(functions)!=1:raise Unsupported('This accelerator requires exactly one cosine branch')
    return primitives,functions[0]


class FunctionBounds:
    def __init__(self,predicate,bits):
        self.predicate=predicate;self.bits=bits;self.calls=0
        self.constant=int(predicate.bias>0) if not predicate.terms else -1
        with ctx.workprec(bits):
            self.bias=qarb(predicate.bias)
            self.terms=tuple((qarb(a),tuple(qarb(v) for v in vector),qarb(phase)) for a,vector,phase in predicate.terms)
    def box(self,lo,hi):
        self.calls+=1
        if self.constant>=0:return self.constant
        coordinates=tuple(arb(float(a)).union(arb(float(b))) for a,b in zip(lo,hi))
        score=self.bias
        for amplitude,vector,phase in self.terms:
            angle=phase
            for weight,coordinate in zip(vector,coordinates):angle=angle+weight*coordinate
            score=score+amplitude*angle.cos()
        return 1 if score>0 else 0 if score<0 else -1
    def grid(self,n):
        coefficients=np.array([enclosure(qarb(a)) for a,_,_ in self.predicate.terms],dtype=np.float64).reshape(-1,2)
        basis=np.empty((len(self.terms),3,n,4),dtype=np.float64)
        for term,(_,vector,phase) in enumerate(self.terms):
            for axis,weight in enumerate(vector):
                for index in range(n):
                    angle=weight*arb((index+.5)/n)+(phase if axis==0 else arb(0))
                    sine,cosine=angle.sin_cos();cl,ch=enclosure(cosine);sl,sh=enclosure(sine)
                    basis[term,axis,index]=[max(-1,cl),min(1,ch),max(-1,sl),min(1,sh)]
        return CertifiedGrid(n,coefficients,basis,enclosure(self.bias),self.constant),coefficients.nbytes+basis.nbytes+16
    def tree(self,max_depth,max_nodes):
        labels=[-1];children=[-1];pending=[(0,(0.,)*3,(1.,)*3,0)];histogram=Counter();depths=Counter()
        while pending:
            index,lo,hi,depth=pending.pop();label=self.box(lo,hi);histogram[label]+=1;depths[depth]+=1
            if label>=0:labels[index]=label;continue
            if depth>=max_depth or len(labels)+8>max_nodes:continue
            start=len(labels);labels[index]=2;children[index]=start;labels.extend([-1]*8);children.extend([-1]*8)
            middle=tuple((a+b)/2 for a,b in zip(lo,hi))
            for k in range(7,-1,-1):
                a=tuple(middle[axis] if k&(1<<axis) else lo[axis] for axis in range(3))
                b=tuple(hi[axis] if k&(1<<axis) else middle[axis] for axis in range(3))
                pending.append((start+k,a,b,depth+1))
        return CertifiedSpace(labels,children),{'nodes':len(labels),'array_bytes':len(labels)*5,'box_sign_counts':dict(histogram),
            'depth_counts':dict(depths),'unknown_leaves':labels.count(-1),'solid_leaves':labels.count(1),'free_leaves':labels.count(0)}


class FieldOracle:
    """An adapter, not a new canonical geometry backend or publication capability."""
    def __init__(self,base,resolution,options=None):
        self.base=base;self.options=settings(options);self.resolution=int(resolution);self.work=Counter();self.local=threading.local()
        if self.resolution<1 or self.resolution>1024 or self.resolution&(self.resolution-1):raise ValueError('A dyadic field grid is required')
        primitives,leaf=extract(base.bundle);self.primitive=IntervalFilter(primitives)
        candidates=[p for p in getattr(base.reference,'branches',()) if isinstance(p,CosineFieldPredicate)]
        if len(candidates)!=1:raise Unsupported('Unchanged canonical cosine predicate is unavailable')
        self.predicate=candidates[0];bits=self.options['bound_precision']
        if bits>max(self.predicate.ladder):raise Unsupported('Acceleration precision exceeds declared canonical policy')
        start=time.perf_counter();cpu=time.process_time()
        with base.lock,ctx.workprec(bits):
            self.bounds=FunctionBounds(self.predicate,bits);t=time.perf_counter()
            self.space,tree=self.bounds.tree(self.options['space_depth'],self.options['space_nodes']);tree_seconds=time.perf_counter()-t
            t=time.perf_counter();self.grid,basis_bytes=self.bounds.grid(self.resolution);grid_seconds=time.perf_counter()-t
        self.setup={'version':VERSION,'settings':self.options,'tree':tree,'grid_basis_bytes':basis_bytes,
            'tree_build_wall':tree_seconds,'grid_basis_build_wall':grid_seconds,'wall_seconds':time.perf_counter()-start,
            'process_seconds':time.process_time()-cpu,'canonical_function_unchanged':True,'ready_capable':base.capability.get('ready_capable',True),
            'proof_semantics':'strict Arb signs over closed cells; exact Arb trig bases; outward binary64 arithmetic; UNKNOWN -> original predicate',
            'approximate_geometry_used_as_truth':False}
    def __getattr__(self,name):return getattr(self.base,name)
    @property
    def metrics(self):return {**self.base.metrics,**{f'field_{k}':v for k,v in self.work.items()}}
    def report(self):return {**self.setup,'work':dict(self.work),'bound_calls':self.bounds.calls}
    def _points(self,points):
        p=np.ascontiguousarray(points,dtype=np.float64).reshape(-1,3)
        if not np.isfinite(p).all() or np.any(p<0) or np.any(p>1):raise ValueError('Invalid world coordinates')
        return p
    def _function_contains(self,p):
        out=self.space.contains(p);unknown=np.flatnonzero(out<0);self.work['space_membership_points']+=len(p)-len(unknown)
        if len(unknown):
            out[unknown]=self.grid.contains(np.ascontiguousarray(p[unknown]));self.work['grid_membership_points']+=int(np.sum(out[unknown]>=0))
        return out
    def _fast_contains(self,p):
        cached=getattr(self.local,'membership',None)
        if cached is not None and np.array_equal(p,cached[0]):self.work['membership_batch_cache_hits']+=1;return cached[1].copy()
        a=self.primitive.contains(p);b=self._function_contains(p)
        out=np.where((a==1)|(b==1),1,np.where((a==0)&(b==0),0,-1)).astype(np.int8)
        if len(p)<=4096:self.local.membership=(p.copy(),out.copy())
        return out
    def _ray_proof(self,source,point):
        """Bound the canonical F(s+t*(p-s)) directly in ONE parameter t.

        This avoids both Fraction reconstruction and the dependency loss from
        an axis-aligned world-space hull. A failed optional proof does not
        consume or truncate the unchanged reference subdivision allowance.
        """
        cache=getattr(self.local,'ray_source',None)
        if cache is None or not np.array_equal(cache[0],source):
            s=tuple(arb(float(v)) for v in source);angles=[]
            for amplitude,vector,phase in self.bounds.terms:
                angle=phase
                for w,x in zip(vector,s):angle=angle+w*x
                angles.append(angle)
            cache=(source.copy(),s,tuple(angles));self.local.ray_source=cache
        difference=tuple(arb(float(v))-s for v,s in zip(point,cache[1]));rows=[]
        for (amplitude,vector,_),angle in zip(self.bounds.terms,cache[2]):
            slope=arb(0)
            for w,x in zip(vector,difference):slope=slope+w*x
            rows.append((amplitude,angle,slope))
        def score(interval):
            self.work['ray_proof_score_evaluations']+=1;value=self.bounds.bias
            for amplitude,angle,slope in rows:value=value+amplitude*(angle+slope*interval).cos()
            return value
        pending=[(0.,1.,0)];splits=0
        while pending:
            left,right,depth=pending.pop();self.work['ray_proof_nodes']+=1
            self.work['ray_proof_max_depth']=max(self.work['ray_proof_max_depth'],depth)
            value=score(arb(left).union(arb(right)))
            if value>0:self.work['ray_proof_hits']+=1;return SegmentDecision.HIT
            if value<0:continue
            middle=left+(right-left)/2
            if score(arb(middle))>0:self.work['ray_proof_hits']+=1;return SegmentDecision.HIT
            if splits>=min(self.options['ray_proof_splits'],self.base.reference.max_subdivisions) or not left<middle<right:
                self.work['ray_proof_inconclusive']+=1;return None
            splits+=1;self.work['ray_proof_subdivisions']+=1
            pending.append((middle,right,depth+1));pending.append((left,middle,depth+1))
        self.work['ray_proof_misses']+=1;return SegmentDecision.MISS
    def contains(self,points,*,budget=None):
        p=self._points(points)
        if budget:budget.check(len(p))
        out=self._fast_contains(p);unknown=np.flatnonzero(out<0)
        for i in unknown:
            if budget:budget.check()
            out[i]=int(self.base.contains(p[i],budget=None)[0]);self.work['composite_point_fallbacks']+=1
        return out.astype(bool)
    def response(self,source,points,*,budget=None):
        p=self._points(points);s=self._points(source)
        if len(s)!=1 or np.any(s<=0) or np.any(s>=1) or np.any(p<=0) or np.any(p>=1):return self.base.response(s,p,budget=budget)
        if budget:budget.check(len(p))
        self.base.evaluated_sources.add(s[0].astype('<f8').tobytes())
        a=self.primitive.visibility(s,p);solid=self._fast_contains(p);out=np.full(len(p),-1,dtype=np.int8)
        out[(a==0)|(solid==1)]=0;todo=np.flatnonzero(out<0)
        if len(todo):
            b,nodes=self.space.visibility(s[0].tolist(),np.ascontiguousarray(p[todo]));self.work['space_ray_node_tests']+=int(nodes)
            out[todo]=np.where(b==0,0,np.where((a[todo]==1)&(b==1),1,-1));self.work['space_response_points']+=int(np.sum(out[todo]>=0))
        for i in np.flatnonzero(out<0):
            if budget:budget.check()
            if a[i]==1 and np.all(p[i]>0) and np.all(p[i]<1):
                with self.base.lock,ctx.workprec(self.options['bound_precision']):decision=self._ray_proof(s[0],p[i])
                if decision is None:
                    with self.base.lock,capture_work() as trace:
                        decision=self.predicate.segment_intersect(s[0],p[i],max_subdivisions=self.base.reference.max_subdivisions)
                    self.work['exact_function_rays']+=1;self.work['exact_function_subdivisions']+=trace.subdivisions
                    self.work['exact_function_precision_passes']+=trace.backend_work_units
                if decision==SegmentDecision.UNKNOWN:
                    self.base.metrics['unresolved']+=1;raise NumericalUnresolved('Canonical function did not certify the residual ray')
                out[i]=int(decision==SegmentDecision.MISS)
            else:
                out[i]=int(self.base.response(s,p[i],budget=None)[0]);self.work['composite_ray_fallbacks']+=1
        return out.astype(bool)
    def certificate(self,source,lo,hi,*,budget=None):
        if budget:budget.check()
        source=np.asarray(source,dtype=np.float64);lo=np.asarray(lo,dtype=np.float64);hi=np.asarray(hi,dtype=np.float64)
        if any(a.shape!=(3,) for a in (source,lo,hi)) or not np.isfinite(np.r_[source,lo,hi]).all() or np.any(lo<=0) or np.any(hi>=1) or np.any(lo>hi) or np.any(source<=0) or np.any(source>=1):
            return self.base.certificate(source,lo,hi,budget=None)
        self.base.evaluated_sources.add(source.astype('<f8').tobytes());self.work['certificate_calls']+=1
        yp,mp=self.primitive.certificate(source.tolist(),lo.tolist(),hi.tolist())
        with self.base.lock,ctx.workprec(self.options['bound_precision']):mf=self.bounds.box(lo,hi)
        m=1 if mp==1 or mf==1 else 0 if mp==0 and mf==0 else -1
        if m==1 or yp==0:return 0,m
        with self.base.lock,ctx.workprec(self.options['bound_precision']):hull_free=self.bounds.box(np.minimum(source,lo),np.maximum(source,hi))==0
        if hull_free:return (1 if yp==1 else -1),m
        yf,nodes=self.space.certificate(source.tolist(),lo.tolist(),hi.tolist());self.work['space_certificate_node_tests']+=int(nodes)
        return (0 if yf==0 else 1 if yf==1 and yp==1 else -1),m


def field_evaluator(base,resolution,options=None):
    """Unsupported expressions keep the exact original evaluator unchanged."""
    try:return FieldOracle(base,resolution,options)
    except Unsupported:return base
