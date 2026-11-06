"""Global Gaussian containment from rigorous outer supports and closed box covers.

This is an experimental numerical proof, not a production topology certificate.
No Gaussian tail is clipped and no sampled face grid is a certificate.
"""
import time
from fractions import Fraction

import numpy as np
from flint import fmpq
import gmpy2 as mp

from ..models import rational,record
from ..queries import FieldOracle


def outward(value,upper):
    v=float(value)
    if not np.isfinite(v):raise ValueError('Finite outer support is required for this proof')
    if (upper and rational(v)<value) or (not upper and rational(v)>value):
        v=np.nextafter(v,np.inf if upper else -np.inf)
    return float(v)


def roots(scene,margin):
    support=FieldOracle(scene).support;m=rational(margin);right=outward(fmpq(1)-m,False)
    result=[]
    for bounds in support:
        lo=[outward(a,False) for a,b in bounds];hi=[outward(b,True) for a,b in bounds]
        for axis in range(3):
            if lo[axis]<=margin:
                a=lo.copy();b=hi.copy();b[axis]=min(b[axis],margin);result.append((a,b))
            if hi[axis]>=right:
                a=lo.copy();b=hi.copy();a[axis]=max(a[axis],right);result.append((a,b))
    unique={tuple(a+b):(a,b) for a,b in result}
    return list(unique.values())


def certify(scene,oracle,margin=.025,*,max_nodes=4096,max_depth=20,seconds=2.):
    if getattr(oracle,'geometry_id',None)!=record(scene)['geometry_id']:
        raise ValueError('Geometry parameters/recipe changed after native engine construction')
    if not 0<margin<.5:raise ValueError('Invalid inset margin')
    t=time.perf_counter();simple=FieldOracle(scene).containment(margin)
    base={'geometry_id':record(scene)['geometry_id'],'margin':margin,'ready_capable':False}
    if simple['status']=='CERTIFIED':
        return base|{'status':'CERTIFIED','kind':'unchanged_Arb_exterior_halfspace_bound','details':simple,'wall_seconds':time.perf_counter()-t}
    if max_nodes<=0:
        return base|{'status':'UNKNOWN','kind':'closed_outer_support_exterior_cover','reason':'containment_budget','wall_seconds':time.perf_counter()-t}
    # A rigorous SOLID point can disprove containment. FREE face samples can
    # never prove it. Proposals target each kernel's analytic plane minimum;
    # rounding is harmless because membership is rechecked at the actual point.
    candidates=[];m=rational(margin);upper_face=outward(fmpq(1)-m,True)
    for axis in range(3):
        for face in (margin,upper_face):
            for kernel in scene.kernels:
                delta=rational(face)-kernel.qcenter[axis];p=[]
                for j in range(3):
                    covariance=sum((rational(kernel.matrix[j,d])*rational(kernel.matrix[axis,d]) for d in range(3)),fmpq(0))
                    p.append(float(kernel.qcenter[j]+covariance*delta/kernel.row_norms[axis]))
                p[axis]=face;candidates.append(p)
    for p,d in zip(candidates,oracle.points(candidates)):
        if d.status=='SOLID':
            return base|{'status':'OUTSIDE_SOLID_PROVED','kind':'certified_exterior_point_witness','witness_xyz':p,
                         'reason':'proved SOLID on or beyond inset face; not a sampled containment certificate','wall_seconds':time.perf_counter()-t}
    initial=roots(scene,margin);nodes=[{'lo':a,'hi':b,'depth':0} for a,b in initial];stack=list(range(len(nodes)));processed=0
    status='CERTIFIED';reason='all_closed_exterior_support_boxes_FREE'
    while stack:
        if processed>=max_nodes or time.perf_counter()-t>seconds:
            status='UNKNOWN';reason='containment_budget';break
        i=stack.pop();node=nodes[i];processed+=1;d=oracle.classify_box(node['lo'],node['hi']);node['decision']=d.status
        if d.status=='FREE':continue
        if d.status=='SOLID':status='OUTSIDE_SOLID_PROVED';reason='closed_exterior_solid_box';break
        if node['depth']>=max_depth:status='UNKNOWN';reason='containment_depth';break
        widths=np.array(node['hi'])-node['lo'];axis=int(np.argmax(widths));middle=float(node['lo'][axis]+widths[axis]/2)
        if not node['lo'][axis]<middle<node['hi'][axis]:status='UNKNOWN';reason='unsplittable_binary64_box';break
        a=node['lo'].copy();b=node['hi'].copy();b[axis]=middle
        c=node['lo'].copy();h=node['hi'].copy();c[axis]=middle
        children=[len(nodes),len(nodes)+1];node.update(axis=axis,middle=middle,children=children)
        nodes.extend([{'lo':a,'hi':b,'depth':node['depth']+1},{'lo':c,'hi':h,'depth':node['depth']+1}]);stack.extend(children)
    result=base|{'status':status,'kind':'closed_outer_support_exterior_cover','reason':reason,'roots':list(range(len(initial))),
                 'nodes':nodes,'processed':processed,'pending':len(stack),'wall_seconds':time.perf_counter()-t,
                 'limits':{'nodes':max_nodes,'depth':max_depth,'seconds':seconds},'positive_tail_support_theorem':'K subset union{g_i >= tau/M}'}
    if status=='OUTSIDE_SOLID_PROVED':result['witness_xyz']=[float(a+(b-a)/2) for a,b in zip(node['lo'],node['hi'])]
    return result


def independently_recheck(scene,proof,oracle):
    if getattr(oracle,'geometry_id',None)!=record(scene)['geometry_id']:
        raise ValueError('Independent engine geometry identity mismatch')
    if proof['geometry_id']!=record(scene)['geometry_id']:raise ValueError('Containment evidence geometry mismatch')
    if proof['status']!='CERTIFIED':return {'status':'INCOMPLETE','reason':'primary containment not certified'}
    if proof['kind']=='unchanged_Arb_exterior_halfspace_bound':
        # Independent Fraction/MPFR construction of the same mathematical
        # halfspace inequality, without calling primary Arb bounds.
        t=scene.threshold
        with mp.context(precision=256,round=mp.RoundDown):
            lower=mp.mpfr(t.value) if t.kind=='binary64' else mp.exp(mp.mpfr(t.value))
        m=Fraction(proof['margin']);checks=[]
        for axis in range(3):
            for side in (0,1):
                with mp.context(precision=256,round=mp.RoundUp):
                    upper=mp.mpfr(0)
                    for k in scene.kernels:
                        distance=Fraction(float(k.center[axis]))-m if side==0 else 1-m-Fraction(float(k.center[axis]))
                        norm=sum((Fraction(float(x))**2 for x in k.matrix[axis]),Fraction(0))
                        qmin=distance**2/norm if distance>0 else Fraction(0)
                        argument=-qmin/2
                        upper+=mp.mpfr(k.weight)*mp.exp(mp.mpfr(mp.mpq(argument.numerator,argument.denominator)))
                checks.append(bool(upper<lower))
        return {'status':'CERTIFIED' if all(checks) else 'UNKNOWN',
                'kind':'independent_Fraction_directed_MPFR_halfspaces','leaf_boxes_checked':0,'face_checks':checks}
    initial=roots(scene,proof['margin']);nodes=proof['nodes'];seen=set();todo=list(proof['roots']);leaves=0;unknown=0
    if todo!=list(range(len(initial))):raise ValueError('Missing or reordered exterior roots')
    for i,(lo,hi) in enumerate(initial):
        if nodes[i]['lo']!=lo or nodes[i]['hi']!=hi:raise ValueError('Exterior root coverage mismatch')
    while todo:
        i=todo.pop()
        if i in seen or not 0<=i<len(nodes):raise ValueError('Overlapping/invalid proof tree')
        seen.add(i);node=nodes[i]
        if 'children' in node:
            axis=node['axis'];middle=node['middle'];left,right=(nodes[j] for j in node['children'])
            if not 0<=axis<3 or not node['lo'][axis]<middle<node['hi'][axis]:raise ValueError('Invalid partition')
            alo=node['lo'].copy();ahi=node['hi'].copy();ahi[axis]=middle
            blo=node['lo'].copy();bhi=node['hi'].copy();blo[axis]=middle
            if (left['lo'],left['hi'],right['lo'],right['hi'])!=(alo,ahi,blo,bhi):raise ValueError('Gap in exterior cover')
            todo.extend(node['children'])
        else:
            if node.get('decision')!='FREE':raise ValueError('Unproved exterior leaf')
            d=oracle.classify_box(node['lo'],node['hi']);leaves+=1
            if d.status=='SOLID':raise AssertionError('Independent containment disagreement')
            if d.status!='FREE':unknown+=1
    return {'status':'CERTIFIED' if not unknown else 'UNKNOWN','kind':'independent_natural_MPFR_closed_leaf_cover',
            'leaf_boxes_checked':leaves,'unresolved_leaves':unknown}
