"""Same exact A predicate at different traversal leaves; B decisions remain separate."""
import numpy as np
import pytest
from tests.test_lite_production import box
from rind3d.lite_generators import FAMILIES,generate
from rind3d.lite_acceptance import sample_source
from rind3d.rendering.context import GeometryContext
from rind3d.lite_certification import small_field
from rind3d.campaign_verify import ParametricSource
from rind3d.projection_research import ProjectionSource

def field(model,n):
 y=np.empty((n,)*3,dtype=bool);m=np.empty_like(y)
 for z in range(0,n,64):
  for v in range(0,n,64):
   for x in range(0,n,64):
    side=min(n,64)
    for raw,out in zip(model.packed_batch([(x,v,z)],side,n)[0],(y,m)):
     a=np.unpackbits(np.frombuffer(raw,dtype=np.uint8),bitorder='little').reshape((64,)*3);out[z:z+side,v:v+side,x:x+side]=a[:side,:side,:side]
 return y,m

@pytest.mark.parametrize('family',FAMILIES)
def test_all_families_complete_32_64_128_match_unchanged_a_and_independent_b(family):
 i=FAMILIES.index(family);ctx=GeometryContext(generate(1209,i,family)[0]);p,_=sample_source(ctx,1209,i)
 for n in (32,64,128):
  reference=small_field(ctx.bind(p),n);independent=small_field(ProjectionSource(ctx,p),n)
  assert all(np.array_equal(a,b) for a,b in zip(reference,independent))
  for leaf in (2,4):assert all(np.array_equal(a,b) for a,b in zip(reference,field(ParametricSource(ctx,p,leaf=leaf),n)))

@pytest.mark.parametrize('half',[(.125,.125,.125),(.2,.2,2**-40)])
def test_unaligned_and_thin_contacts_preserve_a_b_and_canonical(half):
 ctx=GeometryContext(box(half=half));p=np.array([.0625,.5,.5]);base=ctx.bind(p);b=ProjectionSource(ctx,p)
 for origin,side in (((3,7,13),51),((37,41,19),17),((64,64,64),64)):
  expected=base.render(origin,side,128)[:2];other=b.render(origin,side,128)[:2]
  assert all(np.array_equal(a,c) for a,c in zip(expected,other))
  for leaf in (2,4):
   y,m,_=base.native.raster(list(origin),side,128,leaf,False,False)
   assert np.array_equal(y,expected[0]) and np.array_equal(m,expected[1])
 # Canonical closed-segment exact oracle retains endpoint/tangency decisions.
 q=np.array([[.375,.5,.5],[.6875,.75,.5],[.6875,.75,.75],[.6875,np.nextafter(.75,1),.5]])
 canonical=ctx.canonical._values(ctx.canonical.reference.segment_intersect(np.repeat(p[None],len(q),axis=0),q))==0
 assert np.array_equal(base.response(p,q),canonical)
