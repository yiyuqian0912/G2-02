from fractions import Fraction as Q
from itertools import product
from random import Random
import numpy as np

from envelopes import Envelope, clearance_partition, convex_generators
from poly import Poly, form, grid_pieces, within, without
from solver import Budget, Observation, apply_pair, cone
from research import mass_summary, run
from query import cell_mass
from certificates import segment_hits
from intersection import joint_fraction


def box(lo,hi,den=1):
    rows=[]
    for k in range(3):
        n=tuple(den*int(i==k) for i in range(3))
        rows.extend([(*n,-hi[k]),(*(-x for x in n),lo[k])])
    return tuple(rows)


def represented(pieces,s):
    # IN THE INTERIOR, using a separate volume-based vertex test on all faces.
    def orient(a,b,c,x):
        def point(p): return tuple(Q(y,p[3]) for y in p[:3])
        t=point(a)
        (u,v,w),(i,j,k),(l,m,n)=[tuple(y-z for y,z in zip(point(p),t)) for p in (b,c,x)]
        return (v*k-w*j)*l+(w*i-u*k)*m+(u*j-v*i)*n
    for p in pieces:
        if all(not (any(orient(*face[:3],v)*orient(*face[:3],s)<0 for v in p.vertices))
               for face in p.faces):
            return True
    return False


def test_poly_volume_vs_independent_enumeration_and_cells():
    rng=Random(998)
    for _ in range(35):
        rows=[tuple(rng.randint(-6,6) for _ in range(4)) for _ in range(4)]
        p=within(Poly.cube(),rows)
        v=p.volume if p is not None else 0
        assert v==joint_fraction(((0,0,0),(1,1,1)),1,[(r,False) for r in rows])
        rest=without(Poly.cube(),rows)
        assert sum(q.volume for q in rest)+v==1
        if p is not None:
            assert sum(q.volume for _,q in grid_pieces(p,4))==v


def test_cones_closed_and_class_envelopes_independently():
    body=box((6,7,8),(12,14,17),24)
    points=[(20,y,z,24) for y in (3,12,21) for z in (5,18)]
    env=Envelope(body,points)
    for xyz in product(range(0,25,4),repeat=3):
        p=(*xyz,24)
        hits=[segment_hits(body,xyz,r[:3],24) for r in points]
        assert [all(form(h,p)<=0 for h in cone(body,r)) for r in points]==hits
        assert all(form(h,p)<=0 for h in env.inner)==all(hits)
        if any(hits): assert all(form(h,p)<=0 for h in env.outer)


def test_active_hull_vertices_do_not_prove_visibility():
    body=box((4,-1,-1),(6,1,1))
    s=(0,0,0)
    assert not segment_hits(body,s,(10,4,0),1)
    assert not segment_hits(body,s,(10,-4,0),1)
    assert segment_hits(body,s,(10,0,0),1)


def test_generator_deletions_preserve_exact_support():
    rng=np.random.default_rng(71)
    mask=rng.uniform(size=(16,)*3)<.42
    generators=convex_generators(mask)
    original=np.argwhere(mask)
    # Every removed point lies in the convex hull; check all supporting planes
    # proposed from the reduced set independently at exact integer coordinates.
    from scipy.spatial import ConvexHull
    h=ConvexHull(generators)
    for a,b,c in generators[h.simplices]:
        n=np.cross(b-a,c-a)
        z=(generators-a)@n
        direction=1 if z.max()==0 else -1
        assert (direction*((original-a)@n)).max()==0


def test_boolean_pair_multiple_blockers_and_tangencies():
    bodies=[box((3,5,4),(6,11,9),14),box((8,7,3),(10,10,8),14)]
    d=(11,13,12,14)
    a=(12,13,12,14)
    cones=lambda r:[cone(b,r) for b in bodies]
    parts=apply_pair(Poly.cube(),cones(d),cones(a),Budget(10,200))
    assert 0<sum(p.volume for p in parts)<1
    rng=Random(17)
    for _ in range(40):
        s=tuple(rng.randint(1,307) for _ in range(3))+(311,)
        # Evaluations with a common denominator for an independent clip in t.
        xyz=tuple(q*14 for q in s[:3])
        hits=lambda r:any(segment_hits(b,xyz,tuple(x*311 for x in r[:3]),14*311) for b in bodies)
        assert represented(parts,s)==(hits(d) and not hits(a))


def test_clearance_sandwich_exact_slab_and_corner_gap():
    b=box((2,0,0),(3,8,8),8)
    obs=Observation([b],np.zeros((2,)*3,np.uint8),(0,0,0),8)
    legal,shell=clearance_partition(obs,Poly.cube(),Budget(10,128))
    assert not shell
    assert sum(p.volume for p in legal)==1-Q(1,8)-2*Q(1,2**26)
    cube=box((2,2,2),(4,4,4),8)
    obs=Observation([cube],np.zeros((2,)*3,np.uint8),(0,0,0),8)
    _,shell=clearance_partition(obs,Poly.cube(),Budget(10,128))
    assert sum(p.volume for p in shell)>0


def test_normalization_and_lazy_query_intervals():
    p=within(Poly.cube(),[(2,0,0,-1)])
    r=mass_summary([p],[],1)
    assert r['certified_tv_005'] and r['unknown_upper']==0
    assert cell_mass(r,0)==Q(2,128**3)
    assert cell_mass(r,127*128**2)==0


def test_complete_blind_slab_observation_and_grid_law():
    # Thin synthetic world slice. Both FREE classes touch across adjacent
    # sampled centres; all its actual receivers constrain the physics.
    o=(60,30,50)
    n=128
    pts=np.indices((2,)*3)
    cls=(2*(o[0]+pts[0])+1<123).astype(np.uint8)
    # Equality with the slab is SOLID; move its caps strictly into the gap.
    body=box((488,-1024,-1024),(490,2048,2048),1024)
    obs=Observation([body],cls,o,n)
    r=run(obs,'interface',37,8,10,128,1,'envelope')
    assert not r['unknown_pieces'] and r['certified_tv_005']
    # Left of the slab, outside the closed window. The window's legal-left
    # overlap is a rectangular subvolume from x=o[0]/n to x=488/1024-rho.
    left=Q(488,1024)-Q(1,2**26)
    expected=left-(left-Q(o[0],n))*Q(2,n)**2
    total=sum(Poly([tuple(tuple(v) for v in f) for f in faces]).volume for faces in r['inner_pieces'])
    assert total==expected
    assert cell_mass(r,0)==Q(1,128**3)/expected


def test_euclidean_clearance_at_face_edge_and_vertex():
    from membership import clear
    b=box((2,2,2),(4,4,4),8)
    v=Q(1,4)
    r=Q(1,2**26)
    assert not clear(b,(v-r,v,v))
    assert clear(b,(v-r-Q(1,2**90),v,v))
    assert not clear(b,(v-3*r/5,v-4*r/5,v))
    assert clear(b,(v-3*r/5,v-4*r/5,v-r/100))
    assert clear(b,(v-r,v-r,v-r))


def test_reciprocal_raster_against_exact_point_oracle_on_small_crops():
    from raster import encoded,scan
    bodies=[box((6,4,5),(8,10,12),16),box((11,7,1),(12,9,8),16)]
    rng=np.random.default_rng(3101)
    for o in ((0,0,0),(5,4,3),(3,0,2)):
        obs=Observation(bodies,np.zeros((4,)*3,np.uint8),o,8)
        p=within(Poly.cube(),box((1,1,1),(2,3,2),16))
        rows,offsets=encoded(obs,p)
        for loc in product(range(4),repeat=3):
            r=obs.world(loc)
            phys=[]
            for s in p.vertices:
                # Independent exact primal segment intersection.
                xyz=tuple(Q(a,s[3]) for a in s[:3])
                uvw=tuple(Q(a,16) for a in r[:3])
                phys.append(any(segment_hits(b,xyz,uvw,1) for b in bodies))
            if any(all(form(h,r)<=0 for h in b) for b in bodies): continue
            for active in (0,1):
                cls=np.full((4,)*3,2,np.uint8)
                cls[loc]=active
                state,_,_=scan(np.ascontiguousarray(cls.T),np.array(o),16,rows,offsets,np.empty(0,np.int64))
                if state<0: assert all(hit==bool(active) for hit in phys)
                if state==0: assert all(hit!=bool(active) for hit in phys)


def test_verified_support_planes_enclose_actual_shadow():
    from penumbra import directions
    p=within(Poly.cube(),box((1,2,3),(2,3,4),16))
    k=within(Poly.cube(),box((8,7,9),(10,9,11),16))
    rows=directions(k,p)
    assert len(rows)>=3
    for row in rows:
        assert all(form(row,v)<=0 for v in k.vertices)
        assert all(form(row,v)>=0 for v in p.vertices)


def test_grid_raster_exact_slab_global_from_unit_world():
    from membership import contains
    body=box((488,-1024,-1024),(490,2048,2048),1024)
    o=(60,30,50)
    cls=(2*(o[0]+np.indices((2,)*3)[0])+1<123).astype(np.uint8)
    obs=Observation([body],cls,o,128)
    for q in ((Q(1,4),Q(1,4),Q(1,4)),(Q(488,1024)-Q(1,2**26),Q(1,4),Q(1,4)),(Q(9,10),Q(1,4),Q(1,4))):
        assert contains(obs,q)==(q[0]<Q(488,1024)-Q(1,2**26))
    r=run(obs,'interface',37,4,10,128,1,'raster')
    assert r['certified_tv_005'] and r['unknown_pieces']==0


def test_output_grid_can_be_exact_with_unresolved_continuous_mass():
    from research import grid_atom_bound
    outer=within(Poly.cube(),box((20,30,40),(21,31,41),128))
    known=within(outer,[(4096,0,0,-641)])
    atom=grid_atom_bound([known],[outer])
    assert Q(atom['tv'])==0 and atom['cell']==(20*128+30)*128+40
    adjacent=within(Poly.cube(),box((21,30,40),(22,31,41),128))
    atom=grid_atom_bound([known],[outer,adjacent])
    assert Q(atom['tv'])==adjacent.volume/(adjacent.volume+known.volume)
    assert not atom['certified_tv_005']


def test_scanline_rounding_never_upgrades_near_boundary_points():
    from raster import fixed
    # Both signs, sub-quantum displacements from an exactly incident plane,
    # and values at the legal int64 kernel limit.
    for sign in (-1,1):
        for displacement in (-1,0,1):
            m=1<<90
            row=(sign*m, -3*m, 2*m, displacement)
            down,up=(fixed([row],mode)[0] for mode in (False,True))
            for x,y,z in product((1,101,2047),repeat=3):
                p=(x,y,z,2048)
                true=form(row,p)*(1<<45)
                assert form(down,p)*max(map(abs,row))<=true<=form(up,p)*max(map(abs,row))
                assert max(abs(form(r,p)) for r in (down,up)) < 1<<60


def test_joint_raster_certificate_interior_sources_and_unknowns():
    from raster import encoded, scan
    bodies=[box((1,1,5),(4,7,12),16),box((6,3,2),(7,14,10),16)]
    src=within(Poly.cube(),box((0,0,0),(1,3,3),16))
    origin=(1,0,1)
    obs=Observation(bodies,np.zeros((4,)*3,np.uint8),origin,8)
    rr,off=encoded(obs,src)
    states=set()
    for xyz in product(range(4),repeat=3):
        r=obs.world(xyz)
        if any(all(form(h,r)<=0 for h in b) for b in bodies): continue
        # 27 strictly interior source points, on a different lattice from the vertices.
        actual=[any(segment_hits(b,s,tuple(4*x for x in r[:3]),64) for b in bodies)
                for s in product((1,2,3),(1,5,9),(2,7,11))]
        for active in (0,1):
            cls=np.full((4,)*3,2,np.uint8)
            cls[xyz]=active
            count,_,_=scan(np.ascontiguousarray(cls.T),np.array(origin),16,rr,off,np.empty(0,np.int64))
            states.add(np.sign(count))
            if count==0: assert all(h!=bool(active) for h in actual)
            if count<0: assert all(h==bool(active) for h in actual)
    assert states=={-1,0,1}
