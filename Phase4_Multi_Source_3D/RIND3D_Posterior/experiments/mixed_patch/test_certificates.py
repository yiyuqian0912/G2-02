from fractions import Fraction as F
from itertools import product
from random import Random

from certificates import (compile_band, halfspace_fraction, integer_row, mixed_chart,
                          segment_hits, value)
from intersection import band_cuts, joint_fraction


def box_body(lo, hi):
    rows = []
    for k in range(3):
        n = tuple(int(k==i) for i in range(3))
        rows.extend([(*(-x for x in n),lo[k]),(*n,-hi[k])])
    return tuple(rows)


def test_full_patch_compression_and_exact_posterior_volume_on_restricted_domain():
    body = box_body((4,-1,-10),(6,1,10))
    box = ((0,-4,-1),(1,-3,1))
    receiver_box = ((10,5,-1),(11,9,1))
    state, (_,i,j) = mixed_chart([body],box,receiver_box,1)
    assert state == "pencil"
    receivers = list(product(range(10,12),range(5,10),range(-1,2)))
    obs = [(r,not segment_hits(body,(0,-3,0),r,1)) for r in receivers]
    band = compile_band(body[i],body[j],obs,1)
    assert (band.lower,band.upper) == (F(-8,7),-1)
    assert band.fraction_of_box(box,1) == F(1,7)
    assert joint_fraction(box,1,band_cuts(band.front,band.back,band.lower,band.upper)) == F(1,7)
    # At each x the y width is max(0,(4-8*x)/7). Its integral is 1/7.
    halves = [(box[0],(F(1,2),-3,1)), ((F(1,2),-4,-1),box[1])]
    assert [band.fraction_of_box(q,1) for q in halves] == [F(2,7),0]
    assert band.accepts((0,-3,0),1)  # upper equality is inactive (contact)
    assert not band.accepts((F(1,2),-3,0),1)  # lower equality fails an active observation
    for x,y in product(range(13),repeat=2):
        s=(F(x,12),F(y,12)-4,F(1,3))
        assert band.accepts(s,1) == all(segment_hits(body,s,r,1) != active for r,active in obs)


def test_ambiguous_union_and_changing_aspect():
    body=box_body((4,-1,-10),(6,1,10))
    box=((0,-4,-1),(1,-3,1))
    patch=((10,5,-1),(11,9,1))
    assert mixed_chart([body,body],box,patch,1)[0] == "unknown"
    assert mixed_chart([body],((0,-4,-1),(5,0,1)),patch,1)[0] == "unknown"
    assert mixed_chart([body],box,((10,20,-1),(11,21,1)),1)[0] == "constant_miss"
    assert mixed_chart([body],box,((10,3,-1),(11,4,1)),1)[0] == "constant_hit"


def test_joint_volume_endpoint_conventions_and_degeneracy():
    q=((0,0,0),(1,1,1))
    assert joint_fraction(q,1,[]) == 1
    assert joint_fraction(q,1,[((0,0,0,0),True)]) == 0
    assert joint_fraction(q,1,[((0,0,0,0),False)]) == 1
    assert joint_fraction(q,1,[((1,1,1,-1),False)]) == F(1,6)
    assert joint_fraction(q,1,[((2,0,0,-1),False),((0,3,0,-1),True)]) == F(1,6)
    assert joint_fraction(q,1,[((2,0,0,-1),False),((-2,0,0,1),False)]) == 0
    assert joint_fraction(q,1,[((1,0,0,0),True)]) == 0
    rng=Random(19)
    for i in range(70):
        r=tuple(rng.randrange(-6,7) for _ in range(4))
        assert joint_fraction(q,1,[(r,False)]) == halfspace_fraction(r,q,1)


def test_random_products_match_primal_clipping_when_decided():
    rng=Random(37)
    body=box_body((3,-1,-2),(5,1,2))
    counts={}
    for _ in range(250):
        s0=(rng.randrange(-6,2),rng.randrange(-6,7),rng.randrange(-5,6))
        r0=(rng.randrange(5,13),rng.randrange(-6,7),rng.randrange(-5,6))
        q=(s0,tuple(x+1 for x in s0))
        w=(r0,tuple(x+1 for x in r0))
        state,which=mixed_chart([body],q,w,1)
        counts[state]=counts.get(state,0)+1
        if state == 'unknown':
            continue
        for dx in (0,F(1,2),1):
            s=tuple(x+dx for x in s0)
            for shift in product((0,F(2,3),1),repeat=3):
                r=tuple(x+d for x,d in zip(r0,shift))
                if state=='pencil':
                    _,i,j=which
                    expected=value(body[i],s,1)*value(body[j],r,1) <= value(body[j],s,1)*value(body[i],r,1)
                else:
                    expected=state=='constant_hit'
                assert segment_hits(body,s,r,1) == expected
    assert set(counts)=={'pencil','constant_hit','constant_miss','unknown'}


def test_two_bodies_refute_single_blocker_hull_inference():
    # A,B,C are inactive; their triangle contains (2,0,0) on the active ray.
    bodies=[box_body((9,-12,-12),(11,12,-8)),box_body((9,-2,8),(11,2,12))]
    s=(0,0,0)
    obs=[((20,-20,-20),False),((20,20,-20),False),((20,0,20),False),((30,0,0),True)]
    assert all((not any(segment_hits(body,s,r,1) for body in bodies)) == active for r,active in obs)


def test_true_source_contact_need_not_give_positive_feasible_volume():
    plus=box_body((9,4,0),(11,6,2))
    minus=box_body((9,-6,-2),(11,-4,0))
    observations=[((20,10,0),False),((20,-10,0),False),((20,0,0),True)]
    def feasible(s):
        return all((not any(segment_hits(b,s,r,1) for b in (plus,minus)))==active for r,active in observations)
    assert feasible((0,0,0))
    for k in (1,10,10**20):
        assert not feasible((0,0,F(1,k))) and not feasible((0,0,F(-1,k)))


def test_row_rescaling_preserves_orientation():
    assert integer_row((F(-1,3),F(1,6),F(0),F(1,2))) == (-2,1,0,3)
