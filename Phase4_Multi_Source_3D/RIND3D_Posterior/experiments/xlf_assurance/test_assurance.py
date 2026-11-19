import math
import unittest
from fractions import Fraction as Q
from itertools import product

from geometry import affine, inverse_rows, physical_classifier, rows_of, segment_hit
from ledger import Box, CAP, Ledger, M, N, RECORD, WORLD, bound_on_bytes, decode, encode, grid_box


def true_boxes(boxes):
    def classify(region):
        if any(b.contains(region) for b in boxes):
            return "IN"
        if all(region.overlap(b) == 0 for b in boxes):
            return "OUT"
        return "R"
    return classify


def isolate_three(ledger):
    parent = 1
    for axis, point in ((0, Q(1,N)), (1, Q(1,N)), (2, Q(3,N))):
        parent, sibling = ledger.split(parent, axis, point)
        assert ledger.assess(sibling) == "OUT"
    a, rest = ledger.split(parent, 2, Q(1,N))
    b, c = ledger.split(rest, 2, Q(2,N))
    return a, b, c


def physical_corner():
    body = Box((Q(3,8),Q(1,16),Q(1,16)), (Q(7,16),Q(3,8),Q(15,16)))
    # A tiny complete window keeps the reference tests cheap.
    obs = [int(y >= x) for x,y,z in product(range(2), repeat=3)]
    classify = physical_classifier([body], (511,511,511), 1024, 2, obs)
    ledger = Ledger(classify)
    key = 1
    for axis, cut, side in ((0,Q(1,8192),0), (1,Q(2,8192),1),
                            (1,Q(3,8192),0), (2,Q(1,256),0)):
        key = ledger.split(key, axis, cut)[side]
    if ledger.assess(key) != "IN":
        raise ValueError("corner not proven physically feasible")
    return ledger


class AssuranceTests(unittest.TestCase):
    def test_partial_search_against_enumerated_posteriors(self):
        for truth in range(1,8):
            solids = [grid_box(i) for i in range(3) if truth & (1<<i)]
            for visited in range(8):
                ledger = Ledger(true_boxes(solids))
                keys = isolate_three(ledger)
                for i,key in enumerate(keys):
                    if visited & (1<<i):
                        ledger.assess(key)
                state, payload, bound = ledger.certify(1)
                if not truth & visited:
                    self.assertEqual((state,payload,bound), ("ABSTAIN",None,None))
                    continue
                p = decode(payload)
                q = {i:Q(1,len(solids)) for i in range(3) if truth & (1<<i)}
                error = sum(abs(p.get(i,0)-q.get(i,0)) for i in set(p)|set(q))/2
                self.assertLessEqual(error,bound)
                if visited == 7:
                    self.assertEqual(error,bound)

    def test_stale_or_invalid_split(self):
        ledger = Ledger(true_boxes([grid_box(0)]))
        for bad in (0,1,Q(-1),0.5):
            with self.assertRaises(ValueError):
                ledger.split(1,0,bad)
        keys = isolate_three(ledger)
        with self.assertRaises(KeyError):
            ledger.assess(1)
        self.assertEqual(ledger.assess(keys[0]),"IN")
        with self.assertRaises(ValueError):
            ledger.assess(keys[0])
        with self.assertRaises(ValueError):
            ledger.split(keys[0],1,Q(1,8*N))
        self.assertEqual(ledger.credits()[0], {0:CAP})

    def test_failed_predicate_preserves_state(self):
        def unknown(box):
            raise ArithmeticError("capacity")
        ledger = Ledger(unknown)
        with self.assertRaises(ArithmeticError):
            ledger.assess(1)
        self.assertEqual(ledger.certify(1), ("ABSTAIN",None,None))
        self.assertEqual(ledger.split(1,0,Q(1,2)), (2,3))

    def test_grid_probabilities_from_bytes(self):
        lower = {0:CAP, 1:CAP, M-1:CAP}
        p = decode(encode(lower))
        self.assertEqual(sum(p.values()),1)
        self.assertEqual(1-3*Q(float(Q(1,3))),Q(1,2**54))
        bad = (b"", b"!", RECORD.pack(M,1), RECORD.pack(0,math.nan),
               RECORD.pack(0,math.inf), RECORD.pack(0,-1), RECORD.pack(0,0.5),
               RECORD.pack(0,0.5)*2, RECORD.pack(1,0.5)+RECORD.pack(0,0.5))
        for payload in bad:
            with self.assertRaises(ValueError):
                decode(payload)
        p = decode(encode({0:Q(1,2**300),3:Q(1)}))
        self.assertEqual(p,{3:1})

    def test_different_valid_bytes_recosted(self):
        ledger = Ledger(true_boxes([grid_box(0)]))
        keys = isolate_three(ledger)
        for k in keys:
            ledger.assess(k)
        self.assertEqual(ledger.certify(0)[2],0)
        self.assertEqual(bound_on_bytes(encode({1:Q(1)}),*ledger.credits()),1)

    def test_uncertainty_entirely_inside_one_cell(self):
        ledger = Ledger(true_boxes([grid_box(0)]))
        keys = isolate_three(ledger)
        for key in keys[1:]:
            ledger.assess(key)
        left, _ = ledger.split(keys[0], 0, Q(1,2*N))
        ledger.assess(left)
        self.assertEqual(ledger.credits()[0], {0:CAP/2})
        self.assertEqual(ledger.certify(0)[2],0)

    def test_finalization_makes_no_physical_calls(self):
        calls = []
        def classify(box):
            calls.append(box)
            return true_boxes([grid_box(0)])(box)
        ledger = Ledger(classify)
        for key in isolate_three(ledger):
            ledger.assess(key)
        before = len(calls)
        self.assertEqual(ledger.certify(0)[2],0)
        self.assertEqual(ledger.certify(Q(1,20))[2],0)
        self.assertEqual(len(calls),before)

    def test_old_missing_sibling_fixture(self):
        ledger = physical_corner()
        status, payload, upper = ledger.certify(Q(1,20))
        self.assertEqual((status,payload), ("ABSTAIN",None))
        self.assertGreaterEqual(upper,Q(119,120))
        self.assertLess(upper,1)

    def test_physical_eligibility_and_domain(self):
        box = Box((Q(1,4),)*3,(Q(3,4),)*3)
        with self.assertRaises(ValueError):
            physical_classifier([box],(2,2,2),8,1,[1])
        f = physical_classifier([box],(2,2,2),8,1,[2])
        self.assertEqual(f(box),"OUT")
        # Region touching the strict clearance distance is left undecided.
        eps = Q(1,2**26)
        near = Box((Q(1,4)-2*eps,Q(3,8),Q(3,8)),
                   (Q(1,4)-eps,Q(1,2),Q(1,2)))
        self.assertEqual(f(near),"R")
        self.assertEqual(f(grid_box(0)),"IN")

    def test_exact_inverse_against_segment(self):
        body = Box((Q(1,4),)*3,(Q(3,4),)*3)
        rows = rows_of(body)
        coords = (Q(0),Q(1,4),Q(1,2),Q(3,4),Q(1))
        for r in ((Q(1,4),0,Q(1,2)),(0,0,0),(Q(1,2),)*3,(Q(3,4),)*3):
            shadow = inverse_rows(rows,r)
            for s in product(coords,repeat=3):
                self.assertEqual(all(affine(h,s)<=0 for h in shadow),segment_hit(rows,s,r))

    def test_distinct_blockers_at_vertices(self):
        # Two endpoint sources blocked by two objects, but their midpoint is visible.
        receiver=(Q(1,2),1,Q(1,2))
        bodies = [Box((a,Q(7,16),Q(7,16)),(a+Q(1,8),Q(9,16),Q(9,16)))
                  for a in (Q(3,16),Q(11,16))]
        def blocked(x):
            return any(segment_hit(rows_of(b),(x,0,Q(1,2)),receiver) for b in bodies)
        self.assertEqual([blocked(x) for x in (0,Q(1,2),1)],[True,False,True])

    def test_complete_room_observation(self):
        import room
        obs = room.make_observation()
        result = room.solve(obs)
        self.assertEqual((result["status"], result["upper"]),("CERTIFIED","0"))
        self.assertEqual(result["decoded"],{str((64*128+64)*128+64):"1"})
        self.assertGreater(Q(result["positive_lower"]),0)
        self.assertEqual(result["direct_free_interface_edges_inside_room"],15)
        corrupt = bytearray(obs)
        corrupt[-1] = 1
        with self.assertRaises(ValueError):
            room.solve(bytes(corrupt))

    def test_shell_geometry_with_independent_segment_clipping(self):
        import room
        obs = room.make_observation()
        source = room.unit(tuple(515+Q(f,256) for f in (119,147,131)))
        pieces = [rows_of(room.POST)]
        for axis in range(3):
            low,high = list(room.OUTER.lo),list(room.OUTER.hi)
            low[axis],high[axis] = room.CAVITY.hi[axis],room.CAVITY.lo[axis]
            pieces.extend((rows_of(Box(room.OUTER.lo,tuple(high))),
                           rows_of(Box(tuple(low),room.OUTER.hi))))
        checks = {i:r for i,r,_ in room.room_samples()}
        for x,y,z in product((0,1,3,32,63),repeat=3):
            checks[(x*64+y)*64+z] = room.unit(tuple(a+v+Q(1,2)
                                              for a,v in zip(room.ORIGIN,(x,y,z))))
        for i,r in checks.items():
            solid = any(all(affine(row,r)<=0 for row in body) for body in pieces)
            active = not any(segment_hit(body,source,r) for body in pieces)
            self.assertEqual(obs[i],2 if solid else int(active))


if __name__ == "__main__":
    unittest.main()
