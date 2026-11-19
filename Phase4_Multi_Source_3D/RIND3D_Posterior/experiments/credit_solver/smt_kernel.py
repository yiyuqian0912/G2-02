"""Mechanized (z3) lemmas for the arithmetic cores of the trusted kernel.

Each claim is the exact expression the code evaluates, with the code location; the control
flow around it is argued in XLF_CBC_DESIGN.md (section 8, code-level proofs). Run:
  python smt_kernel.py        (prints PROVED for every claim, then ALL PROVED)

  S1  poly.Poly.split           crossing point |v|a + |u|b lies on the plane, strictly inside the edge
  S2  poly.cap_hull / polycheck homogeneous 2D orientation has the sign of the true orientation
  S3  polycheck._lex, cap_hull  homogeneous lexicographic comparison is the true comparison
  C1  solver.cone               segment-meets-halfspaces elimination (front i, back j, tangent k)
  C2  solver.cone               the pair row is implied when the back facet imposes no bound
  E1  envelopes.Envelope.inner  the tightest pair row implies all others under f_i(r) <= 0
  E2  envelopes.Envelope.outer  the loosest pair row contains all others under f_i(r) <= 0
  E3  separators / penumbra     a separating plane bounds every shadow point k + l(k - s)
  Q1  envelopes.clearance_covers directed integer square root: up^2 >= m
  Q2  envelopes.clearance_covers outside the offset row => distance > 2^-26
  T1  frontier theorem          x >= l > 0, 0 <= y <= U  =>  x/(x+y) >= l/(l+U)
  T2  frontier theorem          q >= a  =>  min(p, q) >= min(p, a)
  T3  kernel.wire_bound         min(floor(pC), floor(lC/(l+o))) <= C * min(p, l/(l+o))
"""
import z3


def prove(claim, name, timeout=120000):
    s = z3.Solver()
    s.set("timeout", timeout)
    s.add(z3.Not(claim))
    r = s.check()
    print(f"{name}: {'PROVED' if r == z3.unsat else r}")
    return r == z3.unsat


def main():
    ok = []
    R = z3.Reals
    # S1: crossing point. Forms are linear, so form(row, |v|a + |u|b) = |v| u + |u| v with
    # u = form(row, a), v = form(row, b); weights |v| a_w and |u| b_w are positive.
    u, v, aw, bw = R("u v aw bw")
    absu, absv = z3.If(u >= 0, u, -u), z3.If(v >= 0, v, -v)
    opp = z3.Or(z3.And(u < 0, 0 < v), z3.And(v < 0, 0 < u))
    ok.append(prove(z3.Implies(opp, absv * u + absu * v == 0), "S1a crossing point on the plane"))
    hw = absv * aw + absu * bw
    lam = absv * aw / hw
    ok.append(prove(z3.Implies(z3.And(opp, aw > 0, bw > 0), z3.And(hw > 0, 0 < lam, lam < 1)),
                    "S1b crossing point strictly inside the edge (positive weight, convex combination)"))
    # S2: homogeneous orientation. a = (xa*aw, ya*aw, aw) etc.; cap_hull compares
    # dx*(c_j a_w - a_j c_w) with dy*(c_i a_w - a_i c_w); polycheck uses consecutive differences.
    xa, ya, xb, yb, xc, yc, cw = R("xa ya xb yb xc yc cw")
    ai, aj, bi, bj, ci, cj = xa * aw, ya * aw, xb * bw, yb * bw, xc * cw, yc * cw
    true_orient = (xb - xa) * (yc - ya) - (yb - ya) * (xc - xa)
    cap = (bi * aw - ai * bw) * (cj * aw - aj * cw) - (bj * aw - aj * bw) * (ci * aw - ai * cw)
    pos = z3.And(aw > 0, bw > 0, cw > 0)
    ok.append(prove(z3.Implies(pos, cap == aw * aw * bw * cw * true_orient), "S2a cap_hull orientation identity"))
    chk = (bi * aw - ai * bw) * (cj * bw - bj * cw) - (bj * aw - aj * bw) * (ci * bw - bi * cw)
    turn = (xb - xa) * (yc - yb) - (yb - ya) * (xc - xb)
    ok.append(prove(z3.Implies(pos, chk == aw * bw * bw * cw * turn), "S2b polycheck turn identity"))
    ok.append(prove(z3.Implies(pos, z3.And((cap > 0) == (true_orient > 0), (chk > 0) == (turn > 0))),
                    "S2c orientation signs are exact"))
    # S3: lexicographic comparison of x = p_i / p_w by cross-multiplication with positive w.
    ok.append(prove(z3.Implies(z3.And(aw > 0, bw > 0), (ai * bw < bi * aw) == (xa < xb)), "S3 homogeneous comparison"))
    # C1: segment x(t) = (1-t) r + t s, t in [0, 1], meets {f_i <= 0, f_j <= 0, f_k <= 0}
    # with receiver values hi > 0 (front), hj < 0 (back), hk = 0 (tangent) and source values si, sj, sk.
    hi, hj, si, sj, sk, t = R("hi hj si sj sk t")
    meets = z3.Exists([t], z3.And(0 <= t, t <= 1, (1 - t) * hi + t * si <= 0, (1 - t) * hj + t * sj <= 0,
                                  t * sk <= 0))
    rows = z3.And(si <= 0, sk <= 0, hi * sj - hj * si <= 0)  # solver.cone: facets with h >= 0, pair row
    pre = z3.And(hi > 0, hj < 0)
    ok.append(prove(z3.Implies(pre, meets == rows), "C1 inverse-cone elimination (front, back, tangent)"))
    # C2: with sj <= hj the pair row holds automatically (the back facet bounds nothing).
    ok.append(prove(z3.Implies(z3.And(pre, si <= 0, sj <= hj), hi * sj - hj * si <= 0), "C2 redundant pair row"))
    # E1/E2: rows f_j + rho f_i <= 0 (rho = -h_j/h_i > 0) under f_i(r) <= 0.
    fi, fj, rho, rmin, rmax = R("fi fj rho rmin rmax")
    ok.append(prove(z3.Implies(z3.And(fi <= 0, rmin <= rho, fj + rmin * fi <= 0), fj + rho * fi <= 0),
                    "E1 tightest pair row (minimal ratio) implies the others"))
    ok.append(prove(z3.Implies(z3.And(fi <= 0, rho <= rmax, fj + rho * fi <= 0), fj + rmax * fi <= 0),
                    "E2 loosest pair row (maximal ratio) contains the others"))
    # E3: affine h with h(k) <= 0 <= h(s) and r = k + l (k - s), l >= 0  =>  h(r) <= 0.
    hk, hs, l = R("hk hs l")
    ok.append(prove(z3.Implies(z3.And(hk <= 0, hs >= 0, l >= 0), hk + l * (hk - hs) <= 0),
                    "E3 separating plane bounds the shadow"))
    # Q1: up = isqrt(m) (floor), then up += (up*up < m)  =>  up^2 >= m.
    m, q = z3.Ints("m q")
    up = z3.If(q * q < m, q + 1, q)
    ok.append(prove(z3.Implies(z3.And(m >= 0, q >= 0, q * q <= m, m < (q + 1) * (q + 1)), up * up >= m),
                    "Q1 directed integer square root"))
    # Q2: offset row 2^66 f - up <= 0 with up >= 2^40 |n|; outside it, f > up / 2^66 >= 2^-26 |n|.
    f, nn, upr = R("f nn upr")
    ok.append(prove(z3.Implies(z3.And(nn > 0, upr >= 2 ** 40 * nn, 2 ** 66 * f - upr > 0), f / nn > z3.RealVal(1) / 2 ** 26),
                    "Q2 clearance distance > 2^-26"))
    # T1: per-cell lower bound of q_c = x / (x + y) (x = mass in the cell, y = mass outside).
    x, y, lo, U = R("x y lo U")
    ok.append(prove(z3.Implies(z3.And(x >= lo, lo > 0, 0 <= y, y <= U), x / (x + y) >= lo / (lo + U)),
                    "T1 frontier per-cell lower bound"))
    p, qq, a = R("p qq a")
    ok.append(prove(z3.Implies(qq >= a, z3.If(p < qq, p, qq) >= z3.If(p < a, p, a)), "T2 min monotone"))
    # T3: P = p*C (an integer: p is a multiple of 2^-52, C = 2^96); A = floor(l C / (l + o)).
    P, li, o, C, A = z3.Ints("P li o C A")
    floor_def = z3.And(A * (li + o) <= li * C, li * C < (A + 1) * (li + o))
    exact_min = z3.If(P * (li + o) < li * C, P * (li + o), li * C)  # (l+o) * C * min(p, l/(l+o))
    integer_min = z3.If(P < A, P, A)
    ok.append(prove(z3.Implies(z3.And(li > 0, o >= 0, C > 0, P >= 0, floor_def), integer_min * (li + o) <= exact_min),
                    "T3 integer TV floors round the bound outward"))
    print("ALL PROVED" if all(ok) else "NOT ALL PROVED")
    return all(ok)


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
