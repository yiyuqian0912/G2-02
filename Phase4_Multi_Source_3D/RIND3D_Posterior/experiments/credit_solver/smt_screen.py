"""Mechanized check (z3) of the scanline interval formulas in screen2._intervals.

For one row (a, b, c, d) and one scanline (wy, wz), the screen evaluates
    v = a*(2*ox + 1) + b*wy + c*wz + d*denom
and the row is <= 0 at local x (world coordinate 2*(ox + x) + 1) iff v + 2*a*x <= 0. The code
keeps x in [lo, hi] with
    a > 0:  hi = min(hi, (-v) // (2a))            (Python/Numba floor division)
    a < 0:  lo = max(lo, -((-v) // (-2a)))
    a = 0:  empty if v > 0, else unchanged.
Claims proved here over the integers, for every row and receiver in the verified ranges
(|a|,|b|,|c|,|d| <= 2^45, 0 <= ox, 0 <= x < L <= 256, ox + L <= 1024, wy, wz in [1, 2047], denom = 2048):
  (1) the endpoint formula is exact: v + 2ax <= 0  <=>  x <= hi_row  (a > 0), resp. x >= lo_row (a < 0);
  (2) no intermediate leaves the int64 range (|v| < 2^58, |2a*x| < 2^54, |-v| fits).
SMT-LIB `div` with a positive divisor is floor division, which is Python's `//` and Numba's
documented integer semantics for int64; that correspondence is the remaining assumption.
"""
import z3


def prove(claim, name):
    s = z3.Solver()
    s.set("timeout", 120000)
    s.add(z3.Not(claim))
    r = s.check()
    print(f"{name}: {'PROVED' if r == z3.unsat else r}")
    return r == z3.unsat


def main():
    a, b, c, d, ox, x, wy, wz = z3.Ints("a b c d ox x wy wz")
    B = 2 ** 45
    denom = 2048
    rng = z3.And(*[z3.And(-B <= t, t <= B) for t in (a, b, c, d)],
                 0 <= ox, ox <= 1024 - 64, 0 <= x, x <= 255, ox + x <= 1023,
                 1 <= wy, wy <= 2047, 1 <= wz, wz <= 2047)
    v = a * (2 * ox + 1) + b * wy + c * wz + d * denom
    # (1) endpoint formulas; z3 `/` on Ints is SMT-LIB div (floor for positive divisors)
    hi_row = (-v) / (2 * a)
    lo_row = -((-v) / (-2 * a))
    ok1 = prove(z3.Implies(z3.And(rng, a > 0), (v + 2 * a * x <= 0) == (x <= hi_row)), "a>0 upper endpoint exact")
    ok2 = prove(z3.Implies(z3.And(rng, a < 0), (v + 2 * a * x <= 0) == (x >= lo_row)), "a<0 lower endpoint exact")
    ok3 = prove(z3.Implies(z3.And(rng, a == 0), (v + 2 * a * x <= 0) == (v <= 0)), "a=0 constant row")
    # (2) int64 range of every intermediate
    lim = 2 ** 63 - 1
    ok4 = prove(z3.Implies(rng, z3.And(z3.Abs(v) <= 2 ** 58, z3.Abs(2 * a * x) <= 2 ** 54,
                                         z3.Abs(a * (2 * ox + 1)) <= lim, z3.Abs(-v) <= lim)), "int64 range")
    # (3) directed rounding of raster.fixed: m = max|r_k| > 0, S = 2^45,
    #     up_k = ceil(r_k*S/m) = -((-r_k*S) // m),  down_k = floor(r_k*S/m) = (r_k*S) // m.
    #     For receiver coordinates X_k > 0 (homogeneous, w = 2N > 0):
    #       sum up_k X_k <= 0  ==>  sum r_k X_k <= 0     (guaranteed regions shrink)
    #       sum r_k X_k <= 0   ==>  sum down_k X_k <= 0  (possible regions grow)
    m = z3.Int("m")
    r = z3.Ints("r0 r1 r2 r3")
    X = z3.Ints("X0 X1 X2 X3")
    S = 2 ** 45
    pre = z3.And(m > 0, *[z3.And(-m <= rk, rk <= m) for rk in r], *[Xk > 0 for Xk in X])
    up = [-((-rk * S) / m) for rk in r]
    dn = [(rk * S) / m for rk in r]
    dot = lambda u: sum(uk * Xk for uk, Xk in zip(u, X))
    ok5 = prove(z3.Implies(pre, z3.Implies(dot(up) <= 0, dot(r) <= 0)), "fixed(up) shrinks guaranteed regions")
    ok6 = prove(z3.Implies(pre, z3.Implies(dot(r) <= 0, dot(dn) <= 0)), "fixed(down) grows possible regions")
    print("ALL PROVED" if all((ok1, ok2, ok3, ok4, ok5, ok6)) else "NOT ALL PROVED")


if __name__ == "__main__":
    main()
