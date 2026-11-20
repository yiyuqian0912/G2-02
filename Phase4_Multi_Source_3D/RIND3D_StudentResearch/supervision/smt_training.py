"""Arithmetic lemmas for additive binary32 supervision. No executable/IEEE compiler proof."""
from z3 import BitVec, BitVecVal, Extract, Int, Solver, ULE, ZeroExt, unsat


def prove(name, assumptions, violation):
    solver = Solver()
    solver.add(*assumptions, violation)
    if solver.check() != unsat:
        raise AssertionError(name)
    print(name, "unsat")


k = BitVec("k", 64)
ds = [ZeroExt(40,Extract(s+23,s,k)) if s<48 else ZeroExt(48,Extract(63,48,k)) for s in (0,24,48)]
prove("tick reconstruction", [ULE(k,BitVecVal(1<<52,64))],
      k != ds[0] + (ds[1]<<24) + (ds[2]<<48))
prove("top digit fits binary32 exactly", [ULE(k,BitVecVal(1<<52,64))], ~ULE(ds[2],BitVecVal(16,64)))
a,b = Int("subtotal"),Int("next_term")
prove("int64 chunk addition", [a>=0,a<=1023*(1<<52),b>=0,b<=1<<52], a+b>=1<<63)
