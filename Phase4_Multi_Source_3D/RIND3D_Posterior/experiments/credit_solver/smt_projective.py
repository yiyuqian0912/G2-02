"""Small symbolic obligations for the reciprocal mixed-patch reduction. Run with python."""
from z3 import And, Implies, Reals, SolverFor, Not, unsat


def verify():
    sf, sb, rf, rb, t, u = Reals("sf sb rf rb t u")
    # Front has an entry t, back an exit u. Multiplication by their strictly
    # positive denominators preserves ordering, including exact contact.
    entry_exit = And(sf > 0, rf <= 0, sb < 0, rb > 0,
                     t * (sf-rf) == sf, u * (rb-sb) == -sb)
    a, b, d, e, alpha, low, high = Reals("a b d e alpha low high")
    wt = alpha*d+(1-alpha)*e
    val = alpha*a+(1-alpha)*b
    obligations = {
        "closed contact and entry ordering":
            Implies(entry_exit, (t <= u) == (sf*rb-sb*rf <= 0)),
        "segment with nonpositive endpoints":
            Implies(And(sf<=0,rf<=0,t>=0,t<=1), (1-t)*sf+t*rf<=0),
        "reciprocal sign preservation":
            rf*sb-rb*sf == (-sf)*rb-sb*(-rf),
        "extrema survive same-class convex combinations":
            Implies(And(d>0,e>0,alpha>=0,alpha<=1,low*d<=a,a<=high*d,
                        low*e<=b,b<=high*e), And(wt>0,low*wt<=val,val<=high*wt))
    }
    for name, theorem in obligations.items():
        solver = SolverFor("QF_NRA")
        solver.set(timeout=10000)
        solver.add(Not(theorem))
        if solver.check() != unsat:
            raise ArithmeticError("unproved obligation: " + name)
        print(name, "PROVED")


if __name__ == "__main__":
    verify()
