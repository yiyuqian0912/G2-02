"""Counterexample search for arithmetic consequences. These lemmas do not verify object code."""
from z3 import Ints,Reals,If,And,Not,Implies,Solver,unsat

def prove(name,claim):
    s=Solver()
    s.set(timeout=20000)
    s.add(Not(claim))
    result=s.check()
    print(name,result)
    if result!=unsat:
        raise ArithmeticError(str(result))

a,x,X,t=Reals("a x X t")
prove("affine extrema",Implies(And(x<=t,t<=X),And(If(a>=0,a*x,a*X)<=a*t,a*t<=If(a>=0,a*X,a*x))))
L,l,H,h,U,u,K,n=Reals("L l H h U u K n")
prove("overlap denominator",Implies(And(H-h>=L-l,U>=u,u<=K+n),l+H-h+U-u>=L+If(U>K+n,U-K-n,0)))
old,removed,added,rest=Ints("old removed added rest")
prove("atomic live total",Implies(old==removed+rest,old+added-removed==rest+added))
num,den,low,rem=Ints("num den low rem")
prove("tick bracket",Implies(And(den>0,num==den*low+rem,rem>=0,rem<den),And(den*low<=num,num<=den*(low+If(rem==0,0,1)))))
A,B,z=Reals("A B z")
prove("one-dimensional simplex identity",Implies(And(A>0,z>=0,z<=A),If(z>0,z,0)-If(z>A,z-A,0)==z))
p,q,m=Reals("p q m")
prove("overlap monotonicity",Implies(And(p>=0,q>=m,m>=0),If(p<m,p,m)<=If(p<q,p,q)))
