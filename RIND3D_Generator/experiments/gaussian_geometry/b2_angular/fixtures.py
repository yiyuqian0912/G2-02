"""Explicit synthetic research controls, unrelated to private production labels."""
from fractions import Fraction as Q
import numpy as np
from ..models import Kernel,KernelScene,Threshold

SOURCE=[.125,.5,.5]
RECEIVER=([.875,.375,.46875],[.9375,.625,.53125])
INNER_LO=[[.375,.445,.48],[.5625,.495,.47]]
INNER_HI=[[.405,.502,.52],[.595,.59,.53]]


def exact_hit(s,r,lo,hi):
    low,high=Q(0),Q(1)
    for a,b,l,h in zip(s,r,lo,hi):
        a,b,l,h=map(lambda x:Q(float(x)),(a,b,l,h));d=b-a
        if d==0:
            if not l<=a<=h:return False
        else:
            x,y=sorted(((l-a)/d,(h-a)/d));low=max(low,x);high=min(high,y)
            if low>high:return False
    return True


def union_scene():
    # Every individual term is below tau everywhere; only their SUM is solid.
    kernels=[]
    for lo,hi,scale in zip(INNER_LO,INNER_HI,([.05,.08,.06],[.05,.1,.08])):
        center=(np.array(lo)+hi)/2
        for sign in (-1,1):
            # Distinct nearby terms: not duplicated kernels or a coalesced
            # single ellipsoid. Every individual maximum is still below tau.
            offset=np.array([sign*2**-11,sign*2**-12,0.])
            matrix=np.diag(scale)*(1 if sign<0 else 1+2**-8)
            kernels.append(Kernel(center+offset,matrix,.45))
    return KernelScene('gaussian_sum',kernels,Threshold('binary64',.7))
