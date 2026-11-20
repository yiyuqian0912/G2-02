"""Research policies for two consecutive eligible natural windows. Keep all eligible channels."""
from dataclasses import dataclass
from hashlib import sha256
import math

import numpy as np


def covariates(r, mode):
    x=[math.log2(r["L"]/64),math.log1p(r["bodies"])]
    if mode in ("balance","local","geometric"):
        x.append(math.log(max(r["fraction"],1e-12)))
    if mode in ("local","geometric"):
        x.extend([math.log(r["edges"]/r["L"]**2),r["orientation"],
                  math.log(r["bins"]),math.log(r["clusters"])])
    if mode=="geometric":
        a,b=r["geo"]["steps"]
        x.extend([math.log(max(b["vol"],1e-300)),math.log(max(b["vol"]/a["vol"],1e-300)),
                  math.log(b["pieces"]),math.log(r["geo"]["explanations"])])
    return x


@dataclass
class CostModel:
    mode: str
    mean: np.ndarray
    scale: np.ndarray
    beta: np.ndarray
    intercept: float

    @classmethod
    def fit(cls,rows,mode):
        x=np.array([covariates(r,mode) for r in rows])
        y=np.log([r["wall"] for r in rows])
        scale=x.std(axis=0)
        scale[scale==0]=1
        z=(x-x.mean(axis=0))/scale
        beta=np.linalg.solve(z.T@z+np.eye(z.shape[1]),z.T@(y-y.mean()))
        return cls(mode,x.mean(axis=0),scale,beta,float(y.mean()))

    def predict(self,r):
        return float(self.intercept+((covariates(r,self.mode)-self.mean)/self.scale)@self.beta)


def wants_alternative(first,policy,seed=12119):
    """Can be evaluated before generating a second eligible window."""
    if policy not in ("natural","balance","interface","geometry","cost","mixture"):
        raise ValueError("unknown policy")
    if policy=="natural":
        return False
    if policy!="mixture":
        return True
    key=f"{seed}/{first[0]['slot']}/{first[0]['L']}/{first[0]['w']}"
    return sha256(key.encode()).digest()[0]>=128


def select(pair,policy,model=None,seed=12119):
    """Return an index. A changed eligibility mask keeps the original window (paired estimand)."""
    if len(pair)!=2 or any(not w for w in pair):
        raise ValueError("requires two eligible windows")
    a,b=pair
    if (a[0]["scene"],a[0]["L"])!=(b[0]["scene"],b[0]["L"]):
        raise ValueError("require same scene/source ensemble and side")
    if any(r["stream"]!="natural" for w in pair for r in w):
        raise ValueError("coverage stream is outside this experiment")
    if not wants_alternative(a,policy,seed) or [r["j"] for r in a]!=[r["j"] for r in b]:
        return 0
    if policy=="mixture":
        policy="cost"
    def score(w):
        if policy=="balance":
            return min(r["fraction"] for r in w)
        if policy=="interface":
            return np.mean([math.log(r["edges"]/r["L"]**2)+r["orientation"]+math.log(r["bins"])/3 for r in w])
        if policy=="geometry":
            if any(len((r.get("geo") or {}).get("steps",[]))!=2 or r["geo"]["status"]!="ok" for win in pair for r in win):
                return 0.0  # equal scores preserve first
            return -np.mean([math.log(max(r["geo"]["steps"][1]["vol"],1e-300)) for r in w])
        if policy=="cost":
            return -np.mean([math.exp(model.predict(r)) for r in w])
        raise ValueError("unknown policy")
    return int(score(b)>score(a))
