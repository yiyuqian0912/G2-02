"""Static figure based exclusively on measured rows and the matched prefix experiment."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results",type=Path)
    args=parser.parse_args()
    rows=[r for r in map(json.loads,(args.results/"channels.jsonl").read_text().splitlines()) if "wall" in r]
    data=json.loads((args.results/"summary.json").read_bytes())["measured_pilot"]
    fig,(ax,bx)=plt.subplots(1,2,figsize=(10.4,4.2),layout="constrained",gridspec_kw={"width_ratios":[1.35,1]})
    for side,color in [(64,"#1F648F"),(128,"#368167"),(256,"#94703D")]:
        subset=[r for r in rows if r["L"]==side and r["status"]=="CERTIFIED"]
        ax.scatter([r["fraction"] for r in subset],[r["wall"] for r in subset],s=8,alpha=.55,color=color,label=f"L{side}")
    failed=[r for r in rows if r["status"]!="CERTIFIED"]
    ax.scatter([r["fraction"] for r in failed],[r["wall"] for r in failed],s=28,marker="x",color="#a0394c",label="Incomplete")
    ax.set(xscale="log",yscale="log",xlabel="Minority fraction among FREE receivers",ylabel="Measured solve time (seconds)",title="807 archived capped attempts")
    ax.legend(frameon=False,ncol=2,fontsize=8)
    methods=["natural","balance","interface","geometry"]
    left=np.zeros(len(methods))
    for label,fields,color in [("Bind + proposals",["setup","generation"],"#8aaca6"),("Screen + rank",["features","scoring"],"#edba71"),("CBC",["labeling"],"#426285")]:
        vals=np.array([sum(data[m][f] for f in fields)/data[m]["fresh_certified"] for m in methods])
        bx.barh(methods,vals,left=left,color=color,label=label)
        left+=vals
    bx.invert_yaxis()
    bx.set(xlabel="Seconds per certified source channel",title="10 matched natural-window prefixes")
    bx.legend(frameon=False,fontsize=8,loc="lower center",bbox_to_anchor=(.45,-.30),ncol=3)
    for a in (ax,bx):
        a.spines[["top","right"]].set_visible(False)
        a.grid(axis="x",alpha=.12)
    fig.savefig(args.results/"sampling-cost.png",dpi=160)


if __name__=="__main__":
    main()
