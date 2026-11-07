"""Measured cost figures and illustrative ray samples with checked proof bounds."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from ..models import load


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--metrics',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--root',type=Path,default=Path('outputs/gaussian-b2-max-cert-v1'));p.add_argument('--input',type=Path,default=Path('outputs/gaussian-geometry-v1/distribution-v2'));a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True);j=json.loads(a.metrics.read_text());plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    supports=[('evaluation-final-v5','Cost-stratified holdout','#b76144'),('followup-holdout-v1','Uniform residual holdout','#397aab'),('ordinary-controls-v6','Uniform FREE receivers','#4f9278')]
    fig,axes=plt.subplots(1,2,figsize=(12,5.2),layout='constrained')
    for name,label,color in supports:
        rows=[r for r in j['runs'][name]['matched_ray_medians'] if r['expected']=='MISS']
        x=np.array([r['methods']['A_primary']['seconds_median']*1000 for r in rows]);y=np.array([r['methods']['E_lean']['seconds_median']*1000 for r in rows])
        axes[0].scatter(x,y,s=13,c=color,alpha=.5,label=label)
        x=np.array([r['methods']['A_independent']['seconds_median']*1000 for r in rows]);y=np.array([r['methods']['E_lean_proof']['checker_seconds_median']*1000 for r in rows])
        axes[1].scatter(x,y,s=13,c=color,alpha=.5,label=label)
    for ax,title in zip(axes,('Primary solver: old vs lean hybrid','MISS proof check vs independent search')):
        lo,hi=ax.get_xlim();low=max(1e-3,min(ax.get_xlim()[0],ax.get_ylim()[0]));high=max(ax.get_xlim()[1],ax.get_ylim()[1]);ax.plot([low,high],[low,high],'k--',lw=1)
        ax.set(xscale='log',yscale='log',title=title,xlabel='Old implementation (ms)',ylabel='New implementation (ms)');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Matched MISS queries, median of 3 trials — below diagonal is faster')
    fig.savefig(a.output/'matched-ray-costs.png',dpi=160);plt.close(fig)

    fig,axes=plt.subplots(1,3,figsize=(14,4.8),layout='constrained')
    for ax,(name,label,_) in zip(axes,supports):
        g=j['runs'][name]['groups']['MISS'];mean=lambda key:np.mean(list(g[key]['trial_total_seconds'].values()))
        oldp=mean('A_primary');oldr=mean('A_independent');newp=mean('E_lean_proof');proof=g['E_lean_proof']['proof'];check=proof['checker_seconds_mean_total']
        ser=proof['serialization_seconds_mean_total'];disk=proof['persistence_seconds_mean_total'];denominator=oldp+oldr
        components=[('Primary + proof',[oldp,newp,newp],'#4f9278'),('JSON encode',[0,ser,ser],'#dfb458'),('Checker',[0,check,check],'#568cb2'),('Existing independent',[oldr,0,oldr],'#969ca5'),('Proof files',[0,disk,disk],'#b77da1')]
        bottom=np.zeros(3)
        for title,values,color in components:
            v=np.array(values)/denominator;ax.bar(range(3),v,bottom=bottom,label=title,color=color);bottom+=v
        ax.bar_label(ax.containers[-1],labels=[f'{x:.2f}×' for x in bottom],padding=4)
        ax.set_xticks(range(3),['Old P + R','Hypothetical\nMISS-only change','Retain R\n(current contract)']);ax.tick_params(axis='x',labelsize=8)
        ax.set(title=label,ylabel='Cost / old primary + independent cost',ylim=(0,max(bottom)*1.18));ax.axhline(1,color='black',ls='--',lw=.8)
    handles,labels=axes[-1].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=5,fontsize=9)
    fig.suptitle('Serialized MISS-proof economics; proof checking does not replace the required renderer')
    fig.savefig(a.output/'proof-economics.png',dpi=160);plt.close(fig)

    first=j['field_runs']['fields32-64-v1']['results'];last=j['field_runs']['fields-lean32-64-v2']['results']
    lookup={(r['case'],r['resolution'],r['method']):r for r in first};new={(r['case'],r['resolution'],r['method']):r for r in last}
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for ax,case in zip(axes,('gaussian_sum-014','gaussian_sum-007')):
        positions=[];primary=[];independent=[];check=[];labels=[]
        for n in (32,64):
            rtime=lookup[(case,n,'A_independent')]['cold_total_seconds']
            for method,label in [('A_primary','Old'),('D_hybrid','Hybrid D'),('E_lean','Lean E'),('E_checked','E + check')]:
                row=lookup[(case,n,method)] if method=='D_hybrid' else new[(case,n,method)]
                c=row['work'].get('checker_nanoseconds',0)/1e9
                positions.append(len(positions));primary.append(row['cold_total_seconds']-c);check.append(c);independent.append(rtime);labels.append(f'{n}³\n{label}')
        ax.bar(positions,primary,label='Primary incl. setup/write',color='#4f9278');ax.bar(positions,check,bottom=primary,label='Additional native checker',color='#568cb2')
        ax.bar(positions,independent,bottom=np.array(primary)+check,label='Required original independent',color='#a7aab0');ax.set_xticks(positions,labels);ax.tick_params(axis='x',labelsize=8)
        ax.set(title=case,ylabel='Complete physical field cost (s)');ax.axvline(3.5,c='white',lw=5)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=3)
    fig.suptitle('Complete-field costs: dominant original independent work remains\nN³ samples the whole world; one cold trial per field, not a production throughput estimate')
    fig.savefig(a.output/'complete-field-costs.png',dpi=160);plt.close(fig)

    # A 1D plot is illustration only; actual proof intervals come from a checked certificate.
    run=j['runs']['evaluation-final-v5']['matched_ray_medians'];candidates=[r for r in run if r['case']=='gaussian_sum-014' and r['expected']=='MISS']
    chosen=max(candidates,key=lambda r:r['methods']['A_independent']['seconds_median']);fixture=json.loads((a.root/'frozen-rays.json').read_text());query=next(r for r in fixture['rays'] if r['ray_id']==chosen['ray_id'])
    folder=a.root/'evaluation-final-v5';proof_path=folder/f'certificate-{chosen["ray_id"]}-E_lean_proof-0.json'
    if not proof_path.exists():proof_path=folder/f'certificate-{chosen["ray_id"]}-0.json'
    cert=json.loads(proof_path.read_text())
    assert cert['query_id'];scene=load(a.input/query['case']/'geometry.json');s=np.array(query['source_xyz']);x=np.array(query['receiver_xyz'])
    t=np.linspace(0,1,2049);points=s[None]+t[:,None]*(x-s)[None];f=scene.approx_values(points)
    # approx_values returns F-tau; add the display threshold back.
    tau=scene.threshold.value if scene.threshold.kind=='binary64' else np.exp(scene.threshold.value);f=f+tau
    fig,axes=plt.subplots(2,1,figsize=(11,6.2),layout='constrained',sharex=True)
    axes[0].plot(t,f,c='#333',lw=1,label='Ordinary numerical samples (illustration)')
    for k,row in enumerate(cert['partition']):
        lo,hi,upper=[float.fromhex(row[key]) for key in ('lo','hi','upper')]
        axes[0].plot([lo,hi],[upper,upper],c='#2b8d7b',lw=2,label='Independently checked local upper' if k==0 else None)
        axes[1].plot([lo,hi],[tau-upper,tau-upper],c='#2b8d7b',lw=2)
    axes[0].axhline(tau,c='#b34d4d',ls='--',label='Canonical threshold (display)');axes[0].set(ylabel='Continuous Gaussian sum f(t)',title=f'{query["case"]}: actual checked MISS partition, ray {chosen["ray_id"][:12]}');axes[0].legend(fontsize=9)
    axes[1].set(xlabel='Closed segment parameter t ∈ [0,1]',ylabel='τ − claimed upper\n(display arithmetic)',yscale='log');axes[1].grid(alpha=.2)
    fig.suptitle('The finite cover and rechecked bounds prove MISS; the sampled curve does not')
    fig.savefig(a.output/'checked-ray-partition.png',dpi=160);plt.close(fig)


if __name__=='__main__':main()
