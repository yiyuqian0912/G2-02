"""Small correctness checks, not dataset prevalence estimates."""
import numpy as np
from analyze import graph,response_bins,boundary_segments,geo,RINDDataset,ROOT
# No tolerance chaining.
a=np.array([[0.,.75,1.5]])
b=response_bins(a,1.)
assert list(b[0])==[0,0,1]
# Both centers free, but a thin rectangle intersects the connecting segment.
types=np.array([0],np.uint8);params=np.zeros((1,17));params[0,:5]=[1.,.5,.1,.4,0]
aabb=np.array([geo.compute_obstacle_aabb(0,params[0])])
par,u,v,blocked=graph(np.ones((1,2),bool),np.zeros((1,2),np.int64),0,0,types,params,aabb)
assert blocked==1 and len(u)==0 and par[0]!=par[1]
# Same pair two disjoint dual edges remains two segments; a bend joins.
assert len(boundary_segments(np.array([0,0]),np.array([1,1]),np.array([0,8]),np.array([1,9]),4))==2
assert len(boundary_segments(np.array([0,0]),np.array([1,1]),np.array([0,0]),np.array([1,4]),4))==1
# Ground-truth geometry reference vs accelerated adjacency on real windows.
ds=RINDDataset(ROOT.parent/'Phase2_Multi_Source/data/raw/phase2-multi-source');checked=0
for sid in (317,2233,2836,8753):
    sc=ds.get_scene(sid);t=sc['obstacle_types'];p=sc['obstacle_params'];boxes=np.array([geo.compute_obstacle_aabb(int(z),q) for z,q in zip(t,p)])
    vids=np.flatnonzero(ds.local_views[sid,:int(ds.view_counts[sid]),2]==16)[:2]
    for vid in vids:
        item=ds.get_view(sid,int(vid));x,y,L=map(int,item['view']);free=~item['obstacle']
        _,u,v,n=graph(free,response_bins(item['response'],1e-12),x,y,t,p,boxes);actual=set(zip(u.tolist(),v.tolist()));expected=set()
        for yy in range(L):
            for xx in range(L):
                for dx,dy in ((1,0),(0,1)):
                    qx,qy=xx+dx,yy+dy
                    if qx>=L or qy>=L or not free[yy,xx] or not free[qy,qx]:continue
                    hit=geo.segment_intersects_any_obstacle(x+xx+.5,y+yy+.5,x+qx+.5,y+qy+.5,t,p,len(t));checked+=1
                    if not hit:expected.add((yy*L+xx,qy*L+qx))
        assert actual==expected
print(f'PASS bounded-range bins, thin-obstacle exclusion, segment connectivity, {checked} real pixel-edge geometry checks')
