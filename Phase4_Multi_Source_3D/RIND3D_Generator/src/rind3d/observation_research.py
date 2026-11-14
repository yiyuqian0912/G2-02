"""Bounded exact packed-bit observation samples (private research only)."""
import numpy as np
LUT=np.array([i.bit_count() for i in range(256)],dtype=np.uint8)

def sample_class(response,solid,window,lit,*,count=32,seed=0):
    """Stratify across octants, sample uniformly among class voxels per octant.

    Byte-prefix arrays are local-window sized, never world tensors. Sampling is
    descriptive rather than an unbiased estimator of the full-window mean;
    callers record octant populations for optional weighted estimates.
    """
    side=window.size;y=response.read_window(*window.origin,side,packed=True).data.reshape(side,side,side//8)
    m=solid.read_window(*window.origin,side,packed=True).data.reshape(side,side,side//8)
    if np.any(y&m):raise ValueError('Lit solid observation')
    bits=(y if lit else ~y)&~m;rng=np.random.default_rng(seed);points=[];populations=[];octants=[]
    for k in range(8):
        lo=np.array([(side//2 if k&(1<<a) else 0) for a in range(3)]) # ZYX
        hi=lo+side//2;part=np.ascontiguousarray(bits[lo[0]:hi[0],lo[1]:hi[1],lo[2]//8:hi[2]//8]);flat=part.ravel()
        prefix=np.cumsum(LUT[flat],dtype=np.uint64);n=int(prefix[-1]);populations.append(n)
        target=count//8+(k<count%8)
        if n==0 or target==0:continue
        ranks=np.sort(rng.choice(n,min(target,n),replace=False));positions=np.searchsorted(prefix,ranks,side='right')
        previous=np.where(positions>0,prefix[np.maximum(positions-1,0)],0)
        for i,offset in zip(positions,ranks-previous):
            byte=int(flat[i]);active=[b for b in range(8) if byte&(1<<b)];bit=active[int(offset)]
            z,y,bx=np.unravel_index(int(i),part.shape);point=np.array([z,y,bx*8+bit])+lo;points.append(point[::-1]+window.origin);octants.append(k)
    return np.asarray(points,dtype=np.int64).reshape(-1,3),{'octant_class_counts':populations,'sample_octants':octants,'sampling':'uniform without replacement inside each nonempty octant; bounded class-stratified sample'}

def sparse_observations(response,solid,window,*,count=32,seed=0):
    arrays=[];labels=[]
    for label in (True,False):
        points,_=sample_class(response,solid,window,label,count=count//2,seed=seed+int(label))
        arrays.extend(points);labels.extend([label]*len(points))
    if not arrays or not any(labels) or all(labels):raise ValueError('Sparse observations require both free response classes')
    return (np.asarray(arrays,dtype=float)+.5)/response.shape[0],np.asarray(labels,dtype=bool)

def transition_observations(response,solid,window,*,count=32,seed=0):
    """Exact adjacent FREE lit/shadow voxel pairs; no continuous-plane guess.

    Query actual packed bits along all three axes, then spread pair midpoints
    spatially. A valid window need not have a free/free adjacent transition
    (solid barriers can separate free components), so missing pairs fall back
    to the original exact class-stratified observation samples.
    """
    side=window.size;y=response.read_window(*window.origin,side,packed=True).data.reshape(side,side,side//8);m=solid.read_window(*window.origin,side,packed=True).data.reshape(side,side,side//8)
    if np.any(y&m):raise ValueError('Lit solid observation')
    rng=np.random.default_rng(seed);pairs=[];total_pairs=0
    for axis in range(3): # ZYX
        if axis==2:
            shifted_y=y>>1;shifted_m=m>>1
            shifted_y[:,:,:-1]|=(y[:,:,1:]&1)<<7;shifted_m[:,:,:-1]|=(m[:,:,1:]&1)<<7
            bits=(y^shifted_y)&~m&~shifted_m;bits[:,:,-1]&=127
        else:
            a=[slice(None)]*3;b=list(a);a[axis]=slice(0,-1);b[axis]=slice(1,None)
            bits=(y[tuple(a)]^y[tuple(b)])&~m[tuple(a)]&~m[tuple(b)]
        for bucket in range(8):
            lo=np.array([side//2 if bucket&(1<<k) else 0 for k in range(3)]);hi=np.minimum(lo+side//2,np.array([bits.shape[0],bits.shape[1],bits.shape[2]*8]));part=np.ascontiguousarray(bits[lo[0]:hi[0],lo[1]:hi[1],lo[2]//8:hi[2]//8]);flat=part.ravel()
            if not len(flat):continue
            prefix=np.cumsum(LUT[flat],dtype=np.uint64);n=int(prefix[-1])
            total_pairs+=n
            if not n:continue
            ranks=np.sort(rng.choice(n,min(8,n),replace=False));positions=np.searchsorted(prefix,ranks,side='right');previous=np.where(positions>0,prefix[np.maximum(positions-1,0)],0)
            for i,offset in zip(positions,ranks-previous):
                active=[b for b in range(8) if int(flat[i])&(1<<b)];z,v,x=np.unravel_index(int(i),part.shape);p=np.array([z,v,x*8+active[int(offset)]])+lo;q=p.copy();q[axis]+=1
                label=bool(int(y[p[0],p[1],p[2]//8])&(1<<int(p[2]%8)));pairs.append((p,q,label,axis))
    chosen=[];used=set();axes=set();centres=[]
    while len(chosen)<count//2 and len(used)<len(pairs):
        best=None;score=-1
        for i,(p,q,_,axis) in enumerate(pairs):
            if i in used:continue
            centre=(p+q)/2;distance=min((np.linalg.norm(centre-c) for c in centres),default=float(side));gain=distance+(side*.1 if axis not in axes else 0)
            if gain>score:score=gain;best=i
        used.add(best);p,q,label,axis=pairs[best];chosen.append((p,q,label));axes.add(axis);centres.append((p+q)/2)
    points={}
    for p,q,label in chosen:points[tuple(p[::-1]+window.origin)]=label;points[tuple(q[::-1]+window.origin)]=not label
    if len(points)<count:
        other,labels=sparse_observations(response,solid,window,count=count,seed=seed)
        for p,label in zip(other,labels):
            coord=tuple(np.rint(p*response.shape[0]-.5).astype(int));points.setdefault(coord,bool(label))
            if len(points)>=count:break
    # Preserve the pair-wise farthest order so a prefix retains both sides of
    # several spatially separated transitions, not one dense class cluster.
    order=list(points);labels=np.array([points[p] for p in order],dtype=bool)
    return (np.asarray(order,dtype=float)+.5)/response.shape[0],labels,{'total_adjacent_free_transition_pairs_exact':total_pairs,'sampled_exact_transition_pairs':len(pairs),'pairs_selected':len(chosen),'axes_used':sorted(axes),'axis_order':'zyx','fallback_used':len(chosen)<count//2,'source_space_planes_not_inferred':True}
