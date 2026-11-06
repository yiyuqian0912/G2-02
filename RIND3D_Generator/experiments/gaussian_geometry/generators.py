"""Seeded hierarchical proposals and explicitly approximate density calibration.

Calibration chooses new continuous PARAMETERS; it never supplies a physical
label or topology certificate. Rejection/truncation and target deficits are
retained. There is no registration with the frozen Lite generator/profile.
"""
from dataclasses import replace
import math

from flint import arb,ctx,fmpq
import numpy as np
from scipy.stats import qmc

from .models import Kernel,KernelScene,ConvexScene,SpectralScene,Threshold,rational
from .queries import ARB_LOCK,endpoint

DEFAULT={
    'version':'hierarchical-gaussian-proposals-v1','master_seed':20261105,
    'cluster_count':[2,4],'kernels_per_cluster':[3,6],
    'anchor_interval':[.19,.81],'cluster_std':[.018,.065],
    'kernel_axis_scale':[.022,.065],'amplitude_log_std':.35,
    'small_kernel_probability':.25,'small_kernel_scale':[.3,.55],
    'world_margin':.025,'center_proposal_limit':128,
    'styles':['clustered','chain','branching','elongated','multiscale','dispersed'],
    'occupancy_targets':{'very_sparse':.001,'sparse':.01,'moderate':.05,'dense':.15},
    'calibration_resolution':32,'quality_resolution':64,
    'spectral_modes':24,'correlation_lengths':[.035,.10],
    'spectral_taper_center':[.5,.5,.5],'spectral_taper_radius':[.465,.465,.465],
    'spectral_mean':.5,'spectral_threshold_floor':2**-20,
}


def rng_for(seed,ordinal,stream=0):
    return np.random.default_rng(np.random.SeedSequence([int(seed),int(ordinal),20261105,int(stream)]))


def rotation(rng):
    # Existing unchanged SO(3) recipe, realized as exact canonical binary64 M.
    from rind3d.lite_generators import rotation as original
    return original(rng)


def grid_points(n):
    z,y,x=np.meshgrid(*([(np.arange(n)+.5)/n]*3),indexing='ij')
    return np.stack((x,y,z),axis=-1).reshape(-1,3)


def log_uniform(rng,lo,hi,size=None):return np.exp(rng.uniform(np.log(lo),np.log(hi),size))


def layout(seed,ordinal,cfg=DEFAULT):
    rng=rng_for(seed,ordinal);styles=cfg['styles'];style=styles[ordinal%len(styles)]
    clusters=int(rng.integers(cfg['cluster_count'][0],cfg['cluster_count'][1]+1))
    latin=qmc.LatinHypercube(3,seed=int(rng.integers(0,2**32)))
    low,high=cfg['anchor_interval'];anchors=low+(high-low)*latin.random(clusters)
    frame=rotation(rng)
    if style in ('chain','branching'):
        base=rng.uniform(.28,.72,3)
        if style=='chain':
            t=np.linspace(-1,1,clusters)
            offsets=np.c_[.22*t,.075*np.sin(np.pi*t),.04*np.cos(np.pi*t)]
        else:
            angles=np.arange(clusters)*2*np.pi/clusters
            offsets=np.c_[.20*np.cos(angles),.20*np.sin(angles),.05*np.sin(2*angles)]
        anchors=base+offsets@frame.T
    kernels=[];cluster_records=[];rejected=0
    for j,anchor in enumerate(anchors):
        frame_j=rotation(rng);std=log_uniform(rng,*cfg['cluster_std'],3)
        if style=='dispersed':std*=.45
        if style in ('chain','branching'):std*=.5
        if style=='elongated':std=np.array([.085,.014,.014])
        covariance_factor=frame_j@np.diag(std)
        count=int(rng.integers(cfg['kernels_per_cluster'][0],cfg['kernels_per_cluster'][1]+1))
        cluster_records.append({'anchor':anchor,'covariance_factor':covariance_factor,'count':count})
        for i in range(count):
            center=None
            for _ in range(cfg['center_proposal_limit']):
                proposed=anchor+covariance_factor@rng.normal(size=3)
                if np.all((proposed>.075)&(proposed<.925)):
                    center=proposed;break
                rejected+=1
            if center is None:raise RuntimeError('Declared correlated-center proposal budget exhausted')
            axes=log_uniform(rng,*cfg['kernel_axis_scale'],3)
            orientation=rotation(rng);scale_class='base'
            if style=='elongated':
                axes=np.array([rng.uniform(.07,.11),rng.uniform(.006,.017),rng.uniform(.008,.02)])
                orientation=frame_j
            if style=='multiscale' or rng.random()<cfg['small_kernel_probability']:
                if i%3:
                    axes*=rng.uniform(*cfg['small_kernel_scale']);scale_class='detail'
            if style in ('chain','branching'):axes*=1.2
            matrix=np.asarray(orientation@np.diag(axes),dtype='<f8')
            weight=float(rng.lognormal(0,cfg['amplitude_log_std']))
            kernels.append(Kernel(center,matrix,weight,j,scale_class))
    return kernels,{'seed':int(seed),'ordinal':int(ordinal),'style':style,
                    'clusters':cluster_records,'center_rejections':rejected,
                    'center_distribution':'correlated Normal(anchor,C Cᵀ), conditioned on declared center bounds',
                    'ready_capable':False,'distribution_version':cfg['version']}


def fit_scale(kernels,target,points,margin,radii=None):
    # For ellipsoids each coordinate extent is <= ||row(M)||_1. Bounds are
    # proposal guards only: rigorous containment is checked afterwards.
    gauge=np.full(len(points),np.inf);limit=np.inf
    for i,k in enumerate(kernels):
        local=(points-k.center)@k.approx_inverse.T
        gauge=np.minimum(gauge,np.sqrt(np.einsum('ij,ij->i',local,local)))
        radius=1. if radii is None else radii[i]
        extent=np.sum(np.abs(k.matrix),axis=1)*radius
        limit=min(limit,float(np.min(np.minimum(k.center-margin,1-margin-k.center)/extent)))
    desired=float(np.quantile(gauge,target));chosen=min(desired,limit*.94)
    if chosen<=0:raise ValueError('No positive world-contained proposal scale')
    return chosen,{'density_target':float(target),'approx_desired_scale':desired,'chosen_scale':chosen,
                   'scale_limited_by_world_proposal':chosen<desired,'calibration_is_not_a_certificate':True}


def positive_sum_face_floor(kernels,margin):
    maximum=fmpq(0);m=rational(margin)
    with ARB_LOCK,ctx.workprec(256):
        for axis in range(3):
            for side in (0,1):
                upper=arb(0)
                for k in kernels:
                    distance=k.qcenter[axis]-m if side==0 else 1-m-k.qcenter[axis]
                    q=distance*distance/k.row_norms[axis] if distance>0 else fmpq(0)
                    upper+=arb(rational(k.weight))*(-arb(q)/2).exp()
                maximum=max(maximum,endpoint(upper))
    value=float(maximum)
    if rational(value)<=maximum:value=np.nextafter(value,np.inf)
    # A generative guard on tau, never a tolerance in a physical predicate.
    denominator=int(maximum.denominator)
    serialized={'mantissa':str(maximum.numerator),'exponent':-(denominator.bit_length()-1)}
    return float(value*1.02),serialized


def convex_objects(kernels,rng):
    objects=[]
    for k in kernels:
        kind=str(rng.choice(['box','thin','prism','wedge','poly'],p=[.4,.1,.2,.15,.15]))
        matrix=k.matrix.copy();params={};radius=1.
        if kind=='thin':matrix[:,2]*=.2
        elif kind=='prism':
            n=int(rng.integers(3,9));a=np.arange(n)*2*np.pi/n
            params={'normal':np.r_[np.c_[np.cos(a),np.sin(a),np.zeros(n)],[[0,0,-1],[0,0,1]]],
                    'plane_offset':np.ones(n+2)};radius=2.1
        elif kind=='wedge':
            params={'normal':np.array([[-1,0,0],[0,-1,0],[0,1,0],[0,0,-1],[1,0,1]],dtype=float),
                    'plane_offset':np.array([1,1,1,1,.4])};radius=1.5
        elif kind=='poly':
            n=int(rng.choice([8,12,16]));extra=rng.normal(size=(n-6,3));extra/=np.linalg.norm(extra,axis=1)[:,None]
            params={'normal':np.r_[np.eye(3),-np.eye(3),extra],
                    'plane_offset':np.r_[np.ones(6),rng.uniform(.6,1.05,n-6)]}
        objects.append({'center':k.center,'matrix':matrix,'kind':kind,'cluster':k.cluster,
                        'local_radius_bound':radius,**params})
    return objects


def generate(model,seed,ordinal,target,cfg=DEFAULT):
    points=grid_points(cfg['calibration_resolution']);margin=cfg['world_margin']
    if model=='spectral_grf':
        rng=rng_for(seed,ordinal,1301);count=cfg['spectral_modes'];vectors=[]
        while len(vectors)<count:
            v=tuple(map(int,rng.integers(-6,7,3)))
            if not any(v) or v in vectors or tuple(-x for x in v) in vectors:continue
            vectors.append(v)
        frequencies=(np.array(vectors,dtype=float)@rotation(rng).T)*rng.uniform(.65,1.65,3)
        length=float(rng.uniform(*cfg['correlation_lengths']))
        power=np.sum((2*np.pi*length*frequencies)**2,axis=1)
        density=np.exp(-.5*power)
        if ordinal%len(cfg['styles'])==4:density=.8*density+.2*np.exp(-.5*power/9)
        spectral=np.sqrt(density)
        spectral/=np.sqrt(np.sum(spectral**2))
        a,b=rng.normal(size=(2,count))*spectral
        scene=SpectralScene(frequencies,a,b,cfg['spectral_taper_center'],cfg['spectral_taper_radius'],
                            Threshold('binary64',cfg['spectral_threshold_floor']),cfg['spectral_mean'],
                            {'seed':seed,'ordinal':ordinal,'correlation_length':length,
                             'spectral_standard_deviations':spectral,'density_target':target,
                             'spectral_envelope':'normalized square root of finite squared-exponential spectral weights',
                             'style':'spectral_multiscale' if ordinal%len(cfg['styles'])==4 else 'spectral_anisotropic',
                             'covariance':'E(x) E(y) sum alpha_k² cos(2π f_k·(x-y))',
                             'field_definition':'finite conditional Gaussian spectral expansion with deterministic compact C1 taper',
                             'not_an_infinite_GRF_sample':True})
        values=scene.approx_values(points)+scene.threshold.approximate()
        desired=float(np.quantile(values,1-target));tau=max(desired,cfg['spectral_threshold_floor'])
        scene.threshold=Threshold('binary64',tau)
        scene.metadata.update(approx_desired_threshold=desired,chosen_threshold=tau,
                              positive_threshold_limited=tau>desired)
        return scene
    kernels,meta=layout(seed,ordinal,cfg)
    if model=='gaussian_convex':
        objects=convex_objects(kernels,rng_for(seed,ordinal,3107));gauge=np.full(len(points),np.inf);limit=np.inf
        for o in objects:
            local=(points-o['center'])@np.linalg.inv(o['matrix']).T
            values=np.max(np.asarray(o['normal'])@local.T/np.asarray(o['plane_offset'])[:,None],axis=0) if 'normal' in o else np.max(np.abs(local),axis=1)
            gauge=np.minimum(gauge,values)
            extent=np.sum(np.abs(o['matrix']),axis=1)*o['local_radius_bound']
            limit=min(limit,float(np.min(np.minimum(o['center']-margin,1-margin-o['center'])/extent)))
        desired=float(np.quantile(gauge,target));scale=min(desired,limit*.94)
        for o in objects:o['matrix']=np.asarray(o['matrix']*scale,dtype='<f8')
        meta.update(density_target=target,approx_desired_scale=desired,chosen_scale=scale,
                    scale_limited_by_world_proposal=scale<desired,calibration_is_not_a_certificate=True)
        return ConvexScene(objects,meta)
    scale,calibration=fit_scale(kernels,target,points,margin)
    kernels=[Kernel(k.center,k.matrix*scale,k.weight,k.cluster,k.scale_class) for k in kernels]
    meta.update(calibration)
    if model=='ellipsoid_union':
        meta['union_semantics']='union of independent levels w_i exp(-q_i/2)>=w_i exp(-1/2); amplitude cancels'
        return KernelScene(model,kernels,Threshold('exp',-.5),meta)
    if model!='gaussian_sum':raise ValueError('Unknown Gaussian research model')
    scene=KernelScene(model,kernels,Threshold('binary64',.5),meta)
    values=scene.approx_values(points)+.5
    desired=float(np.quantile(values,1-target));floor,exact_bound=positive_sum_face_floor(kernels,margin)
    tau=max(desired,floor,np.nextafter(0.,1.))
    scene.threshold=Threshold('binary64',tau)
    meta.update(approx_desired_threshold=desired,chosen_threshold=tau,containment_threshold_floor=floor,
                containment_upper_dyadic=exact_bound,threshold_limited_by_containment=tau>desired)
    return scene
