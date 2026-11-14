import copy
import numpy as np
import pytest
from tests.test_lite_production import box
from rind3d.rendering.context import GeometryContext
from rind3d.inverse_research import InverseConstraints,explore
from rind3d.inverse_certificate import verify,partition
from rind3d.windows import Window

def test_independent_full_fme_rechecks_pruning_and_partition():
    ctx=GeometryContext(box());truth=np.array([.1,.2,.3]);w=Window((768,384,384),128)
    points=(np.array([[800,450,450],[820,490,490],[850,390,410],[770,500,430]])+.5)/1024;labels=ctx.bind(truth).response(truth,points)
    c=InverseConstraints(ctx,points,labels);_,r=explore(c,truth,w,seconds=10,max_regions=80,max_depth=3,separation=0,record_rejections=True)
    result=verify(ctx,points,labels,truth,w,r)
    assert result['complete'] and result['partition_complete'] and not result['continuous_uniqueness_proven']
    bad=copy.deepcopy(r);bad['outer_cells'].pop()
    with pytest.raises(ValueError):verify(ctx,points,labels,truth,w,bad)
    bad=copy.deepcopy(r);bad['outer_cover_volume_exact']='2'
    with pytest.raises(ValueError,match='volume'):verify(ctx,points,labels,truth,w,bad)
    outside=points.copy();outside[0,0]=(700+.5)/1024
    with pytest.raises(ValueError,match='observed voxel'):verify(ctx,outside,labels,truth,w,r)

def test_dyadic_partition_rejects_overlaps_and_holes():
    root={'lower':[0,0,0],'upper':[1,1,1],'depth':0};assert partition([root])
    with pytest.raises(ValueError):partition([root,root])
    with pytest.raises(ValueError):partition([{'lower':[0,0,0],'upper':[.5,.5,.5],'depth':1}])
