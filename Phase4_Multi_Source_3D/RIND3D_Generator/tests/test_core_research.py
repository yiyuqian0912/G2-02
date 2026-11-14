import copy,json,zlib
import multiprocessing as mp,signal
import os,time,select
import numpy as np
import pytest
from tests.test_lite_production import box,config
from rind3d import _research
from rind3d.pack import atomic_json
from rind3d.geometry import save_geometry
from rind3d.rendering.context import GeometryContext
from rind3d.stream import stream_response
from rind3d.core_research import verify_quantum,checkpoint,Worker,producer_lock
from rind3d.cli import load_config,validate_config
from rind3d.physical_evidence import load_evidence
from rind3d.windows import Window
from rind3d.fast_geometry import generate,FAMILIES
from rind3d.lite_acceptance import geometry_acceptance

def test_slicing_crc_matches_independent_zlib():
    rng=np.random.default_rng(41)
    for n in (0,1,7,8,9,63,64,65,32768,65537):
        data=rng.bytes(n);assert _research.crc(data)==zlib.crc32(data)

def test_two_coordinators_cannot_write_the_same_root(tmp_path):
    with producer_lock(tmp_path):
        with pytest.raises(RuntimeError,match='producer'): 
            with producer_lock(tmp_path):pass
    with producer_lock(tmp_path):pass

def _temporary_coordinator(conn):
    ctx=mp.get_context('spawn');output=ctx.Queue(2);w=Worker(ctx,0,output);conn.send(w.process.pid)
    while True:time.sleep(.2)

def test_worker_dies_when_coordinator_is_killed_before_resume():
    ctx=mp.get_context('spawn');read,write=ctx.Pipe(False);p=ctx.Process(target=_temporary_coordinator,args=(write,));p.start();pid=None;fd=None
    try:
        assert read.poll(20);pid=read.recv();fd=os.pidfd_open(pid);poll=select.poll();poll.register(fd,select.POLLIN)
        p.kill();p.join(timeout=5);assert poll.poll(10000),'Old scientific worker survived coordinator death'
    finally:
        if p.is_alive():p.kill();p.join()
        if fd is not None:os.close(fd)
        read.close();write.close()

def test_explicit_zero_core_budget_preserves_legacy_configuration():
    c=load_config('configs/lite-core-research-v1.yaml');assert c['window_search_seconds']==0 and c['ambiguity']['enabled'] is False
    old=load_config('configs/lite-v0.1.yaml');old['window_search_seconds']=0
    with pytest.raises(ValueError):validate_config(old)
    c['ambiguity']['enabled']=True
    with pytest.raises(ValueError):validate_config(c)

def test_quantum_safe_resume_identity_and_journal_corruption(tmp_path):
    cfg=config();ctx=GeometryContext(box(),cfg['numerical_policy']);p=np.array([.1,.2,.3]);save_geometry(tmp_path/'geometry.npz',ctx.bundle);p.astype('<f8').tofile(tmp_path/'driver.f64')
    stream_response(ctx,p,tmp_path,cfg,renderer=ctx.bind(p));w={'origin':[1,2,3],'side':64,'witnesses':[[1,2,3],[64,65,66]]}
    desc={'stage':str(tmp_path),'cfg':cfg,'geometry_id':ctx.geometry_id,'verifier':'silhouette','windows':[w]}
    a=verify_quantum(desc,1e-10);assert a['status']=='INCOMPLETE' and sum(a['done_counts'])==1
    path=tmp_path/'physical-work.private.json';state=json.loads(path.read_text());bad=copy.deepcopy(state);bad['key']['resolution']=1024;atomic_json(path,bad)
    with pytest.raises(ValueError,match='checksum'):verify_quantum(desc,1)
    atomic_json(path,state)
    result=verify_quantum(desc,10);assert result['status']=='COMPLETE'
    record=json.loads((tmp_path/'physical_evidence.private.json').read_text());evidence,_=load_evidence(tmp_path/'physical_evidence.private.json',record['key'],[Window((1,2,3),64)])
    assert len(evidence.covered)==8 and record['quantums']==2
    assert verify_quantum(desc,1)['cached_complete_evidence']

def test_isolated_worker_restart_reuses_only_certified_checkpoint(tmp_path):
    cfg=config();ctx=GeometryContext(box(),cfg['numerical_policy']);p=np.array([.1,.2,.3]);save_geometry(tmp_path/'geometry.npz',ctx.bundle);p.astype('<f8').tofile(tmp_path/'driver.f64')
    stream_response(ctx,p,tmp_path,cfg,renderer=ctx.bind(p));desc={'slot':0,'stage':str(tmp_path),'cfg':cfg,'geometry_id':ctx.geometry_id,'verifier':'silhouette',
            'windows':[{'origin':[1,2,3],'side':64,'witnesses':[[1,2,3],[64,65,66]]}]}
    context=mp.get_context('spawn');output=context.Queue(2);w=Worker(context,0,output)
    try:
        w.assign(desc,1e-10,hard=.001);w.process.join(timeout=20)
        assert w.process.exitcode==-signal.SIGALRM and not (tmp_path/'physical_evidence.private.json').exists()
        w.kill();w.start()
        w.assign(desc,1e-10)
        with pytest.raises(RuntimeError,match='collision'):w.assign(desc,1)
        number,slot,result,error=output.get(timeout=30);assert error is None and result['status']=='INCOMPLETE'
        w.kill();w.start();w.assign(desc,10);_,_,result,error=output.get(timeout=30)
        assert error is None and result['status']=='COMPLETE'
    finally:w.close();output.close();output.cancel_join_thread()

@pytest.mark.parametrize('family',FAMILIES)
def test_fast_mixture_declarative_caps_and_original_acceptance(family):
    b,r=generate(20261031,FAMILIES.index(family),family);ctx=GeometryContext(b)
    assert r['no_render_time_selection'] and len(r['objects'])<=8 and all(o['declared_planes']<=12 for o in r['objects'])
    assert geometry_acceptance(ctx,family,20261031,FAMILIES.index(family))['passed']
