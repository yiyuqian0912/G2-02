import copy,io,json,math,os,subprocess,sys,tempfile,unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch
import torch
from gpu_runtime import select_gpus,batch_candidates,memory_budget,choose_batch,_measure,GIB
from train_decomposition import load_config,semantic_config,make_model
from test_disentangled import config as tiny_config
from test_boundary_refined import example

ROOT=Path(__file__).resolve().parents[3]

# The attached H20 snapshot, retaining actual occupied VRAM. IDs are data,
# not a whitelist in the selector; permutation tests verify discovery.
def snapshot():
    used=[3020,69307,75199,71603,44886,59397,73404,83184]
    util=[0,99,98,96,0,88,100,100]
    return '\n'.join(f'{i}, GPU-{i}, 97871, {u}, {97871-u}, {v}' for i,(u,v) in enumerate(zip(used,util)))


class GPURuntimeTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_adaptive_uses_all_suitable_cards_despite_retained_processes(self):
        with patch('gpu_runtime.subprocess.check_output',return_value=snapshot()) as query,redirect_stderr(io.StringIO()) as log:
            self.assertEqual(select_gpus(samples=3,interval=0),['0','4'])
        self.assertEqual(query.call_count,3)
        self.assertIn('min-free=51.7 GiB',log.getvalue())
        self.assertNotIn('query-compute-apps',' '.join(str(c) for c in query.call_args_list))
        # The same metrics on different IDs select those IDs instead.
        changed=snapshot().replace('0, GPU-0,','8, GPU-8,').replace('4, GPU-4,','9, GPU-9,')
        with patch('gpu_runtime.subprocess.check_output',return_value=changed),redirect_stderr(io.StringIO()):
            self.assertEqual(select_gpus(samples=1),['8','9'])

    def test_recent_activity_or_free_memory_dips_are_excluded(self):
        quiet=snapshot();busy=quiet.replace('4, GPU-4, 97871, 44886, 52985, 0','4, GPU-4, 97871, 44886, 52985, 90')
        with patch('gpu_runtime.subprocess.check_output',side_effect=[quiet,busy,quiet]),redirect_stderr(io.StringIO()):
            self.assertEqual(select_gpus(samples=3,interval=0),['0'])
        low=quiet.replace('4, GPU-4, 97871, 44886, 52985, 0','4, GPU-4, 97871, 77871, 20000, 0')
        with patch('gpu_runtime.subprocess.check_output',side_effect=[quiet,low,quiet]),redirect_stderr(io.StringIO()):
            self.assertEqual(select_gpus(samples=3,interval=0),['0'])

    def test_strict_idle_retained_and_explicit_ids_validated(self):
        quiet='0, GPU-0, 97871, 1000, 96871, 0\n4, GPU-4, 97871, 1000, 96871, 0'
        with patch('gpu_runtime.subprocess.check_output',side_effect=[quiet,'GPU-4, 19\n']),redirect_stderr(io.StringIO()):
            self.assertEqual(select_gpus(policy='idle',samples=1),['0'])
        with patch('gpu_runtime.subprocess.check_output',return_value=snapshot()),redirect_stderr(io.StringIO()):
            self.assertEqual(select_gpus(ids='4,0',samples=1),['4','0'])
            with self.assertRaisesRegex(RuntimeError,'not found'):select_gpus(ids='9',samples=1)
            with self.assertRaisesRegex(RuntimeError,'No eligible'):select_gpus(min_free_mib=97871,samples=1)
        with self.assertRaisesRegex(ValueError,'duplicate'):select_gpus(ids='0,00')

    def test_candidates_keep_the_same_effective_global_batch(self):
        for world in (1,2,3,4,5,6,7,8):
            candidates=batch_candidates(32,world,256)
            target=32*world*math.ceil(256/(32*world))
            self.assertTrue(candidates)
            for b in candidates:self.assertEqual(b*world*math.ceil(256/(b*world)),target)
        self.assertEqual(batch_candidates(32,2,256),[8,16,32,64,128])
        self.assertEqual(batch_candidates(32,8,256),[8,16,32])

    def test_memory_budget_counts_only_free_and_our_own_allocations(self):
        budget=memory_budget(GIB,GIB,52*GIB,96*GIB)
        self.assertEqual(budget,47*GIB)
        self.assertLess(budget,.9*96*GIB)
        self.assertEqual(memory_budget(GIB,GIB,2*GIB,96*GIB),0)

    def test_common_batch_rejects_one_rank_oom_and_uses_throughput(self):
        results=[{'batch':32,'ranks':[{'feasible':True,'step_seconds':.11}]*2},
                 {'batch':64,'ranks':[{'feasible':True,'step_seconds':.14},{'feasible':True,'step_seconds':.15}]},
                 {'batch':128,'ranks':[{'feasible':True,'step_seconds':.18},{'feasible':False,'out_of_memory':True}]}]
        self.assertEqual(choose_batch(results,2),64)
        results[1]['ranks'][1]['step_seconds']=.3
        self.assertEqual(choose_batch(results,2),32)
        with self.assertRaisesRegex(RuntimeError,'No common'):choose_batch(results[-1:],2)

    def test_no_update_measurement_runs_actual_loss_backward_without_changing_weights(self):
        c=tiny_config();model=make_model(c,1024,torch.device('cpu'),False);b=example()
        before={k:v.clone() for k,v in model.state_dict().items()}
        with patch('torch.cuda.synchronize'):
            measured=_measure(model,b,c['loss'],torch.device('cpu'),False,1)
        self.assertGreater(measured,0)
        for key,value in before.items():torch.testing.assert_close(model.state_dict()[key],value)
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_runtime_batch_override_does_not_invalidate_checkpoint_signature(self):
        path=ROOT/'Phase2_Multi_Source/configs/masks_disentangled.json'
        original=load_config(path)
        with patch.dict(os.environ,{'RIND_BATCH_SIZE':'64'}):changed=load_config(path)
        self.assertEqual(changed['training']['batch_size'],64)
        self.assertEqual(semantic_config(original),semantic_config(changed))
        with patch.dict(os.environ,{'RIND_BATCH_SIZE':'0'}):
            with self.assertRaisesRegex(ValueError,'RIND_BATCH_SIZE'):load_config(path)

    def test_calibration_control_flow_oom_consensus_rng_and_weight_preservation(self):
        from contextlib import ExitStack
        from gpu_runtime import calibrate_batch
        class FakeCUDADevice:type='cuda'
        device=FakeCUDADevice();original_to=torch.Tensor.to
        class Records:
            entries=[{'size':16,'intensities':[.3,.5]}]
            def __len__(self):return 1
            def __getitem__(self,i):
                import numpy as np
                b=example()
                return {k:v[0].numpy() for k,v in b.items()}
        for world in (1,2):
            sequences=[]
            for rank in range(world):
                c=tiny_config();c['training'].update(batch_size=32,target_global_batch_size=256)
                model=make_model(c,1024,torch.device('cpu'),False).eval()
                before={k:v.clone() for k,v in model.state_dict().items()};rng=torch.get_rng_state().clone()
                current={'peak':0};sequence=[]
                def measured(model,batch,*args,**kwargs):
                    micro=len(batch['response']);sequence.append(micro);torch.rand(2)
                    current['peak']=int(micro/4*GIB)
                    if micro>=128:raise torch.cuda.OutOfMemoryError('mocked device OOM')
                    return .1+micro/512
                def to(tensor,*args,**kwargs):
                    if args and args[0] is device:return original_to(tensor,'cpu')
                    return original_to(tensor,*args,**kwargs)
                def gather(output,local):
                    output[:]=[dict(local,rank=r) for r in range(world)]
                    if local.get('feasible') and local.get('peak_reserved_bytes',0)>=16*GIB:
                        output[1]={'rank':1,'feasible':False,'out_of_memory':True}
                with tempfile.TemporaryDirectory() as tmp,ExitStack() as stack:
                    for name,value in [('synchronize',lambda *a:None),('empty_cache',lambda:None),
                        ('mem_get_info',lambda d:(52*GIB,96*GIB)),('memory_allocated',lambda d:GIB),
                        ('memory_reserved',lambda d:GIB),('reset_peak_memory_stats',lambda d:None),
                        ('max_memory_reserved',lambda d:current['peak']),
                        ('get_device_name',lambda d:'mocked H20'),('get_rng_state',lambda d:torch.tensor([1],dtype=torch.uint8)),('set_rng_state',lambda *a:None)]:
                        stack.enter_context(patch('torch.cuda.'+name,side_effect=value))
                    stack.enter_context(patch.object(torch.Tensor,'to',to))
                    stack.enter_context(patch('gpu_runtime._measure',side_effect=measured))
                    stack.enter_context(patch('torch.distributed.all_gather_object',side_effect=gather))
                    selected=calibrate_batch(model,Records(),c,device,rank,world,Path(tmp),0)
                    self.assertEqual(selected,64 if world==1 else 32)
                    self.assertEqual(c['training']['batch_size'],selected)
                    if rank==0:
                        report=json.loads((Path(tmp)/'gpu_batch_probe.json').read_text())
                        self.assertEqual(report['effective_global_batch'],256)
                        self.assertFalse(report['weights_updated'])
                self.assertFalse(model.training)
                self.assertTrue(torch.equal(rng,torch.get_rng_state()))
                for key,value in before.items():torch.testing.assert_close(model.state_dict()[key],value)
                sequences.append(sequence)
            if world==2:self.assertEqual(sequences[0],sequences[1])

    def test_resume_launcher_reuses_expanded_config_without_requiring_N_again(self):
        import shlex
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp);binary=directory/'bin';binary.mkdir();out=directory/'expanded'
            config=directory/'expanded.json';config.write_text(json.dumps({'output_root':str(out),
                'training_expansion':{'mode':'append','additional_complex_windows':20000}}))
            pointer=directory/'pointer';pointer.write_text(str(config))
            frontend=directory/'frontend.sh'
            text=(ROOT/'run_disentangled_masks_tmux.sh').read_text()
            text=text.replace('ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"',f'ROOT={shlex.quote(str(ROOT))}')
            text=text.replace('pointer="$ROOT/Phase2_Multi_Source/outputs/disentangled_v5_launcher/$session.txt"',
                              f'pointer={shlex.quote(str(pointer))}')
            frontend.write_text(text)
            fake=binary/'tmux';fake.write_text('#!/usr/bin/env bash\nif [[ "$1" == has-session ]];then exit 1;fi\nexit 0\n');fake.chmod(0o755)
            env={**os.environ,'PATH':str(binary)+os.pathsep+os.environ['PATH'],'RIND_PYTHON':sys.executable}
            env.pop('RIND_MASKS_CONFIG',None);env.pop('RIND_TMUX_SESSION',None)
            result=subprocess.run(['bash',str(frontend),'start','train','--session','rind_expanded',
                                   '--gpus','auto','--auto-batch'],env=env,capture_output=True,text=True,check=True)
            self.assertIn('Reusing previous session config',result.stderr)
            run=Path((out/'tmux_runs/latest_run.txt').read_text().strip())
            command=shlex.split((run/'command.txt').read_text())
            self.assertIn('RIND_MASKS_CONFIG='+str(config.resolve()),command)
            self.assertIn('RIND_TMUX_TASK=train',command)
            self.assertNotIn('--gpu-ids',command)

    def test_shell_auto_discovery_dispatches_all_eligible_cards_and_autobatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp);binary=directory/'bin';binary.mkdir();capture=directory/'dispatch'
            c=load_config(ROOT/'Phase2_Multi_Source/configs/masks_disentangled.json')
            file=directory/'config.json';file.write_text(json.dumps(c))
            smi=binary/'nvidia-smi';smi.write_text('#!/usr/bin/env python3\nprint('+repr(snapshot())+')\n');smi.chmod(0o755)
            python=binary/'python-shim'
            python.write_text('''#!/usr/bin/env bash
if [[ "$1" == -c || "$1" == research/implicit_slots/gpu_runtime.py ]]; then exec "$REAL_PYTHON" "$@";fi
printf '%s\\n' "$CUDA_VISIBLE_DEVICES" "$RIND_AUTO_BATCH" "${RIND_BATCH_SIZE:-}" "$@" > "$GPU_DISPATCH"
''');python.chmod(0o755)
            env={**os.environ,'PATH':str(binary)+os.pathsep+os.environ['PATH'],'REAL_PYTHON':sys.executable,
                 'RIND_PYTHON':str(python),'RIND_MASKS_CONFIG':str(file),'GPU_DISPATCH':str(capture)}
            result=subprocess.run(['bash',str(ROOT/'run_masks.sh'),'train','--gpus','auto','--gpu-samples','1'],
                                  env=env,capture_output=True,text=True,check=True)
            lines=capture.read_text().splitlines();self.assertEqual(lines[:2],['0,4','1'])
            self.assertIn('--nproc-per-node=2',lines);self.assertIn('policy=adaptive',result.stdout)
            result=subprocess.run(['bash',str(ROOT/'run_masks.sh'),'train','--gpus','auto','--gpu-samples','1',
                                   '--batch-size','64'],env=env,capture_output=True,text=True,check=True)
            self.assertEqual(capture.read_text().splitlines()[:3],['0,4','0','64'])

if __name__=='__main__':unittest.main()
