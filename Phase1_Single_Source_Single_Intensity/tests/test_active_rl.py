import copy,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from rind_phase1.active_rl.core import TileEnv,Window,ActorCritic,potential,advantages
from rind_phase1.active_rl.locator import Belief,FrozenLocator,policy_inputs
from rind_phase1.active_rl.experiment import ppo_update
from rind_phase1.model import SourceEnergyField
from rind_phase1.sampling import grid_coordinates


class ActiveRLTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1);torch.manual_seed(9)

    def test_exact_native_pixel_union_budget_and_free_revisit(self):
        e=TileEnv(256,16,Window(64,64,32),4096)
        self.assertEqual(e.spent,1024);self.assertEqual(len(e.initial_tiles),4)
        old=e.window;new=e.step(2);self.assertEqual(len(new),2);self.assertEqual(e.spent,1536)
        e.step(1);self.assertEqual(e.window,old);self.assertEqual(e.spent,1536)
        rng=np.random.default_rng(5)
        while not e.done:
            _,legal=e.legal();e.step(int(rng.choice(np.flatnonzero(legal))))
            self.assertEqual(e.spent,int(e.coverage.sum())*256);self.assertLessEqual(e.spent,e.budget)

    def test_reward_prioritizes_truth_and_telescopes(self):
        xy=torch.tensor([[0.,0.],[10.,0.],[100.,100.]])
        distributions=[torch.tensor([.2,.1,.7]),torch.tensor([.4,.4,.2]),torch.tensor([.45,.45,.1])]
        scores=[potential(p,xy,[0,0])[0] for p in distributions]
        self.assertGreater(scores[1],scores[0]);self.assertAlmostEqual(float((scores[1]-scores[0])+(scores[2]-scores[1])),float(scores[2]-scores[0]))
        wrong=torch.tensor([.001,.001,.998]);self.assertLess(float(potential(wrong,xy,[0,0])[0]-scores[-1]),0)

    def test_fusion_is_order_independent_and_duplicate_idempotent(self):
        locator=SimpleNamespace(grid=8,world=128,tile=16,spacing=16,device=torch.device('cpu'),prior=torch.full((64,),-np.log(64)))
        answers={(0,i):(torch.randn(64).log_softmax(0),torch.rand(1,1)) for i in (0,1)}
        a=Belief(locator);b=Belief(locator)
        a.update(0,[0,1],answers);b.update(0,[1],answers);b.update(0,[0],answers)
        torch.testing.assert_close(a.probability,b.probability,rtol=0,atol=0)
        before=a.probability.clone();a.update(0,[0,1],answers)
        torch.testing.assert_close(a.probability,before,rtol=0,atol=0)

    def test_actor_features_cannot_read_source_truth(self):
        class Context:
            @property
            def truth(self):raise AssertionError('truth leaked')
        loc=SimpleNamespace(grid=8,world=128,tile=16,spacing=16,device=torch.device('cpu'),prior=torch.full((64,),-np.log(64)))
        c=Context();c.env=TileEnv(128,16,Window(32,32,32),4096);c.belief=Belief(loc)
        inputs=policy_inputs([c],loc);d,value=ActorCritic()(*inputs)
        self.assertEqual(d.probs.shape,(1,10));self.assertTrue(torch.isfinite(value).all())
        self.assertTrue((d.probs[~inputs[-1]]==0).all())

    def test_gae_does_not_cross_terminal_episode(self):
        rewards=torch.tensor([[1.],[2.],[3.]])
        result,returns=advantages(rewards,torch.zeros_like(rewards),torch.tensor([[False],[True],[False]]),torch.tensor([5.]),lam=1.)
        torch.testing.assert_close(result,torch.tensor([[3.],[2.],[8.]]))

    def test_bit_decoder_matches_acquired_pixels_and_hidden_changes_do_not_leak(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);model=SourceEnergyField(hidden=16,fourier_frequencies=2,coordinate_scale=128.,use_obstacle=False,pixel_coordinates=True,normalization_support='world')
            cp=root/'locator.pt';torch.save(dict(interface_version='phase1-v1',model_config=dict(hidden=16,fourier_frequencies=2,coordinate_scale=128.,use_obstacle=False,pixel_coordinates=True,normalization_support='world'),model=model.state_dict(),epoch=0),cp)
            scene=np.zeros((128,128),dtype=np.uint8);scene[:16,:8]=1
            def fake(image):return SimpleNamespace(manifest={'global_size':128},_base=SimpleNamespace(visibility_bits=np.packbits(image.ravel(),bitorder='little')[None,None]))
            first=FrozenLocator(fake(scene),cp,'cpu',tile=16,spacing=16)
            # Lower Fourier frequency is required for a sufficiently resolved 128-world grid.
            torch.testing.assert_close(first.pixels(0,[0])[0],torch.from_numpy(scene[:16,:16]).float())
            scene[96:,96:]=1
            second=FrozenLocator(fake(scene),cp,'cpu',tile=16,spacing=16)
            a=first.query([(0,0)]);b=second.query([(0,0)])
            torch.testing.assert_close(a[(0,0)][0],b[(0,0)][0],rtol=0,atol=0)
            self.assertTrue(all(not p.requires_grad and p.grad is None for p in first.model.parameters()))

    def test_ppo_updates_policy_without_a_phase1_optimizer(self):
        model=ActorCritic();b=4
        image=torch.rand(b,6,16,16);scalar=torch.rand(b,6);actions=torch.rand(b,10,7);legal=torch.ones(b,10,dtype=torch.bool)
        with torch.no_grad():d,v=model(image,scalar,actions,legal);a=d.sample();lp=d.log_prob(a)
        buffer=dict(image=image,scalar=scalar,actions=actions,legal=legal,action=a,logp=lp,value=v,
            advantage=torch.tensor([1.,-.5,.7,-.3]),returns=v+torch.tensor([1.,-.5,.7,-.3]))
        old=[p.clone() for p in model.parameters()]
        config=dict(ppo_epochs=2,minibatch_size=2,clip_ratio=.2,value_weight=.5,action_entropy_bonus=.01,grad_clip=.5,target_kl=.1)
        metrics=ppo_update(model,buffer,torch.optim.Adam(model.parameters(),lr=.0003),config,1)
        self.assertTrue(all(np.isfinite(v) for v in metrics.values()));self.assertTrue(any(not torch.equal(x,y) for x,y in zip(old,model.parameters())))

    def test_unknown_field_agent_needs_only_acquired_sensor_values(self):
        from rind_phase1.active_rl.deploy import UnknownFieldAgent
        from rind_phase1.active_rl.prepare import sha
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);mc=dict(hidden=16,fourier_frequencies=2,coordinate_scale=128.,use_obstacle=False,pixel_coordinates=True,normalization_support='world')
            m=SourceEnergyField(**mc);source=root/'phase1.pt'
            torch.save(dict(interface_version='phase1-v1',model_config=mc,model=m.state_dict(),epoch=0),source)
            c=dict(hidden=32,tile_size=16,spacing=16,candidate_chunk=64,locator_batch_size=4,max_steps=8,fusion_discount=.5)
            policy=ActorCritic(32);cp=root/'policy.pt'
            torch.save(dict(config=c,policy=policy.state_dict(),locator_snapshot_sha256=sha(source)),cp)
            agent=UnknownFieldAgent(cp,source,[32,32,32],4096)
            request=agent.request();self.assertIsNone(agent.locator.dataset)
            self.assertEqual(request['new_pixels'],1024)
            with self.assertRaises(ValueError):agent.observe({})
            inputs={tile:np.zeros((16,16),dtype=np.float32) for tile in request['tiles']}
            p=agent.observe(inputs);self.assertAlmostEqual(float(p.sum()),1.)
            self.assertEqual(len(agent.context.belief.seen),4)
            self.assertEqual(agent.locator.queries,0)  # uniform observation stays neutral

    def test_root_cpu_launcher_and_tmux_preserve_run_arguments(self):
        import os,shlex,subprocess,sys
        from rind_phase1.data import PROJECT_ROOT
        repo=PROJECT_ROOT.parent
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);capture=root/'argv';shim=root/'python'
            shim.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "$CAPTURE"\n');shim.chmod(0o755)
            env={**os.environ,'RIND_PYTHON':str(shim),'CAPTURE':str(capture),'MPLCONFIGDIR':str(root/'mpl'),'NUMBA_CACHE_DIR':str(root/'numba')}
            subprocess.run(['bash',str(repo/'run_single_source_rl.sh'),'evaluate','--device','cpu','--run-name','unit'],env=env,check=True,capture_output=True)
            arguments=capture.read_text().splitlines()
            self.assertEqual(arguments[:4],['-u','-m','rind_phase1.active_rl.experiment','evaluate'])
            self.assertEqual(arguments[-4:],['--device','cpu','--run-name','unit'])
            # Exercise persistent command construction without starting a job.
            mock=root/'tmux';mock.write_text('#!/usr/bin/env bash\nif [[ "$1" == has-session ]];then exit 1;fi\nexit 0\n');mock.chmod(0o755)
            frontend=root/'frontend.sh';text=(repo/'run_single_source_rl_tmux.sh').read_text()
            text=text.replace('ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"','ROOT='+shlex.quote(str(repo)))
            text=text.replace('"$ROOT/Phase1_Single_Source_Single_Intensity/outputs/active_rl/$run_name"',shlex.quote(str(root/'result')))
            frontend.write_text(text)
            env.update(PATH=str(root)+os.pathsep+env['PATH'],RIND_PYTHON=sys.executable)
            result=subprocess.run(['bash',str(frontend),'start','train','--run-name','unit','--device','cpu'],env=env,check=True,capture_output=True,text=True)
            directory=Path((root/'result/tmux_runs/latest_run.txt').read_text().strip())
            command=shlex.split((directory/'command.txt').read_text())
            self.assertIn('RIND_TMUX_TASK=train',command);self.assertIn('RIND_TMUX_RUNNER='+str(repo/'run_single_source_rl.sh'),command)
            self.assertEqual(command[-4:],['--run-name','unit','--device','cpu'])
            self.assertIn('rind_rl_unit',result.stdout)


if __name__=='__main__':unittest.main()
