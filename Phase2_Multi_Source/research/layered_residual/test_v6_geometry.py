import json
from pathlib import Path
import tempfile
import unittest

import torch

from rind_phase1.model import SourceEnergyField
from train_layers import _geometry_parameter,_load_blind_initialization,read_config


CONFIG=Path(__file__).resolve().parents[2]/'configs'/'layer_student_geometry_residual.json'


class V6GeometryTests(unittest.TestCase):
    def write_config(self,mutate):
        config=json.loads(CONFIG.read_text())
        mutate(config['training'])
        temporary=tempfile.NamedTemporaryFile('w',suffix='.json',delete=False)
        json.dump(config,temporary);temporary.close()
        self.addCleanup(Path(temporary.name).unlink,missing_ok=True)
        return Path(temporary.name)

    def test_v6_configuration_is_valid(self):
        config=read_config(CONFIG)
        self.assertEqual(config['training']['geometry_fusion'],'residual')
        self.assertEqual(config['training']['geometry_cross_attention_heads'],4)

    def test_invalid_geometry_cross_attention_heads_fail_early(self):
        path=self.write_config(lambda training:training.__setitem__(
            'geometry_cross_attention_heads',3))
        with self.assertRaisesRegex(ValueError,'head counts must divide hidden'):
            read_config(path)

    def test_residual_geometry_requires_blind_initialization(self):
        path=self.write_config(lambda training:training.pop('initial_checkpoint'))
        with self.assertRaisesRegex(ValueError,'initial blind checkpoint'):
            read_config(path)

    def test_warm_start_is_exact_and_geometry_only_step_preserves_base(self):
        torch.manual_seed(9)
        blind=SourceEnergyField(hidden=16,fourier_frequencies=2)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint=Path(directory)/'best.pt'
            torch.save({'model':blind.state_dict(),'epoch':4},checkpoint)
            geometry=SourceEnergyField(
                hidden=16,fourier_frequencies=2,use_scene_geometry=True,
                geometry_parameter_width=8,geometry_attention_heads=4,
                geometry_cross_attention_heads=4,geometry_layers=1,
                geometry_fusion='residual')
            _load_blind_initialization(geometry,checkpoint)
        base_before={name:value.detach().clone() for name,value in geometry.state_dict().items()
                     if not _geometry_parameter(name)}
        response=torch.rand(2,16,16)
        obstacle=torch.zeros(2,16,16,dtype=torch.bool)
        window=torch.tensor([[0.,0.,16.],[32.,48.,16.]])
        candidate=torch.rand(2,7,2)*1024
        types=torch.tensor([[0,2],[1,3]])
        params=torch.rand(2,2,8)*100
        mask=torch.ones(2,2,dtype=torch.bool)
        blind.eval();geometry.eval()
        with torch.no_grad():
            expected=blind(response,obstacle,window,candidate)
            actual=geometry(response,obstacle,window,candidate,types,params,mask)
        torch.testing.assert_close(actual,expected)
        for name,parameter in geometry.named_parameters():
            parameter.requires_grad_(_geometry_parameter(name))
        optimizer=torch.optim.AdamW(
            [parameter for parameter in geometry.parameters() if parameter.requires_grad],lr=1e-2)
        geometry.train()
        energy=geometry(response,obstacle,window,candidate,types,params,mask)
        loss=(energy-torch.linspace(-1,1,7)[None]).square().mean()
        optimizer.zero_grad();loss.backward();optimizer.step()
        geometry.eval()
        with torch.no_grad():
            changed=geometry(response,obstacle,window,candidate,types,params,mask)
        self.assertFalse(torch.equal(changed,expected))
        for name,value in geometry.state_dict().items():
            if name in base_before:
                torch.testing.assert_close(value,base_before[name])


if __name__=='__main__':
    unittest.main()
