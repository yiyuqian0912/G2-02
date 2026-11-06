import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from mask_only_data import MaskRecords
from decomposition_data import collate_decomposition_records, decomposed_batches
from gpu_mask_inputs import PackedDeviceBatches, enable_gpu_inputs


def fixture(root):
    rows=[];streams={'response':[], 'obstacle':[], 'masks':[]}
    offsets=dict.fromkeys(streams,0)
    rng=np.random.default_rng(7)
    for i,(side,count) in enumerate(((16,1),(16,3),(32,2),(16,1))):
        obstacle=np.zeros((side,side),dtype=bool);obstacle[:2]=True
        masks=rng.integers(0,2,(count,side,side)).astype(bool);masks[:,obstacle]=False
        strengths=.2+np.arange(count)*.15
        response=np.sum(masks*strengths[:,None,None],axis=0).astype('<f4')
        arrays={'response':response.ravel(), 'obstacle':np.packbits(obstacle.ravel(),bitorder='little'),
                'masks':np.packbits(masks.reshape(count,-1),axis=1,bitorder='little').ravel()}
        rows.append({'scene_id':10+i,'view_id':2*i,'size':side,'window':[i*32,64,side],
                     'intensities':strengths.tolist(),'response_offset':offsets['response'],
                     'obstacle_offset':offsets['obstacle'],'mask_offset':offsets['masks']})
        for key,value in arrays.items():streams[key].append(value);offsets[key]+=value.size
    for key,arrays in streams.items():
        np.concatenate(arrays).tofile(root/f'train_{key}.{"f32" if key=="response" else "u8"}')
    return MaskRecords(rows,root,'train',4)


class DeviceInputTests(unittest.TestCase):
    def test_resident_and_stream_match_original_dense_labels_bit_for_bit(self):
        with tempfile.TemporaryDirectory() as tmp:
            records=fixture(Path(tmp))
            for mode in ('resident','stream'):
                provider=PackedDeviceBatches(records,'cpu',mode=mode,chunk_mib=1)
                for indices in ([3,0,3],[1,3],[2]):
                    expected=collate_decomposition_records([records[i] for i in indices])
                    actual=provider.batch(indices)
                    self.assertEqual(set(expected),set(actual))
                    for key in actual:torch.testing.assert_close(actual[key],expected[key],rtol=0,atol=0)
                with self.assertRaisesRegex(ValueError,'one native'):provider.batch([0,2])
                self.assertFalse(provider.summary['cpu_bit_decoding'])

    def test_sampling_ddp_padding_and_order_unchanged_without_cpu_getitem(self):
        with tempfile.TemporaryDirectory() as tmp:
            records=fixture(Path(tmp))
            expected=list(decomposed_batches(records,2,np.random.default_rng(11),rank=1,world_size=2))
            provider=PackedDeviceBatches(records,'cpu',mode='resident')
            records.device_batch=provider.batch
            with patch.object(MaskRecords,'__getitem__',side_effect=AssertionError('CPU dense decoding must not run')):
                actual=list(decomposed_batches(records,2,np.random.default_rng(11),prefetch=3,rank=1,world_size=2))
            self.assertEqual(len(expected),len(actual))
            for a,b in zip(expected,actual):
                for key in a:torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)

    def test_cpu_training_is_unmodified_and_auto_mode_streams_without_cuda(self):
        with tempfile.TemporaryDirectory() as tmp:
            records=fixture(Path(tmp))
            self.assertIsNone(enable_gpu_inputs(records,'cpu',{'gpu_inputs':'auto'}))
            self.assertFalse(hasattr(records,'device_batch'))
            provider=PackedDeviceBatches(records,'cpu',mode='auto')
            self.assertEqual(provider.summary['mode'],'stream')

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA parity runs on the training server')
    def test_cuda_stream_and_resident_decode_exactly_match_cpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            records=fixture(Path(tmp))
            expected=collate_decomposition_records([records[i] for i in [3,1,0]])
            for mode in ('stream','resident'):
                provider=PackedDeviceBatches(records,'cuda',mode=mode)
                actual=provider.batch([3,1,0])
                for key in expected:torch.testing.assert_close(actual[key].cpu(),expected[key],rtol=0,atol=0)


if __name__=='__main__':unittest.main()
