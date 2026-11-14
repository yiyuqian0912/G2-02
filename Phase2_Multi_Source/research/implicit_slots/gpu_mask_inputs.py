"""GPU mask decoding and native-size batches from the unchanged packed cache.

Resident mode uploads packed streams once. Stream mode transfers only responses
and packed bytes. Both decode bits and build dense label tensors on the device;
neither sends teacher answers to the model's forward interface.
"""
import math
from pathlib import Path
import numpy as np
import torch

GIB = 1024 ** 3


class PackedDeviceBatches:
    def __init__(self, records, device, *, mode='auto', fraction=.55,
                 reserve_gib=32., chunk_mib=64):
        if mode not in ('auto', 'resident', 'stream'):
            raise ValueError('GPU input mode must be auto, resident, stream or off')
        if not 0 < fraction < 1 or not math.isfinite(reserve_gib) or reserve_gib < 0:
            raise ValueError('invalid GPU cache memory policy')
        if type(chunk_mib) is not int or chunk_mib < 1:
            raise ValueError('GPU upload chunk must be a positive integer')
        self.entries, self.maps, self.max_slots = records.entries, records.maps, records.max_slots
        self.device, self.buffers = torch.device(device), {}
        self.shifts = torch.arange(8, device=self.device, dtype=torch.uint8)
        self.slots = torch.arange(records.max_slots, device=self.device)
        n, slots = len(records), records.max_slots
        offsets = np.empty((n, 3), dtype=np.int64)
        windows = np.empty((n, 3), dtype=np.float32)
        strengths = np.zeros((n, slots), dtype=np.float32)
        counts = np.empty(n, dtype=np.int64)
        identities = np.empty((n, 2), dtype=np.int64)
        for i, row in enumerate(records.entries):
            counts[i] = len(row['intensities'])
            offsets[i] = [row['response_offset'], row['obstacle_offset'], row['mask_offset']]
            windows[i] = row['window']
            strengths[i, :counts[i]] = row['intensities']
            identities[i] = [row['scene_id'], row['view_id']]
        self.meta = {name: torch.from_numpy(value).to(self.device) for name, value in
                     [('offsets', offsets), ('window', windows), ('strengths', strengths),
                      ('counts', counts), ('identities', identities)]}
        cache_bytes = sum(value.nbytes for value in records.maps.values())
        free_bytes, resident_limit = None, None
        if self.device.type == 'cuda':
            free_bytes, _ = torch.cuda.mem_get_info(self.device)
            resident_limit = max(0, min(int(free_bytes * fraction), int(free_bytes - reserve_gib * GIB)))
        use_resident = mode == 'resident' or (mode == 'auto' and resident_limit is not None and cache_bytes <= resident_limit)
        reason = 'requested stream mode' if mode == 'stream' else 'cache exceeds available resident budget'
        if use_resident:
            try:
                for name, value in records.maps.items():
                    dtype = torch.float32 if name == 'response' else torch.uint8
                    target = torch.empty(len(value), device=self.device, dtype=dtype)
                    self.buffers[name] = target
                    elements = max(1, chunk_mib * 1024 ** 2 // value.dtype.itemsize)
                    for start in range(0, len(value), elements):
                        # This bounded host copy is file I/O, not CPU bit decoding.
                        host = torch.from_numpy(np.array(value[start:start + elements], copy=True))
                        target[start:start + len(host)].copy_(host)
                reason = 'packed cache uploaded once'
            except torch.cuda.OutOfMemoryError:
                self.buffers.clear()
                if 'target' in locals():
                    del target
                torch.cuda.empty_cache()
                if mode == 'resident':
                    raise
                reason = 'resident upload OOM; using packed stream without CPU decoding'
        self.resident = len(self.buffers) == len(records.maps)
        self.summary = {'mode': 'resident' if self.resident else 'stream',
                        'records': n, 'packed_cache_bytes': cache_bytes,
                        'free_bytes_before_upload': free_bytes,
                        'resident_budget_bytes': resident_limit,
                        'metadata_bytes': sum(v.numel() * v.element_size() for v in self.meta.values()),
                        'cpu_bit_decoding': False, 'reason': reason}

    def _transfer(self, value):
        tensor = torch.from_numpy(value)
        if self.device.type == 'cuda':
            tensor = tensor.pin_memory()
        return tensor.to(self.device, non_blocking=self.device.type == 'cuda')

    def _decode(self, packed, pixels):
        bits = ((packed[..., None] >> self.shifts) & 1).flatten(-2)
        return bits[..., :pixels].bool()

    def batch(self, indices):
        if not indices:
            raise ValueError('empty device batch')
        side = int(self.entries[indices[0]]['size'])
        if any(int(self.entries[i]['size']) != side for i in indices):
            raise ValueError('device batch must use one native window size')
        pixels, b = side * side, len(indices)
        nbytes = (pixels + 7) // 8
        ids = torch.tensor(indices, dtype=torch.long, device=self.device)
        counts = self.meta['counts'][ids]
        active = self.slots[None] < counts[:, None]
        if self.resident:
            offsets = self.meta['offsets'][ids]
            pixel_axis = torch.arange(pixels, device=self.device)
            byte_axis = torch.arange(nbytes, device=self.device)
            response = self.buffers['response'][offsets[:, 0, None] + pixel_axis]
            obstacle = self.buffers['obstacle'][offsets[:, 1, None] + byte_axis]
            positions = offsets[:, 2, None, None] + self.slots[None, :, None] * nbytes + byte_axis
            # Inactive padded slots can extend beyond the final record. They
            # never contribute labels, and must not issue out-of-bounds gathers.
            positions = positions.clamp_max(len(self.buffers['masks']) - 1)
            masks = self.buffers['masks'][positions]
        else:
            response = np.empty((b, pixels), dtype=np.float32)
            obstacle = np.empty((b, nbytes), dtype=np.uint8)
            masks = np.zeros((b, self.max_slots, nbytes), dtype=np.uint8)
            for j, i in enumerate(indices):
                row = self.entries[i]
                count = len(row['intensities'])
                start = row['response_offset'];response[j] = self.maps['response'][start:start + pixels]
                start = row['obstacle_offset'];obstacle[j] = self.maps['obstacle'][start:start + nbytes]
                start = row['mask_offset'];masks[j, :count] = self.maps['masks'][start:start + count * nbytes].reshape(count, nbytes)
            response, obstacle, masks = map(self._transfer, (response, obstacle, masks))
        obstacle = self._decode(obstacle, pixels).reshape(b, side, side)
        masks = self._decode(masks, pixels).reshape(b, self.max_slots, side, side)
        masks &= active[:, :, None, None]
        identity = self.meta['identities'][ids]
        return {'response': response.reshape(b, side, side), 'obstacle': obstacle,
                'window': self.meta['window'][ids], 'teacher_mask': masks,
                'teacher_intensity': self.meta['strengths'][ids], 'layer_mask': active,
                'count_target': counts, 'scene_id': identity[:, 0], 'view_id': identity[:, 1]}


def enable_gpu_inputs(records, device, settings):
    """Attach without changing records, cache bytes, ordering or sampling."""
    mode = settings.get('gpu_inputs', 'off')
    if mode == 'off' or torch.device(device).type != 'cuda':
        return None
    if getattr(records, 'device_batch', None) is not None:
        return records.device_batch.__self__.summary
    provider = PackedDeviceBatches(records, device, mode=mode,
                                  fraction=settings.get('gpu_cache_fraction', .55),
                                  reserve_gib=settings.get('gpu_cache_reserve_gib', 32.),
                                  chunk_mib=settings.get('gpu_upload_chunk_mib', 64))
    # A few real cached windows verify endian, offsets and padded slots on the
    # actual server GPU before any optimizer update. No bulk CPU decoding.
    selected = {}
    for i, row in enumerate(records.entries):
        selected.setdefault(int(row['size']), i)
    indices = list(dict.fromkeys([*selected.values(), len(records)-1]))
    for i in indices:
        actual = provider.batch([i])
        expected = records[i]
        for key, value in expected.items():
            reference = torch.as_tensor(value, device=device)
            if not torch.equal(actual[key][0], reference):
                raise ValueError(f'GPU packed cache parity failed at record {i}: {key}')
    provider.summary['real_window_parity_checks'] = len(indices)
    records.device_batch = provider.batch
    return provider.summary
