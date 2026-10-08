"""Deterministic synthetic teachers; these do not simulate RIND physics."""
import torch


def make_mock_records(count=12, candidate_count=64, image_size=16, seed=0, prefix='mock', random_targets=False):
    if count < 1 or candidate_count < 1 or image_size < 4:
        raise ValueError('positive count/candidates and image_size >= 4 required')
    rng = torch.Generator().manual_seed(seed)
    side = int(candidate_count ** 0.5)
    if side * side == candidate_count:
        y, x = torch.meshgrid(torch.linspace(-1, 2, side), torch.linspace(-1, 2, side), indexing='ij')
        xy = torch.stack((x.flatten(), y.flatten()), -1)
    else:
        xy = torch.rand(candidate_count, 2, generator=rng) * 3 - 1
    iy, ix = torch.meshgrid(torch.linspace(0, 1, image_size), torch.linspace(0, 1, image_size), indexing='ij')
    records = []
    for i in range(count):
        family = ('one_peak', 'two_peaks', 'ridge')[i % 3]
        offset = 0.04 * ((i // 3) % 4 - 1.5)
        # All target centers and ridges are outside the unit observation window.
        left = ((xy - xy.new_tensor([-0.6 + offset, 0.5])) ** 2).sum(-1) / (2 * 0.28 ** 2)
        right = ((xy - xy.new_tensor([1.6 + offset, 0.5])) ** 2).sum(-1) / (2 * 0.28 ** 2)
        if family == 'one_peak':
            log_q, response = -left, (ix > 0.5).float()
        elif family == 'two_peaks':
            log_q, response = torch.logaddexp(-left, -right), (iy > 0.5).float()
        else:
            log_q = -((xy[:, 0] + 0.6 - offset) / 0.22) ** 2 / 2 - ((xy[:, 1] - 0.5) / 1.2) ** 2 / 2
            response = (ix + iy > 1).float()
        # Distinguishable intensities encode the four small mock target shifts.
        # This is a fitting fixture, not a physical response generator.
        response = response * (0.8 + offset)
        if random_targets:
            family = 'random'
            response = torch.rand(image_size, image_size, generator=rng)
            log_q = -torch.rand(candidate_count, generator=rng) / 0.1
        records.append({'response': response[None], 'candidate_xy': xy.clone(),
                        'coordinate_space': 'normalized', 'physical_cost': -0.1 * log_q,
                        'teacher_prob': log_q.softmax(-1), 'area_weight': torch.ones(candidate_count),
                        'scene_id': f'{prefix}-scene-{i}', 'view_id': '0', 'family': family})
    return records
