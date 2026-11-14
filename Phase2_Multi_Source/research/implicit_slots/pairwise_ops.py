"""Bounded-memory pairwise mask reductions, independent of einsum planning."""
import torch


def masked_pairwise_dot(left, right, weight):
    """[B,T,P] x [B,S,P], weighted over P, returns [B,T,S].

    Weight one input BEFORE contraction. A three-input einsum can multiply
    the slot axes before reducing P when opt_einsum is absent, materializing
    [B,T,S,P]. This explicit bmm needs only [B,T,P] workspace and [B,T,S]
    output, and retains ordinary autograd for both inputs and the weights.
    """
    if (left.ndim!=3 or right.ndim!=3 or weight.ndim!=2 or
            left.shape[0]!=right.shape[0] or left.shape[0]!=weight.shape[0] or
            left.shape[2]!=right.shape[2] or left.shape[2]!=weight.shape[1]):
        raise ValueError('masked pairwise dot requires aligned [B,T,P], [B,S,P], [B,P]')
    return torch.bmm(left*weight[:,None],right.transpose(1,2))
