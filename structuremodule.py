import torch
import torch.nn as nn
from dataclasses import dataclass
from einops import rearrange, repeat

from invariant_point_attention import IPABlock
import torch.nn.functional as F

def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def quaternion_raw_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Multiply two quaternions.
    Usual torch rules for broadcasting apply.

    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.

    Returns:
        The product of a and b, a tensor of quaternions shape (..., 4).
    """
    aw, ax, ay, az = torch.unbind(a, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((ow, ox, oy, oz), -1)

class StructureModule(nn.Module):
  def __init__(self, channel_size_msa, heads=8, scalar_key_dim=16, scalar_value_dim=16, point_key_dim=4, point_value_dim=4):
    super().__init__()
    self.ipa_block = IPABlock(
        dim = channel_size_msa,
        heads = heads,
        scalar_key_dim = scalar_key_dim,
        scalar_value_dim = scalar_value_dim,
    )
    self.out_proj = nn.Linear(channel_size_msa, 6)

  def forward(self, target_seq, pair_repr, translations, quaternions):
    rotations = quaternion_to_matrix(quaternions)
    single_repr = self.ipa_block(
        target_seq,
        pairwise_repr = pair_repr,
        rotations = rotations,
        translations = translations,
    )

    # update quaternion and translation
    updates = self.out_proj(single_repr)
    quaternion_update, translation_update = updates.chunk(2, dim=-1)
    quaternion_update = F.pad(quaternion_update, (1, 0), value = 1.)

    quaternions = quaternion_raw_multiply(quaternions, quaternion_update)
    translations = translations + torch.einsum('b n c, b n c r -> b n r', translation_update, rotations)

    return translations, quaternions

class PredictStructure(nn.Module):
  def __init__(self, channel_size_msa, depth=2):
    super().__init__()
    self.structure_module = StructureModule(channel_size_msa)
    self.depth = depth
    self.to_points = nn.Linear(channel_size_msa, 3)

  def forward(self, msa_repr, pair_repr):
    b, _, n, _ = msa_repr.shape
    translations = torch.randn(b, n, 3)
    quaternions = repeat(torch.tensor([1., 0., 0., 0.]), 'd -> b n d', b = b, n = n)
    single_repr = msa_repr[:,0,:,:]
    for i in range(self.depth):
      translations, quaternions = self.structure_module(single_repr, pair_repr, translations, quaternions)
    points_local = self.to_points(single_repr)
    rotations = quaternion_to_matrix(quaternions)
    coords = torch.einsum('b n c, b n c d -> b n d', points_local, rotations) + translations

    return coords
