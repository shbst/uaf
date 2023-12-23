import torch
import torch.nn as nn
from dataclasses import dataclass
from einops import rearrange, repeat

class InputEmbedder(nn.Module):
  """
  Algorithm3: Embeddings for initial representations
  """
  def __init__(self, channel_size_feat, channel_size_msa, channel_size_pair, sequence_size, sequence_num):
    super().__init__()
    self.to_a = nn.Linear(channel_size_feat, channel_size_pair)
    self.to_b = nn.Linear(channel_size_feat, channel_size_pair)
    self.to_m1 = nn.Linear(channel_size_feat, channel_size_msa)
    self.to_m2 = nn.Linear(channel_size_feat, channel_size_msa)
    self.pos_proj = nn.Linear(65, channel_size_pair)
    self.sequence_size = sequence_size
    self.sequence_num = sequence_num

  def forward(self, target_feat, residue_index, msa_feat):
    a = self.to_a(target_feat)
    b = self.to_b(target_feat)
    a = repeat(a, 'b s c -> b r s c', r=self.sequence_size)
    b = repeat(b, 'b s c -> b s r c', r=self.sequence_size)
    z = a + b
    z += self.relpos(residue_index)

    target_feat = repeat(target_feat, 'b r c -> b n r c', n=self.sequence_num)
    m = self.to_m1(msa_feat) + self.to_m2(target_feat)
    return m, z

  def relpos(self, residue_index, vbins=(torch.arange(65)-32)):
    """
    Algorithm4: Relative position enccoding
    """
    d_left = repeat(residue_index, 'b r -> b i r', i=self.sequence_size)
    d_right = repeat(residue_index, 'b r -> b r i', i=self.sequence_size)
    d = d_left - d_right
    p = self.one_hot(d, vbins)
    p = p.to(torch.float32)
    p = self.pos_proj(p)
    return p

  def one_hot(self, x, vbins):
    """
    Algorithm5: One-hot ecoding with nearest bin
    """
    bin_size = vbins.shape[0]
    b, r1, r2 = x.shape
    x = repeat(x, 'b r1 r2 -> b r1 r2 d', d=bin_size)
    vbins = repeat(vbins, 'd -> b r1 r2 d', b=b, r1=r1, r2=r2)
    index = torch.argmin(torch.abs(x - vbins), dim=-1)
    index = index.flatten()

    p = torch.zeros_like(x, dtype=x.dtype)
    p = rearrange(p, 'b r1 r2 v -> (b r1 r2) v')
    p[index] = 1
    p = rearrange(p, '(b r1 r2) v -> b r1 r2 v', r1=self.sequence_size, r2=self.sequence_size)
    return p
