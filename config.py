import torch
import torch.nn as nn
from dataclasses import dataclass
from einops import rearrange, repeat

@dataclass(frozen=True)
class Config:
  batch_size:int = 2
  sequence_size:int = 100
  sequence_num:int = 7
  channel_size_feat: int = 23
  channel_size_msa:int = 10
  channel_size_pair:int = 10

def get_inputs():
	config = Config()
	residue_index = repeat(torch.arange(config.sequence_size), 's -> b s', b=config.batch_size)
	msa_feat = torch.randn(
    config.batch_size,
    config.sequence_num,
    config.sequence_size,
    config.channel_size_feat
	)
	target_feat = torch.randn(
    config.batch_size,
    config.sequence_size,
    config.channel_size_feat
	)
	return config, residue_index, msa_feat, target_feat
