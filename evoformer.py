import torch
import torch.nn as nn
from dataclasses import dataclass
from einops import rearrange, repeat

class MSARowWiseAttentionWithPairBias(nn.Module):
  """
  Algorithm7: MSA row-wise gated self-attetion with pair bias
  """
  def __init__(self, sequence_num, sequence_size, channel_size_msa, channel_size_pair, hidden_channel=32, nhead=8):
    super().__init__()
    self.layernorm = nn.LayerNorm(sequence_size)
    self.to_q = nn.Linear(channel_size_msa, nhead*hidden_channel, bias=False)
    self.to_k = nn.Linear(channel_size_msa, nhead*hidden_channel, bias=False)
    self.to_v = nn.Linear(channel_size_msa, nhead*hidden_channel, bias=False)
    self.to_bias = nn.Linear(channel_size_pair, nhead)
    self.to_g = nn.Linear(channel_size_msa, nhead*hidden_channel)
    self.last_layer = nn.Linear(nhead*hidden_channel, channel_size_msa)

    self.nhead = nhead
    self.hidden_channel = hidden_channel
    self.sequence_size = sequence_size

    self.sigmoid = nn.Sigmoid()
    self.softmax = nn.Softmax(dim=3)
    self.scale = 1 / (hidden_channel ** 0.5)

  def forward(self, msa_repr, pair_repr):
    #Input Projecction
    x = rearrange(msa_repr, 'b s r c -> b c s r')
    x = self.layernorm(x)
    x = rearrange(x, 'b c s r -> b s r c')
    tbatch = x.shape[1]
    x = rearrange(x, 'b h w c -> (b h) w c') #here should be changed in column-wise attention
    q = self.to_q(x)
    k = self.to_k(x)
    v = self.to_v(x)
    q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.nhead), (q, k, v))

    bias = self.to_bias(pair_repr)
    bias = rearrange(bias, 'b i j h -> b h i j')
    bias = repeat(bias, 'b i j h -> (a b) i j h', a=tbatch)

    g = self.sigmoid(self.to_g(x))
    g = rearrange(g, 'b i (h d) -> b h i d', d = self.hidden_channel)

    #Attention
    dots = torch.einsum('b h i d, b h j d -> b h i j', q, k)
    attn = self.softmax(self.scale * dots + bias)
    out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
    out = g * out

    #Output projetion
    out = rearrange(out, '(b h) n w d -> b h w n d', h=tbatch)
    out = torch.concat([out[:,:,:,i,:] for i in range(out.shape[3])], dim=-1) #out.shape = (b,h,w,n*d)
    out = self.last_layer(out) #out.shape = (b,h,w,c)

    return out

class MSAColumnWiseAttention(nn.Module):
  """
  Algorithm8: MSA column-wise gated self-attention
  """
  def __init__(self, sequence_num, sequence_size, channel_size_msa, channel_size_pair, hidden_channel=32, nhead=8):
    super().__init__()
    self.layernorm = nn.LayerNorm(sequence_num)
    self.to_q = nn.Linear(channel_size_msa, nhead*hidden_channel, bias=False)
    self.to_k = nn.Linear(channel_size_msa, nhead*hidden_channel, bias=False)
    self.to_v = nn.Linear(channel_size_msa, nhead*hidden_channel, bias=False)
    self.to_bias = nn.Linear(channel_size_pair, nhead)
    self.to_g = nn.Linear(channel_size_msa, nhead*hidden_channel)
    self.last_layer = nn.Linear(nhead*hidden_channel, channel_size_msa)

    self.nhead = nhead
    self.hidden_channel = hidden_channel
    self.sequence_size = sequence_size

    self.sigmoid = nn.Sigmoid()
    self.softmax = nn.Softmax(dim=3)
    self.scale = 1 / (hidden_channel ** 0.5)

  def forward(self, msa_repr):
    #Input Projecction
    msa_repr = rearrange(msa_repr, 'b s r c -> b r s c')
    x = rearrange(msa_repr, 'b s r c -> b c s r')
    x = self.layernorm(x)
    x = rearrange(x, 'b c s r -> b s r c')
    tbatch = x.shape[1]
    x = rearrange(x, 'b h w c -> (b h) w c') #here should be changed in column-wise attention
    q = self.to_q(x)
    k = self.to_k(x)
    v = self.to_v(x)
    q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.nhead), (q, k, v))

    g = self.sigmoid(self.to_g(x))
    g = rearrange(g, 'b i (h d) -> b h i d', d = self.hidden_channel)

    #Attention
    dots = torch.einsum('b h i d, b h j d -> b h i j', q, k)
    attn = self.softmax(self.scale * dots)
    out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
    out = g * out

    #Output projetion
    out = rearrange(out, '(b h) n w d -> b h w n d', h=tbatch)
    out = torch.concat([out[:,:,:,i,:] for i in range(out.shape[3])], dim=-1) #out.shape = (b,h,w,n*d)
    out = self.last_layer(out) #out.shape = (b,h,w,c)
    out = rearrange(out, 'b h w c -> b w h c')

    return out

class Trasition(nn.Module):
  """
  Algorithm9: Transition layer in the MSA stack
  """
  def __init__(self, channel_size_msa, n=4):
    super().__init__()
    self.norm = nn.LayerNorm(channel_size_msa)
    self.linear1 = nn.Linear(channel_size_msa, n*channel_size_msa)
    self.relu = nn.ReLU()
    self.linear2 = nn.Linear(n*channel_size_msa, channel_size_msa)

  def forward(self, msa_repr):
    x = self.norm(msa_repr)
    x = self.linear1(x)
    x = self.relu(x)
    x = self.linear2(x)

    return x

class OuterProductMean(nn.Module):
  """
  Algorithm10: Outer product mean
  """
  def __init__(self, sequence_num, sequence_size, channel_size_msa, channel_size_pair, hidden_channel=32):
    super().__init__()
    self.norm = nn.LayerNorm(channel_size_msa)
    self.left_proj = nn.Linear(channel_size_msa, hidden_channel)
    self.right_proj = nn.Linear(channel_size_msa, hidden_channel)
    self.out_proj = nn.Linear(hidden_channel**2, channel_size_pair)

    self.sequence_size = sequence_size
  def forward(self, msa_repr):
    x = rearrange(msa_repr, 'b s r c -> (b r) s c')
    x = self.norm(x)
    left = self.left_proj(x) #left.shape = (b,s,d)
    right = self.right_proj(x)
    left = repeat(left, '(b r) s d -> b r s d', r=self.sequence_size)
    right = repeat(right, '(b r) s d -> b r s d', r=self.sequence_size)
    outer = rearrange(left, 'b r s d -> b r () s d') * rearrange(right, 'b r s d -> b () r s d')
    outer = rearrange(outer, 'b a c s d -> b a c s d ()') * rearrange(outer, 'b a c s d -> b a c s () d')
    outer = outer.mean(dim=3)
    outer = rearrange(outer, 'b i j d e -> b i j (d e)')
    z = self.out_proj(outer)
    return z

class TriangularMultiplicativeUpdate(nn.Module):
  """
  Algorithm 11: Triangular multiplicative update using 'outgoing' edges
  Algorithm 12: Triangular multiplicative update using 'incoming' edges
  """
  def __init__(self, channel_size_pair, hidden_channel=128, mode='outgoing'):
    super().__init__()
    self.norm = nn.LayerNorm(channel_size_pair)
    self.to_g1 = nn.Linear(channel_size_pair, hidden_channel)
    self.to_g2 = nn.Linear(channel_size_pair, hidden_channel)
    self.to_g3 = nn.Linear(channel_size_pair, channel_size_pair)
    self.left_proj = nn.Linear(channel_size_pair, hidden_channel)
    self.right_proj = nn.Linear(channel_size_pair, hidden_channel)
    self.norm2 = nn.LayerNorm(channel_size_pair)
    self.out_proj = nn.Linear(hidden_channel, channel_size_pair)
    self.sigmoid = nn.Sigmoid()

    assert mode in ['outgoing', 'incoming'], 'mode must be either "outgoing" or "incoming"'
    self.mode = mode

  def forward(self, pair_repr):
    z = self.norm(pair_repr)
    g1 = self.sigmoid(self.to_g1(z))
    g2 = self.sigmoid(self.to_g2(z))
    left = g1 * self.left_proj(z)
    right = g2 * self.right_proj(z)
    if self.mode == 'outgoing':
      x = torch.einsum('b i k d, b j k d -> b i j d', left, right)
    elif self.mode == 'incoming':
      x = torch.einsum('b k i d, b k j d -> b i j d', left, right)
    x = self.sigmoid(self.to_g3(z)) * self.norm2(self.out_proj(x))

    return x

class TriangularGatedSelfAttention(nn.Module):
  """
  Algorithm 13: Triangular gated self-attetion around starting node
  Algorithm 14: Triangular gated self-attetion around ending node
  """
  def __init__(self, channel_size_pair, sequence_size, hidden_channel=32, nhead=4, around='starting'):
    super().__init__()
    self.norm = nn.LayerNorm(channel_size_pair)
    self.to_q = nn.Linear(channel_size_pair, nhead*hidden_channel, bias=False)
    self.to_k = nn.Linear(channel_size_pair, nhead*hidden_channel, bias=False)
    self.to_v = nn.Linear(channel_size_pair, nhead*hidden_channel, bias=False)
    self.to_bias = nn.Linear(channel_size_pair, nhead, bias=False)
    self.to_g = nn.Linear(channel_size_pair, nhead*hidden_channel)
    self.out_proj = nn.Linear(nhead*hidden_channel, channel_size_pair)

    self.sigmoid = nn.Sigmoid()
    self.softmax = nn.Softmax(dim=4)

    assert around in ['starting', 'ending'], 'around must be either "starting" or "ending"'
    self.around = around
    self.nhead = nhead
    self.sequence_size = sequence_size
    self.scale = 1 / (hidden_channel**0.5)

  def forward(self, pair_repr):
    #Input Projections
    z = self.norm(pair_repr)
    q, k, v = self.to_q(z), self.to_k(z), self.to_v(z)
    q, k, v = map(lambda t: rearrange(t, 'b h w (n d) -> b n h w d', n = self.nhead), (q, k, v))
    bias = self.to_bias(z) #bias.shape = (b, h, w, n)
    bias = rearrange(bias, 'b h w n -> b n h w')
    if self.around=='starting':
      bias = repeat(bias, 'b n h w -> b n k h w', k=self.sequence_size)
    elif self.around=='ending':
      bias = repeat(bias, 'b n h w -> b n h k w', k=self.sequence_size)
    g = self.sigmoid(self.to_g(z))
    g = rearrange(g, 'b h w (n d) -> b n h w d', n=self.nhead)

    #Attetion
    if self.around == 'starting':
      dots = self.scale * torch.einsum('b n i j d, b n i k d -> b n i j k', q, k) + bias
    elif self.around == 'ending':
      dots = self.scale * torch.einsum('b n i j d, b n k j d -> b n i j k', q, k) + bias
    dots = self.softmax(dots)
    if self.around == 'starting':
      out = g * torch.einsum('b n i j k, b n i j d -> b n i j d', dots, v)
    elif self.around == 'ending':
      out = g * torch.einsum('b n i j k, b n k j d -> b n i j d', dots, v)
    out = rearrange(out, 'b n i j d -> b i j (n d)')

    #Outer Projection
    z = self.out_proj(out)

    return z

class EvoformerBlock(nn.Module):
  """
  Algorithm6: Evoformer stack
  """
  def __init__(self, sequence_num, sequence_size, channel_size_msa, channel_size_pair, depth=5):
    super().__init__()
    self.msa_row = MSARowWiseAttentionWithPairBias(
      sequence_num,
      sequence_size,
      channel_size_msa,
      channel_size_pair
      )
    self.msa_column = MSAColumnWiseAttention(
      sequence_num,
      sequence_size,
      channel_size_msa,
      channel_size_pair
    )
    self.trans_msa = Trasition(
      channel_size_msa,
    )
    self.outer_prod = OuterProductMean(
      sequence_num,
      sequence_size,
      channel_size_msa,
      channel_size_pair,
    )
    self.outgoing = TriangularMultiplicativeUpdate(
      channel_size_pair,
      mode='outgoing'
    )
    self.incoming = TriangularMultiplicativeUpdate(
      channel_size_pair,
      mode='incoming'
    )
    self.starting = TriangularGatedSelfAttention(
      channel_size_pair,
      sequence_size,
      around='starting'
    )
    self.ending = TriangularGatedSelfAttention(
      channel_size_pair,
      sequence_size,
      around='ending'
    )
    self.trans_pair = Trasition(
      channel_size_pair,
    )
    self.depth = depth
    self.dropout15 = nn.Dropout(p=0.15)
    self.dropout25 = nn.Dropout(p=0.25)
    self.to_s = nn.Linear(channel_size_msa, channel_size_msa)
  def forward(self, msa_repr, pair_repr):
    x = msa_repr
    z = pair_repr
    for i in range(self.depth):
      #MSA stack
      x += self.dropout15(self.msa_row(x, z))
      x += self.msa_column(x)
      x += self.trans_msa(x)

      #Communication
      z += self.outer_prod(x)

      #Pair stack
      z += self.dropout25(self.outgoing(z))
      z += self.dropout25(self.incoming(z))
      z += self.dropout25(self.starting(z))
      z += self.dropout25(self.ending(z))
      z += self.trans_pair(z)
    #Extract the sigle represetation
    s = self.to_s(x[:,0,:,:])
    return x, z, s

