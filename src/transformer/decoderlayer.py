import torch.nn as nn

from .clones import clones
from .sublayerconnection import SublayerConnection
from .multiheadedattention import MultiHeadedAttention
from .positionwisefeedforward import PositionwiseFeedForward



class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, n_features, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.n_features = n_features
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(n_features, dropout), 3)
 
    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)

def decoderlayer(n_features, h, d_ff, dropout):
    self_attn = MultiHeadedAttention(h, n_features)
    src_attn = MultiHeadedAttention(h, n_features)
    ff = PositionwiseFeedForward(n_features, d_ff, dropout)
    return DecoderLayer(n_features, self_attn, src_attn, ff, dropout)
