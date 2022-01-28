import torch.nn as nn

from .clones import clones
from .attention import attention

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, n_features, dropout=0.1):
        """
        Take in model size and number of heads.

        :param h: number of heads
        :param n_features: number of dimensions in the word-embedding space;
                           originally named d_model
        """
        super(MultiHeadedAttention, self).__init__()
        assert n_features % h == 0
        # We assume d_v always equals d_k; d_v is the number of columns in the
        # output tensor. d_v does not have to be equal to n_features since there
        # is one additional linear layer before the final output of the multi-headed
        # attention is passed to the subsequent stage.
        self.d_k = n_features // h
        self.h = h
        # technically, this should be `h` (the number of heads) batches of:
        #   - query: Linear(n_features, d_k),
        #   - key:   Linear(n_features, d_k),
        #   - value: Linear(n_features, d_v)
        #   - mapping to output: Linear(d_v, n_features)
        #
        # However, all heads are packed into a single large matrix which improves the
        # efficiency and shortens the code. Also, since d_k == d_v, all 4 matrices are
        # of the same shape.
        self.linears = clones(nn.Linear(n_features, n_features), 4)
        # remembers attention weights for visualization purposes
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        """
        Implements Figure 2

        :param query: Tensor, batch_size x phrase_length x n_features
        :param key:   Tensor, batch_size x phrase_length x n_features
        :param value: Tensor, batch_size x phrase_length x n_features
        :param mask:  Tensor, phrase_length x phrase_length; where mask == 0, weights will
                      be zeroed-out before applying softmax
        """
        # MultiHeadedAttention assumes batched processing in how it juggles
        # matrix dimensions, hence, each input must be a 3D Tensor: minibach x phrase x word embedding
        assert 3 == query.dim()
        assert 3 == key.dim()
        assert 3 == value.dim()

        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        #
        # why this sequence (view + transpose) instead of a single view()?
        # see: https://stats.stackexchange.com/questions/562150/why-the-sequence-of-matrix-transformations-in-the-annotated-transformer
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)
