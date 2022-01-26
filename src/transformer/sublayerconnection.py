import torch.nn as nn
from .layernorm import LayerNorm

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, n_features, dropout):
        """
        :param n_features: the number of inputs of the layer being normalized
        :param dropout: the dropout probability
        """
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(n_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))
