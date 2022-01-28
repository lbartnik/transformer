import torch.nn as nn
import torch.nn.functional as F

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, n_features, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(n_features, d_ff)
        self.w_2 = nn.Linear(d_ff, n_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
