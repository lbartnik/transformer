import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, n_features, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(n_features, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)
