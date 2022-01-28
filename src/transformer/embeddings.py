import torch.nn as nn
import math

class Embeddings(nn.Module):
    def __init__(self, n_features, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, n_features)
        self.n_features = n_features

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.n_features)
