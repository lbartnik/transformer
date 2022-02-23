import torch.nn as nn
import math

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab_size, padding_idx):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.d_model = d_model

    def forward(self, x):
        # in "Attention Is All You Need" embeddings are multiplied my sqrt(d_model)
        return self.lut(x) * math.sqrt(self.d_model)
