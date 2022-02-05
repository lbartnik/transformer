import torch.nn as nn
from .encoderdecoder import encoderdecoder

def make_model(src_vocab, tgt_vocab, N=6, 
               n_features=512, d_ff=2048, h=8, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    model = encoderdecoder(src_vocab, tgt_vocab, n_features, h, N, dropout, d_ff)
    
    # This was important from their code. 
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model
