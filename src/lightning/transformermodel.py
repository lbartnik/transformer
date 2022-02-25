import torch
import torch.nn as nn
import torch.nn.functional as F


from .embeddings import Embeddings
from .positionalencoding import PositionalEncoding



class TransformerModel(nn.Module):
    def __init__(self, src_vocab, trg_vocab, d_model=512, h=8, N=6, ff_dim=2048, padding_idx=0):
        super(TransformerModel, self).__init__()

        self.src_embed = nn.Sequential(Embeddings(d_model, src_vocab, padding_idx), PositionalEncoding(d_model))
        self.trg_embed = nn.Sequential(Embeddings(d_model, trg_vocab, padding_idx), PositionalEncoding(d_model))
        self.transformer = nn.Transformer(d_model=d_model, nhead=h, num_encoder_layers=N, num_decoder_layers=N,
                                          dim_feedforward=ff_dim, batch_first=True)
        self.generator = nn.Linear(d_model, trg_vocab)

        self.padding_idx = padding_idx

        # TODO differentiate
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src, trg):
        out = self.transformer(self.src_embed(src), self.trg_embed(trg),
                               tgt_mask=nn.Transformer.generate_square_subsequent_mask(trg.size(-1)),
                               src_key_padding_mask=self._padding_mask(src),
                               tgt_key_padding_mask=self._padding_mask(trg))
        probs = self.generator(out)
        return F.log_softmax(probs, dim=-1)

    def _padding_mask(self, x):
        return x == self.padding_idx
