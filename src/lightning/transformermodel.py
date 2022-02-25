import torch
import torch.nn as nn
import torch.nn.functional as F


from .embeddings import Embeddings
from .positionalencoding import PositionalEncoding
from .paddingmask import padding_mask


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
                               src_key_padding_mask=padding_mask(src, self.padding_idx),
                               tgt_key_padding_mask=padding_mask(trg, self.padding_idx))
        probs = self.generator(out)
        return F.log_softmax(probs, dim=-1)


    def greedy_decode(self, src, start_symbol, max_len=50, padding_idx=0, repeat_allowed=True):
        """
        Generates the output/decoded/translated sequence given the input/source sequence.

        :param model: EncoderDecoder object
        :param src: tokens (LongTensor)
        :param src_mask: True where token is not padding, False for padding tokens
        :param max_len: number of decoding steps
        :param start_symbol: initial symbol in the output/translated sequence
        """

        memory = self.transformer.encoder(self.src_embed(src), src_key_padding_mask=padding_mask(src, padding_idx))
        ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)

        for _ in range(max_len-1):
            out = self.transformer.decoder(self.trg_embed(ys), memory,
                                           tgt_mask=nn.Transformer.generate_square_subsequent_mask(ys.size(-1)),
                                           tgt_key_padding_mask=None)
            prob = self.generator(out[:, -1])

            if not repeat_allowed:
                prob[0, ys[0][-1]] = -100 # do not return the same word as the last word in the input
            
            _, next_word = torch.max(prob, dim = 1)
            next_word = next_word.data[0]
            ys = torch.cat([ys, 
                            torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
        return ys
