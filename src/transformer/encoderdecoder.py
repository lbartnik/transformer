import torch.nn as nn

from .decoderlayer import decoderlayer
from .decoder import Decoder
from .encoderlayer import EncoderLayer
from .encoder import Encoder
from .multiheadedattention import MultiHeadedAttention
from .embeddings import Embeddings
from .positionalencoding import PositionalEncoding
from .generator import Generator

class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        
    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.generator(
            self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask))
    
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)



def encoderdecoder(src_vocab, tgt_vocab, n_features, h=8, N=6, dropout=.1, d_ff=2048):

    enc_layer = EncoderLayer(n_features, MultiHeadedAttention(h, n_features, dropout), nn.Linear(n_features, n_features), dropout)
    enc = Encoder(enc_layer, N)

    dec_layer = decoderlayer(n_features, h, d_ff, dropout)
    dec = Decoder(dec_layer, N)

    src = nn.Sequential(Embeddings(n_features, src_vocab), PositionalEncoding(n_features, dropout))
    tgt = nn.Sequential(Embeddings(n_features, tgt_vocab), PositionalEncoding(n_features, dropout))

    return EncoderDecoder(enc, dec, src, tgt, Generator(n_features, tgt_vocab))
