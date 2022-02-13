import torch
import numpy as np

from .labelsmoothing import LabelSmoothing
from .noamopt import NoamOpt
from .makemodel import make_model
from .batch import Batch
from .runepoch import run_epoch
from .simplelosscompute import SimpleLossCompute
from .greedydecode import greedy_decode

def data_gen(V, batch, nbatches):
    "Generate random data for a src-tgt copy task."
    for i in range(nbatches):
        data = torch.randint(1, V, size=(batch, 10))
        data[:, 0] = 1
        src = data.detach()
        tgt = data.detach()
        yield Batch(src, tgt, 0)


class SyntheticData:

    def __init__(self, vocab=11, n_features=512, h=8, padding_idx=0):
        self.vocab = vocab
        self.model = make_model(vocab, vocab, N=2, n_features=n_features, h=h)
        self.padding_idx = padding_idx

    def train(self, nepoch=10):
        criterion = LabelSmoothing(size=self.vocab, padding_idx=self.padding_idx, smoothing=0.0)
        
        model_opt = NoamOpt(self.model.src_embed[0].n_features, 1, 400,
                torch.optim.Adam(self.model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

        for epoch in range(nepoch):
            self.model.train()
            run_epoch(data_gen(self.vocab, 30, 20), self.model, 
                    SimpleLossCompute(criterion, model_opt))
            
            self.model.eval()
            with torch.no_grad():
                loss = run_epoch(data_gen(self.vocab, 30, 5), self.model, 
                                SimpleLossCompute(criterion, None))
                print(f"Epoch {epoch} completed with validation loss {loss}")

    def translate(self, src, start_symbol, max_len=5000):
        if type(src) is list:
            src = torch.LongTensor(src)
            while len(src.shape) != 2:
                src.unsqueeze_(0)

        return greedy_decode(self.model, src, src != self.padding_idx, max_len, start_symbol)
