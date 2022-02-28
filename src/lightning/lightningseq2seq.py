import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import time

from .scheduler import NoamScheduler
from .transformermodel import TransformerModel
from .seq2seqloss import Seq2SeqLoss


class LightningSeq2Seq(pl.LightningModule):
    def __init__(self, src_vocab, trg_vocab, d_model=512, h=8, N=6, ff_dim=2048, padding_idx=0, warmup=4000, factor=1):
        super(LightningSeq2Seq, self).__init__()
        self.model = TransformerModel(src_vocab, trg_vocab, d_model, h, N, ff_dim, padding_idx)
        self.loss_module = Seq2SeqLoss(padding_idx)

        self.padding_idx = padding_idx
        self.d_model = d_model
        self.warmup = warmup
        self.factor = factor

        # Example input for visualizing the graph in Tensorboard
        self.example_input_array = torch.zeros((1, 4), dtype=torch.long)

    def forward(self, src, trg=None):
        # when PyTorch Lightning runs a test call to estimate model size
        if trg is None:
            trg = src
        
        return self.model(src, trg)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=0, betas=(0.9, 0.98))
        scheduler = NoamScheduler(optimizer, d_model=self.d_model, warmup=self.warmup, factor=self.factor)
        return [optimizer], {"scheduler": scheduler, "interval": "step"}

    def training_step(self, batch, batch_idx=None):
        start = time.time()

        src, trg, trg_y = _prep_batch(batch)
        log_probs = self.model(src, trg)
        loss = self.loss_module(log_probs, trg_y)

        elapsed = time.time() - start

        # Logs the accuracy per epoch to tensorboard (weighted average over batches)
        self.log("train_loss", loss)
        self.log("ntokens", self.loss_module.ntokens)
        self.log("tok_sec", self.loss_module.ntokens/elapsed, prog_bar=True)
        return loss  # Return tensor to call ".backward" on

    def validation_step(self, batch, batch_idx=None):
        src, trg, trg_y = _prep_batch(batch)
        log_probs = self.model(src, trg)
        loss = self.loss_module(log_probs, trg_y)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx=None):
        src, trg, trg_y = _prep_batch(batch)
        log_probs = self.model(src, trg)
        loss = self.loss_module(log_probs, trg_y)
        self.log("test_loss", loss)

def _prep_batch(batch):
    src, trg = batch
    trg, trg_y = trg[:,:-1], trg[:,1:]
    return src, trg, trg_y
