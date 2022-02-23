import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from torchtext.data.metrics import bleu_score

from .embeddings import Embeddings
from .positionalencoding import PositionalEncoding
from .scheduler import NoamScheduler


class Transformer(pl.LightningModule):
    def __init__(self, src_vocab, trg_vocab, d_model=512, h=8, N=6, ff_dim=2048, padding_idx=0, warmup=4000):
        super().__init__()
        self.src_embed = nn.Sequential(Embeddings(d_model, src_vocab, padding_idx), PositionalEncoding(d_model))
        self.trg_embed = nn.Sequential(Embeddings(d_model, trg_vocab, padding_idx), PositionalEncoding(d_model))
        self.transformer = nn.Transformer(d_model=d_model, nhead=h, num_encoder_layers=N, num_decoder_layers=N,
                                          dim_feedforward=ff_dim, batch_first=True)
        self.generator = nn.Linear(d_model, trg_vocab)
        self.loss_module = nn.NLLLoss(reduction='sum')

        self.padding_idx = padding_idx
        self.d_model = d_model
        self.warmup = warmup

        # Example input for visualizing the graph in Tensorboard
        self.example_input_array = torch.zeros((1, 4), dtype=torch.long)

        # TODO differentiate
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, trg=None):
        if trg is None:
            trg = src
        out = self.transformer(self.src_embed(src), self.trg_embed(trg),
                               tgt_mask=nn.Transformer.generate_square_subsequent_mask(trg.size(-1)),
                               src_key_padding_mask=src == self.padding_idx,
                               tgt_key_padding_mask=trg == self.padding_idx)
        probs = self.generator(out)
        return F.log_softmax(probs, dim=-1)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=0, betas=(0.9, 0.98))
        scheduler = NoamScheduler(optimizer, d_model=self.d_model, warmup=self.warmup, factor=1)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx=None):
        src, trg, trg_y, ntokens = self._prep_batch(batch)
        
        log_probs = self.forward(src, trg)
        loss = self.loss_module(log_probs.view(-1, log_probs.size(-1)), trg_y.contiguous().view(-1)) / ntokens

        # Logs the accuracy per epoch to tensorboard (weighted average over batches)
        self.log("train_loss", loss)
        return loss  # Return tensor to call ".backward" on

    def validation_step(self, batch, batch_idx=None):
        src, trg, trg_y, ntokens = self._prep_batch(batch)

        log_probs = self.forward(src, trg)
        loss = self.loss_module(log_probs.view(-1, log_probs.size(-1)), trg_y.contiguous().view(-1))
        self.log("val_loss", loss)

        _, preds = torch.max(log_probs, dim = -1)
        self.log("val_acc", (trg_y == preds).sum().float()/ntokens)

    def test_step(self, batch, batch_idx=None):
        src, trg, trg_y, ntokens = self._prep_batch(batch)
        log_probs = self.forward(src, trg)
        _, preds = torch.max(log_probs, dim = -1)
        self.log("test_acc", (trg_y == preds).sum().float()/ntokens)


    def _prep_batch(self, batch):
        src, trg = batch
        trg, trg_y = trg[:,:-1], trg[:,1:]
        ntokens = (trg_y != self.padding_idx).sum().float().item()

        return src, trg, trg_y, ntokens
