import torch
import torch.utils.data as data
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from .transformer import Transformer


class SynthDataset(data.Dataset):
    def __init__(self, L, N):
        self.data = torch.randint(1, L, size=(N, L))

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        r = self.data[idx]
        return (r, r)


def run(L=10, N=10000, batch_size=100, max_epochs=180, device='cpu', checkpoint_path='./synth'):
    train = data.DataLoader(SynthDataset(L, N), batch_size=batch_size)
    val = data.DataLoader(SynthDataset(L, N // 100), batch_size=batch_size)

    trainer = pl.Trainer(
        default_root_dir=checkpoint_path,
        # We run on a single GPU (if possible)
        gpus=1 if str(device) == "cuda:0" else 0,
        # How many epochs to train for if no patience is set
        max_epochs=max_epochs,
        callbacks=[
            # Save the best checkpoint based on the maximum val_acc recorded. Saves only weights and not optimizer
            ModelCheckpoint(
                save_weights_only=True, mode="max", monitor="val_acc"
            ),
            # Log learning rate every epoch
            LearningRateMonitor("epoch")
        ]
    )

    model = Transformer(L, L)
    trainer.fit(model, train, val)
    