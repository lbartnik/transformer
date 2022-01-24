import torch
import torch.nn as nn
import pytest

from transformer.clones import clones 

def test_length():
    assert len(clones(nn.Linear(10, 10), 3)) == 3
    assert len(clones(nn.Linear(10, 10), 5)) == 5


# Unlike nn.Sequential, nn.ModuleList cannot be used as a layer
def test_cannot_be_called_directly():
    with pytest.raises(NotImplementedError):
        clones(nn.Linear(10, 10), 3)(torch.Tensor([range(10)]))

def test_can_call_layers():
    x = torch.Tensor([range(10)])
    layers = clones(nn.Linear(10, 10), 3)

    for l in layers:
        y = l(x)
        assert y.shape == (1,10)
