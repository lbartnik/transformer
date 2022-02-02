import torch
import torch.nn as nn
import pytest

from transformer import encoderdecoder

def test_encoderdecoder():
    torch.manual_seed(0)
    ed = encoderdecoder(10, 10, 4, 2)

    src = torch.LongTensor([[1,3,2,0], [2,1,2,3], [1,1,0,1]])
    tgt = torch.LongTensor([[5,4,9,8], [1,3,2,5], [0,3,1,2]])

    output = ed(src, tgt, None, None)

    criterion = nn.CrossEntropyLoss()
    # cross-entropy does not support sequences of words, but only minibatches
    # of words; thus, we "concatenate" all sequences in the minibatch into
    # one large sequence-batch
    loss = criterion(torch.exp(output.view(-1, 10)), tgt.reshape(-1))
    assert 2.6558 == pytest.approx(loss.item())