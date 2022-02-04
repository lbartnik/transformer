import torch
from transformer import Batch

def test_batch():
    src = torch.Tensor([[1,2,3,0], [5,1,0,0], [8,1,5,2]])
    trg = torch.Tensor([[5,0,0,0], [7,2,0,0], [5,2,4,1]])
    b = Batch(src, trg)

    # unsqueeze(1) adds a new dimension between dimensions associated
    # with mini-batch (0) and phrase (1)
    exp_src_mask = torch.Tensor([[1,1,1,0], [1,1,0,0], [1,1,1,1]]).unsqueeze(1) != 0
    assert torch.eq(exp_src_mask, b.src_mask).all()

    exp_trg_mask = torch.Tensor([
        [
            [0,0,0],
            [1,0,0],
            [1,0,0]
        ],
        [
            [0,0,0],
            [1,0,0],
            [1,1,0]
        ],
        [
            [0,0,0],
            [1,0,0],
            [1,1,0]
        ]
    ]) != 0
    assert torch.eq(exp_trg_mask, b.trg_mask).all()
