import random
from transformer import Batchify


def test_small_batch():
    random.seed(0)

    b = Batchify([
        ([1,2,3], [7,8,9]),
        ([2,3,4], [8,9,10]),
        ([3,4,5], [9,10,11]),
        ([4,5,6], [10,11,12])
    ], 10)

    bb = list(b)
    assert len(bb) == 4
    
    assert bb[0].src.tolist() == [[2,3,4,5,3]]
    assert bb[0].trg.tolist() == [[2,9,10,11]]
    
    assert bb[1].src.tolist() == [[2,1,2,3,3]]
    assert bb[1].trg.tolist() == [[2,7,8,9]]
    
    assert bb[2].src.tolist() == [[2,2,3,4,3]]
    assert bb[2].trg.tolist() == [[2,8,9,10]]
    
    assert bb[3].src.tolist() == [[2,4,5,6,3]]
    assert bb[3].trg.tolist() == [[2,10,11,12]]


def test_large_batch():
    random.seed(0)

    b = Batchify([
        ([1,2], [3, 4]),
        ([3,4], [5, 6]),
        ([5,6], [7, 8]),
        ([7,8], [9, 10]),
        ([9,10], [11, 12]),
        ([11,12], [13, 14])
    ], 16)

    bb = list(b)
    assert len(bb) == 3
    
    assert bb[0].src.tolist() == [[2,1,2,3], [2,3,4,3]]
    assert bb[0].trg.tolist() == [[2,3,4], [2,5,6]]
    
    assert bb[1].src.tolist() == [[2,9,10,3], [2,11,12,3]]
    assert bb[1].trg.tolist() == [[2,11,12], [2,13,14]]
    
    assert bb[2].src.tolist() == [[2,5,6,3], [2,7,8,3]]
    assert bb[2].trg.tolist() == [[2,7,8], [2,9,10]]
