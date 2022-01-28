from transformer import subsequent_mask

def test_mask():
    m = subsequent_mask(2)
    assert m.tolist() == [[[False, False], [True, False]]]