from transformer import SyntheticData

# this is a very simple test: transformer's main dimension is only 4 and we train for only 1 epoch
def test_syntheticdata():
    s = SyntheticData(11, 4, 2)
    s.train(1)
    o = s.translate([1, 2, 3], 1, 5)
    assert o.squeeze().tolist() == [1, 3, 3, 3, 3]
