from transformer import SyntheticData

def test_syntheticdata():
    s = SyntheticData(11, 4, 2)
    s.train(1)
