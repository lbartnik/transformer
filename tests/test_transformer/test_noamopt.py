import pytest
from transformer import NoamOpt

def test_noamopt():
    n = NoamOpt(100, 1, 100, None)
    assert n.rate(1) == 0.0001
    assert n.rate(100) == pytest.approx(0.01)
