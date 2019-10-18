import pytest

from aigerbv import bundle as bdl


def test_smoke():
    b = bdl.Bundle("test", 3)
    assert b.size == 3
    assert b.name == "test"
    assert b == bdl.Bundle("test", 3)
    assert str(b) == "test[:3]"


def test_validator():
    with pytest.raises(ValueError):
        bdl.Bundle("test", -1)


def test_add():
    x = bdl.Bundle("x", 3)
    y = bdl.Bundle("y", 5)

    z = x + y
    assert len(z) == len(x) + len(y)
    assert repr(z) == "x#+#y[:8]"


def test_idx():
    x = bdl.Bundle("x", 10)

    assert x[0] == "x[0]"
    assert x[5] == "x[5]"
    assert x[9] == "x[9]"

    with pytest.raises(AssertionError):
        x[-1]

    with pytest.raises(AssertionError):
        x[-2]

    assert x[:4] == ("x[0]", "x[1]", "x[2]", "x[3]")
