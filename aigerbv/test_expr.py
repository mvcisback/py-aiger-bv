from aigerbv.expr import atom
from aigerbv import common

# additional imports for testing frammework
import hypothesis.strategies as st
from hypothesis import given


@given(st.integers(-4, 3), st.integers(-4, 3))
def test_expr_bitwise_and(a, b):
    expr = atom(4, a) & atom(4, b)
    assert common.decode_int(expr()) == a & b


@given(st.integers(-4, 3), st.integers(-4, 3))
def test_expr_bitwise_or(a, b):
    expr = atom(4, a) | atom(4, b)
    assert common.decode_int(expr()) == a | b


@given(st.integers(-4, 3), st.integers(-4, 3))
def test_expr_bitwise_xor(a, b):
    expr = atom(4, a) ^ atom(4, b)
    assert common.decode_int(expr()) == a ^ b


@given(st.integers(-4, 3))
def test_expr_bitwise_invert(a):
    expr = ~atom(4, a)
    assert common.decode_int(expr()) == ~a


@given(st.integers(-4, 3), st.integers(-4, 3))
def test_expr_eq(a, b):
    expr = atom(4, a) == atom(4, b)
    assert expr()[0] == (a == b)


@given(st.integers(-4, 3), st.integers(-4, 3))
def test_expr_ne(a, b):
    expr = atom(4, a) != atom(4, b)
    assert expr()[0] == (a != b)


@given(st.integers(-4, 3), st.integers(-4, 3))
def test_expr_add(a, b):
    expr = atom(4, a) + atom(4, b)
    assert common.decode_int(expr()) == a + b


@given(st.integers(-4, 3), st.integers(-4, 3))
def test_expr_sub(a, b):
    expr = atom(4, a) - atom(4, b)
    assert common.decode_int(expr()) == a - b


@given(st.integers(-4, 3), st.integers(-4, 3))
def test_expr_le(a, b):
    expr = atom(4, a) <= atom(4, b)
    assert expr()[0] == (a <= b)


@given(st.integers(-4, 3), st.integers(-4, 3))
def test_expr_lt(a, b):
    expr = atom(4, a) < atom(4, b)
    assert expr()[0] == (a < b)


@given(st.integers(-4, 3), st.integers(-4, 3))
def test_expr_ge(a, b):
    expr = atom(4, a) >= atom(4, b)
    assert expr()[0] == (a >= b)


@given(st.integers(-4, 3), st.integers(-4, 3))
def test_expr_gt(a, b):
    expr = atom(4, a) > atom(4, b)
    assert expr()[0] == (a > b)


@given(st.integers(0, 7), st.integers(0, 7))
def test_expr_unsigned_le(a, b):
    expr = atom(4, a, signed=False) <= atom(4, b, signed=False)
    assert expr()[0] == (a <= b)


@given(st.integers(0, 7), st.integers(0, 7))
def test_expr_unsigned_lt(a, b):
    expr = atom(4, a, signed=False) < atom(4, b, signed=False)
    assert expr()[0] == (a < b)


@given(st.integers(0, 7), st.integers(0, 7))
def test_expr_unsigned_gt(a, b):
    expr = atom(4, a, signed=False) > atom(4, b, signed=False)
    assert expr()[0] == (a > b)


@given(st.integers(0, 7), st.integers(0, 7))
def test_expr_unsigned_ge(a, b):
    expr = atom(4, a, signed=False) >= atom(4, b, signed=False)
    assert expr()[0] == (a >= b)


@given(st.integers(-4, 3))
def test_expr_neg(a):
    expr = -atom(4, a)
    assert common.decode_int(expr()) == -a


@given(st.integers(-4, 3))
def test_expr_abs(a):
    expr = abs(atom(4, a))
    assert common.decode_int(expr()) == abs(a)
