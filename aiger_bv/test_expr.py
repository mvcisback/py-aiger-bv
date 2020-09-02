from aiger_bv.expr import uatom, atom, ite
from aiger_bv import common

# additional imports for testing frammework
import hypothesis.strategies as st
from hypothesis import given, settings


@settings(max_examples=5, deadline=None)
@given(st.integers(0, 7), st.integers(0, 3))
def test_srl_unsigned(a, b):
    expr = uatom(4, a) >> b
    assert common.decode_int(expr(), signed=False) == a >> b


@settings(max_examples=5, deadline=None)
@given(st.integers(-8, 7), st.integers(0, 3))
def test_srl_signed(a, b):
    expr = atom(4, a) >> b
    assert common.decode_int(expr()) == a >> b


@settings(max_examples=5, deadline=None)
@given(st.integers(0, 7), st.integers(0, 3))
def test_sll(a, b):
    wordlen = 4
    expr = atom(wordlen, a, signed=False) << b
    mask = (1 << wordlen) - 1
    assert bin(common.decode_int(expr(), signed=False)) == bin((a << b) & mask)


@settings(max_examples=5, deadline=None)
@given(st.integers(0, 7), st.integers(0, 7))
def test_expr_unsigned_lt_literal(a, b):
    expr = uatom(4, a) < b
    assert expr()[0] == (a < b)


@settings(max_examples=5, deadline=None)
@given(st.integers(0, 7), st.integers(0, 7))
def test_expr_unsigned_gt_literal(a, b):
    expr = atom(4, a, signed=False) > b
    assert expr()[0] == (a > b)


@settings(max_examples=5, deadline=None)
@given(st.integers(0, 7), st.integers(0, 7))
def test_expr_unsigned_ge_literal(a, b):
    expr = atom(4, a, signed=False) >= b
    assert expr()[0] == (a >= b)


@settings(max_examples=5, deadline=None)
@given(st.integers(0, 7), st.integers(0, 7))
def test_expr_unsigned_le_literal(a, b):
    expr = atom(4, a, signed=False) <= b
    assert expr()[0] == (a <= b)


@settings(max_examples=5, deadline=None)
@given(st.integers(-7, 7), st.integers(0, 7))
def test_expr_signed_lt_literal(a, b):
    expr = atom(4, a, signed=True) < b
    assert expr()[0] == (a < b)


@settings(max_examples=5, deadline=None)
@given(st.integers(-7, 7), st.integers(0, 7))
def test_expr_signed_gt_literal(a, b):
    expr = atom(4, a, signed=True) > b
    assert expr()[0] == (a > b)


@settings(max_examples=5, deadline=None)
@given(st.integers(-7, 7), st.integers(0, 7))
def test_expr_signed_ge_literal(a, b):
    expr = atom(4, a, signed=True) >= b
    assert expr()[0] == (a >= b)


@settings(max_examples=5, deadline=None)
@given(st.integers(-7, 7), st.integers(0, 7))
def test_expr_signed_le_literal(a, b):
    expr = atom(4, a, signed=True) <= b
    assert expr()[0] == (a <= b)


@settings(max_examples=5, deadline=None)
@given(st.integers(-4, 3), st.integers(-4, 3))
def test_expr_bitwise_and(a, b):
    expr = atom(4, a) & atom(4, b)
    assert common.decode_int(expr()) == a & b


@settings(max_examples=5, deadline=None)
@given(st.integers(-4, 3))
def test_expr_bitwise_and2(a):
    expr = atom(4, a) & atom(4, a)
    assert common.decode_int(expr()) == a


@settings(max_examples=5, deadline=None)
@given(st.integers(-4, 3), st.integers(-4, 3))
def test_expr_bitwise_or(a, b):
    expr = atom(4, a) | atom(4, b)
    assert common.decode_int(expr()) == a | b


@settings(max_examples=5, deadline=None)
@given(st.integers(-4, 3), st.integers(-4, 3))
def test_expr_bitwise_xor(a, b):
    expr = atom(4, a) ^ atom(4, b)
    assert common.decode_int(expr()) == a ^ b


@settings(max_examples=5, deadline=None)
@given(st.integers(-4, 3))
def test_expr_bitwise_invert(a):
    expr = ~atom(4, a)
    assert common.decode_int(expr()) == ~a


@settings(max_examples=5, deadline=None)
@given(st.integers(-4, 3), st.integers(-4, 3))
def test_expr_eq(a, b):
    expr = atom(4, a) == atom(4, b)
    assert expr()[0] == (a == b)


@settings(max_examples=5, deadline=None)
@given(st.integers(-4, 3), st.integers(-4, 3))
def test_expr_ne(a, b):
    expr = atom(4, a) != atom(4, b)
    assert expr()[0] == (a != b)


@settings(max_examples=5, deadline=None)
@given(st.integers(-4, 3), st.integers(-4, 3))
def test_expr_add(a, b):
    expr = atom(4, a) + atom(4, b)
    assert common.decode_int(expr()) == a + b


@settings(max_examples=5, deadline=None)
@given(st.integers(-4, 3), st.integers(-4, 3))
def test_expr_sub(a, b):
    expr = atom(4, a) - atom(4, b)
    assert common.decode_int(expr()) == a - b


@settings(max_examples=5, deadline=None)
@given(st.integers(-4, 3), st.integers(-4, 3))
def test_expr_le(a, b):
    expr = atom(4, a) <= atom(4, b)
    assert expr()[0] == (a <= b)


@settings(max_examples=5, deadline=None)
@given(st.integers(-4, 3), st.integers(-4, 3))
def test_expr_lt(a, b):
    expr = atom(4, a) < atom(4, b)
    assert expr()[0] == (a < b)


@settings(max_examples=5, deadline=None)
@given(st.integers(-4, 3), st.integers(-4, 3))
def test_expr_ge(a, b):
    expr = atom(4, a) >= atom(4, b)
    assert expr()[0] == (a >= b)


@settings(max_examples=5, deadline=None)
@given(st.integers(-4, 3), st.integers(-4, 3))
def test_expr_gt(a, b):
    expr = atom(4, a) > atom(4, b)
    assert expr()[0] == (a > b)


@settings(max_examples=5, deadline=None)
@given(st.integers(0, 7), st.integers(0, 7))
def test_expr_unsigned_le(a, b):
    expr = atom(4, a, signed=False) <= atom(4, b, signed=False)
    assert expr()[0] == (a <= b)


@settings(max_examples=5, deadline=None)
@given(st.integers(0, 7), st.integers(0, 7))
def test_expr_unsigned_lt(a, b):
    expr = atom(4, a, signed=False) < atom(4, b, signed=False)
    assert expr()[0] == (a < b)


@settings(max_examples=5, deadline=None)
@given(st.integers(0, 7), st.integers(0, 7))
def test_expr_unsigned_gt(a, b):
    expr = atom(4, a, signed=False) > atom(4, b, signed=False)
    assert expr()[0] == (a > b)


@settings(max_examples=5, deadline=None)
@given(st.integers(0, 7), st.integers(0, 7))
def test_expr_unsigned_ge(a, b):
    expr = atom(4, a, signed=False) >= atom(4, b, signed=False)
    assert expr()[0] == (a >= b)


@settings(max_examples=5, deadline=None)
@given(st.integers(-4, 3))
def test_expr_neg(a):
    expr = -atom(4, a)
    assert common.decode_int(expr()) == -a


@settings(max_examples=5, deadline=None)
@given(st.integers(-7, 7))
def test_expr_abs(a):
    expr = abs(atom(4, a))
    assert common.decode_int(expr()) == abs(a)


def test_expr_abs_max_negative():
    expr = abs(atom(4, -8))
    assert common.decode_int(expr()) == -8  # undefined behavior


@settings(max_examples=5, deadline=None)
@given(st.integers(-4, 3))
def test_expr_getitem(a):
    expr = atom(4, a)
    for i in range(4):
        assert common.decode_int(expr[i](), signed=False) == (a >> i) & 1


@settings(max_examples=5, deadline=None)
@given(st.integers(-4, 3), st.integers(-4, 3))
def test_expr_concat(a, b):
    expr1, expr2 = atom(4, a), atom(4, b)
    expr3 = expr1.concat(expr2)
    assert expr3.size == expr1.size + expr2.size
    assert expr3() == expr1() + expr2()


@settings(max_examples=5, deadline=None)
@given(st.booleans(), st.integers(1, 5))
def test_expr_repeat(a, b):
    expr = atom(1, a, signed=False)
    assert expr.repeat(b)() == b * expr()


@settings(max_examples=5, deadline=None)
@given(st.integers(-4, 3), st.integers(-4, 3))
def test_expr_dotprod_mod2(a, b):
    expr1, expr2 = atom(4, a), atom(4, b)
    expr3 = expr1 @ expr2
    val = sum([x * y for x, y in zip(expr1(), expr2())])
    assert expr3()[0] == bool(val % 2)


@settings(max_examples=5, deadline=None)
@given(st.booleans(), st.integers(-4, 3), st.integers(-4, 3))
def test_ite(test, a, b):
    _test, _a, _b = atom(1, test, signed=False), atom(4, a), atom(4, b)
    expr = ite(_test, _a, _b)
    val = common.decode_int(expr())
    assert val == (a if test else b)


def test_ite2():
    test, a, b = atom(1, 'test', signed=False), atom(2, 'x'), atom(2, 'y')
    expr = ite(test, a, b)
    val = expr({'test': (False,), 'x': (False, False), 'y': (True, False)})
    assert val == (True, False)


def test_set_output():
    x, y = atom(2, 'x'), atom(2, 'y')
    f = (x < y).with_output('z')
    assert f.output == 'z'


def test_bundle_inputs():
    x, y = atom(2, 'x'), atom(2, 'y')
    f = x.concat(y)
    assert f.inputs == {'x', 'y'}
    f = f.bundle_inputs('xy')
    assert f.inputs == {'xy'}
    assert f.aigbv.imap['xy'].size == 4


def test_indexing_preserves_inputs():
    x = atom(2, 'x')
    assert x[0].inputs == {'x'}


def test_and_preserves_inputs():
    x = atom(2, 'x')
    y = atom(2, 'y')
    assert (x & y).inputs == {'x', 'y'}


@settings(max_examples=5, deadline=None)
@given(st.integers(0, 15), st.integers(0, 15))
def test_unsigned_add(x, y):
    if x + y > 15:
        return
    adder = atom(4, 'a', signed=False) + atom(4, 'b', signed=False)
    x_atom, y_atom = atom(4, x, signed=False), atom(4, y, signed=False)
    res = adder(inputs={'a': x_atom(), 'b': y_atom()})
    assert res == atom(4, x + y, signed=False)()


@settings(max_examples=5, deadline=None)
@given(st.integers(0, 15), st.integers(0, 15))
def test_unsigned_multiply(x, y):
    if x * y > 15:
        return
    multiplier = atom(4, 'a', signed=False) * atom(4, 'b', signed=False)
    assert len(multiplier.inputs) == 2
    x_atom, y_atom = atom(4, x, signed=False), atom(4, y, signed=False)
    res = multiplier(inputs={'a': x_atom(), 'b': y_atom()})
    assert common.decode_int(res, signed=False) == x * y


@settings(max_examples=5, deadline=None)
@given(st.integers(-8, 7))
def test_sign(x):
    sign_expr = atom(4, x).sign()
    assert sign_expr() == (x < 0,)


@settings(max_examples=5, deadline=None)
@given(st.integers(-4, 3), st.integers(-4, 3))
def test_multiply(x, y):
    multiplier = atom(6, 'a') * atom(6, 'b')
    x_atom, y_atom = atom(6, x), atom(6, y)
    res = multiplier(inputs={'a': x_atom(), 'b': y_atom()})
    assert common.decode_int(res) == x * y


@settings(max_examples=5, deadline=None)
@given(st.integers(-4, 3), st.integers(-4, 3))
def test_multiply_lit(x, y):
    multiplier = atom(6, 'a') * y
    x_atom = atom(6, x)
    res = multiplier(inputs={'a': x_atom()})
    assert common.decode_int(res) == x * y


def test_encoded_add():
    adder = uatom(3, 'x') + 2
    res = adder(inputs={'x': 0b000})
    assert common.decode_int(res) == 2
