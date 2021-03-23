from aiger_bv import common

# additional imports for testing frammework
import hypothesis.strategies as st
from hypothesis import given, settings


@given(st.integers(-4, 3))
def test_seqcomp(a):
    circ1 = common.identity_gate(4, input='a', output='tmp')
    circ2 = common.identity_gate(4, input='tmp', output='out')
    circ3 = circ1 >> circ2
    assert circ3.inputs == circ1.inputs
    assert circ3.outputs == circ2.outputs

    val = circ3({
        'a': common.encode_int(4, a),
    })[0]['out']
    assert common.decode_int(val) == a


@given(st.integers(-4, 3))
def test_source(int_value):
    var = common.source(wordlen=4, value=int_value, name='x')
    assert common.decode_int(var({})[0]['x']) == int_value


@given(st.integers(-4, 3), st.integers(-4, 3))
def test_bitwise_and(a, b):
    circ = common.bitwise_and(4, left='a', right='b', output='out')
    val = circ({
        'a': common.encode_int(4, a),
        'b': common.encode_int(4, b),
    })[0]['out']
    assert common.decode_int(val) == a & b


@given(st.integers(-4, 3), st.integers(-4, 3))
def test_bitwise_or(a, b):
    circ = common.bitwise_or(4, left='a', right='b', output='out')
    val = circ({
        'a': common.encode_int(4, a),
        'b': common.encode_int(4, b),
    })[0]['out']
    assert common.decode_int(val) == a | b


@given(st.integers(-4, 3))
def test_bitwise_negate(a):
    circ = common.bitwise_negate(4, input='a', output='out')
    val = circ({'a': common.encode_int(4, a)})[0]['out']
    assert common.decode_int(val) == ~a


@given(st.integers(-4, 3))
def test_is_nonzero(a):
    circ = common.is_nonzero_gate(4, input='a', output='out')
    val = circ({'a': common.encode_int(4, a)})[0]['out']
    assert val[0] == (a != 0)


@given(st.integers(-4, 3))
def test_is_zero(a):
    circ = common.is_zero_gate(4, input='a', output='out')
    val = circ({'a': common.encode_int(4, a)})[0]['out']
    assert val[0] == (a == 0)


@given(st.integers(-4, 3), st.integers(-4, 3))
def test_ne(a, b):
    circ = common.ne_gate(4, left='a', right='b', output='out')
    val = circ({
        'a': common.encode_int(4, a),
        'b': common.encode_int(4, b),
    })[0]['out']
    assert val[0] == (a != b)


@given(st.integers(-4, 3), st.integers(-4, 3))
def test_eq(a, b):
    circ = common.eq_gate(4, left='a', right='b', output='out')
    val = circ({
        'a': common.encode_int(4, a),
        'b': common.encode_int(4, b),
    })[0]['out']
    assert val[0] == (a == b)


@given(st.integers(-4, 3))
def test_identity(a):
    circ = common.identity_gate(4, input='a', output='out')
    val = circ({'a': common.encode_int(4, a)})[0]['out']
    assert common.decode_int(val) == a


@given(st.integers(-4, 3))
def test_reverse(a):
    circ = common.reverse_gate(4, input='a', output='out')
    val = circ({'a': common.encode_int(4, a)})[0]['out']
    assert common.decode_int(val[::-1]) == a


@given(st.integers(-4, 3), st.integers(-4, 3))
def test_combine_gate(a, b):
    circ = common.combine_gate(4, 'a', 4, 'b', output='out')
    val = circ({
        'a': common.encode_int(4, a),
        'b': common.encode_int(4, b)
    })[0]['out']
    assert common.decode_int(val[:4]) == a
    assert common.decode_int(val[4:]) == b


@given(st.integers(-4, 3), st.integers(-4, 3))
def test_split_gate(a, b):
    circ = common.split_gate('input', 4, 'a', 4, 'b')
    val = circ({
        'input': common.encode_int(4, a) + common.encode_int(4, b),
    })[0]
    assert common.decode_int(val['a']) == a
    assert common.decode_int(val['b']) == b


@given(st.integers(-4, 3), st.integers(-4, 3))
def test_addition(a, b):
    circ = common.add_gate(4, left='a', right='b', output='out')
    val = circ({
        'a': common.encode_int(4, a),
        'b': common.encode_int(4, b),
    })[0]['out']
    assert common.decode_int(val) == a + b


@settings(deadline=None)
@given(st.integers(-4, 3))
def test_inc(a):
    circ = common.inc_gate(4, input='a', output='out')
    assert circ.inputs == {'a'}
    assert len(circ.aig.inputs) == 4
    assert circ.outputs == {'out'}
    assert len(circ.aig.outputs) == 4
    val = circ({'a': common.encode_int(4, a)})[0]['out']
    assert common.decode_int(val) == a + 1


@given(st.integers(-4, 3))
def test_negate(a):
    circ = common.negate_gate(4, input='a', output='out')
    assert circ.inputs == {'a'}
    assert len(circ.aig.inputs) == 4
    assert circ.outputs == {'out'}
    assert len(circ.aig.outputs) == 4
    val = circ({'a': common.encode_int(4, a)})[0]['out']
    assert common.decode_int(val) == -a


@given(st.integers(-4, 3))
def test_dec(a):
    circ = common.dec_gate(4, input='a', output='out')
    assert circ.inputs == {'a'}
    assert len(circ.aig.inputs) == 4
    assert circ.outputs == {'out'}
    assert len(circ.aig.outputs) == 4
    val = circ({'a': common.encode_int(4, a)})[0]['out']
    assert common.decode_int(val) == a - 1


@given(st.integers(-4, 3), st.integers(-4, 3))
def test_subtraction(a, b):
    circ = common.subtract_gate(4, left='a', right='b', output='out')
    val = circ({
        'a': common.encode_int(4, a),
        'b': common.encode_int(4, b),
    })[0]['out']
    assert common.decode_int(val) == a - b


@given(st.integers(0, 3), st.integers(0, 3))
def test_unsigned_lt(a, b):
    circ = common.unsigned_lt_gate(2, left='a', right='b', output='out')
    val = circ({
        'a': common.encode_int(2, a, signed=False),
        'b': common.encode_int(2, b, signed=False),
    })[0]['out'][0]
    assert val == (a < b)


@given(st.integers(0, 3), st.integers(0, 3))
def test_unsigned_le(a, b):
    circ = common.unsigned_le_gate(2, left='a', right='b', output='out')
    val = circ({
        'a': common.encode_int(2, a, signed=False),
        'b': common.encode_int(2, b, signed=False),
    })[0]['out'][0]
    assert val == (a <= b)


@given(st.integers(0, 3), st.integers(0, 3))
def test_unsigned_gt(a, b):
    circ = common.unsigned_gt_gate(2, left='a', right='b', output='out')
    val = circ({
        'a': common.encode_int(2, a, signed=False),
        'b': common.encode_int(2, b, signed=False),
    })[0]['out'][0]
    assert val == (a > b)


@given(st.integers(0, 3), st.integers(0, 3))
def test_unsigned_ge(a, b):
    circ = common.unsigned_ge_gate(2, left='a', right='b', output='out')
    val = circ({
        'a': common.encode_int(2, a, signed=False),
        'b': common.encode_int(2, b, signed=False),
    })[0]['out'][0]
    assert val == (a >= b)


@given(st.integers(-4, 3), st.integers(-4, 3))
def test_signed_lt(a, b):
    circ = common.signed_lt_gate(3, left='a', right='b', output='out')
    val = circ({
        'a': common.encode_int(3, a, signed=True),
        'b': common.encode_int(3, b, signed=True),
    })[0]['out'][0]
    assert val == (a < b)


@given(st.integers(-4, 3), st.integers(-4, 3))
def test_signed_gt(a, b):
    circ = common.signed_gt_gate(3, left='a', right='b', output='out')
    val = circ({
        'a': common.encode_int(3, a, signed=True),
        'b': common.encode_int(3, b, signed=True),
    })[0]['out'][0]
    assert val == (a > b)


@given(st.integers(-4, 3), st.integers(-4, 3))
def test_signed_ge(a, b):
    circ = common.signed_ge_gate(3, left='a', right='b', output='out')
    val = circ({
        'a': common.encode_int(3, a, signed=True),
        'b': common.encode_int(3, b, signed=True),
    })[0]['out'][0]
    assert val == (a >= b)


@given(st.integers(-4, 3), st.integers(-4, 3))
def test_signed_le(a, b):
    circ = common.signed_le_gate(3, left='a', right='b', output='out')
    val = circ({
        'a': common.encode_int(3, a, signed=True),
        'b': common.encode_int(3, b, signed=True),
    })[0]['out'][0]
    assert val == (a <= b)


@given(st.integers(-3, 3), st.integers(0, 3))
def test_left_shift(a, b):
    circ = common.left_shift_gate(8, b, 'a', output='out')
    val = circ({
        'a': common.encode_int(8, a, signed=True),
    })[0]['out']
    assert common.decode_int(val) == a << b


@given(st.integers(-4, 3), st.integers(0, 3))
def test_arithmetic_right_shift(a, b):
    circ = common.arithmetic_right_shift_gate(8, b, 'a', output='out')
    val = circ({
        'a': common.encode_int(8, a, signed=True),
    })[0]['out']
    assert common.decode_int(val) == a >> b


@given(st.integers(-4, 3), st.integers(0, 3))
def test_logical_right_shift(a, b):
    circ = common.logical_right_shift_gate(8, b, 'a', output='out')
    val = circ({
        'a': common.encode_int(8, a, signed=True),
    })[0]['out']
    val2 = common.decode_int(val)
    assert (val2 & (0xff >> b)) == ((a >> b) & (0xff >> b))
    if b != 0:
        assert val[-1] is False


@given(st.integers(-4, 3))
def test_abs(a):
    circ = common.abs_gate(8, 'a', output='out')
    val = circ({
        'a': common.encode_int(8, a, signed=True),
    })[0]['out']
    assert common.decode_int(val) == abs(a)


@given(st.integers(0, 3))
def test_lookup(a):
    lookup = {0: 0, 1: 1, 2: 0, 3: -1}
    circ = common.lookup(mapping=lookup,
                         input='x',
                         output='out',
                         inlen=2,
                         outlen=4,
                         in_signed=False)

    val = circ({
        'x': common.encode_int(2, a, signed=False),
    })[0]['out']
    assert common.decode_int(val) == lookup[a]


def test_kmodels():
    def _test_kmodels(k):
        circ = common.kmodels(4, k, input='x', output='x')
        models = 0
        for i in range(2**4):
            val = circ({
                'x': common.encode_int(4, i, signed=False)
            })[0]['x'][0]
            if val:
                models += 1
        assert models == k

    _test_kmodels(0)
    _test_kmodels(5)
    _test_kmodels(4)
    _test_kmodels(15)


def test_reinit():
    circ = common.add_gate(4, left='a', right='b', output='out')
    circ = circ.feedback(
        inputs=['a'],
        outputs=['out'],
        latches=['c'],
        initials=[(False, False, False, False)],
        keep_outputs=True
    )

    latch2init2 = {'c': (True, True, False, False)}
    circ2 = circ.reinit(latch2init2)
    assert circ2.latch2init == latch2init2
