import operator as op
from functools import reduce
from uuid import uuid1

import aiger

from aigerbv import aigbv


def _fresh():
    return str(uuid1())


def named_indexes(wordlen, root):
    return tuple(f"{root}[{i}]" for i in range(wordlen))


def bitwise_binop(binop, wordlen, left='x', right='y', output='x&y'):
    lefts = named_indexes(wordlen, left)
    rights = named_indexes(wordlen, right)
    outputs = named_indexes(wordlen, output)

    aig = reduce(op.or_, (binop([l, r], o)
                          for l, r, o in zip(lefts, rights, outputs)))
    return aigbv.AIGBV(
        aig=aig,
        input_map=frozenset([(left, lefts), (right, rights)]),
        output_map=frozenset([(output, outputs)]),
        latch_map=frozenset(),
    )


def bitwise_and(wordlen, left='x', right='y', output='x&y'):
    return bitwise_binop(aiger.and_gate, wordlen, left, right, output)


def bitwise_or(wordlen, left='x', right='y', output='x&y'):
    return bitwise_binop(aiger.or_gate, wordlen, left, right, output)


def bitwise_xor(wordlen, left='x', right='y', output='x&y'):
    return bitwise_binop(aiger.parity_gate, wordlen, left, right, output)


def bitwise_negate(wordlen, input='x', output='not x'):
    inputs = named_indexes(wordlen, input)
    outputs = named_indexes(wordlen, output)
    return aigbv.AIGBV(
        aig=aiger.bit_flipper(inputs=inputs, outputs=outputs),
        input_map=frozenset([(input, inputs)]),
        output_map=frozenset([(output, outputs)]),
        latch_map=frozenset(),
    )


def is_nonzero_gate(wordlen, input='x', output='is_nonzero'):
    inputs = named_indexes(wordlen, input)
    outputs = named_indexes(1, output)
    return aigbv.AIGBV(
        aig=aiger.or_gate(inputs, outputs[0]),
        input_map=frozenset([(input, inputs)]),
        output_map=frozenset([(output, outputs)]),
        latch_map=frozenset(),
    )


def neq_gate(wordlen, left='x', right='y', output='x!=y'):
    return bitwise_xor(wordlen, left, right, left+right) >> \
        is_nonzero_gate(wordlen, left+right, output)


def is_zero_gate(wordlen, input='x', output='is_zero'):
    return is_nonzero_gate(wordlen, input, input+'out') >> \
        bitwise_negate(1, input+'out', output)


def eq_gate(wordlen, left='x', right='y', output='x=y'):
    return neq_gate(wordlen, left, right, left+right) >> \
        bitwise_negate(1, left+right, output)


def source(wordlen, value, name='x', signed=True, byteorder='little'):
    assert 2**wordlen > value
    bits = value.to_bytes(wordlen, byteorder, signed=signed)
    names = named_indexes(wordlen, name)
    return aigbv.AIGBV(
        aig=aiger.source({name: bit
                          for name, bit in zip(names, bits)}),
        input_map=frozenset(),
        output_map=frozenset([(name, names)]),
        latch_map=frozenset(),
    )


def identity_gate(wordlen, input='x', output='x'):
    return aigbv.aig2aigbv(
        aiger.identity(
            inputs=named_indexes(wordlen, input),
            outputs=named_indexes(output)))


def reverse_gate(wordlen, input='x', output='rev(x)'):
    circ = identity_gate(wordlen, input, output)
    return identity_gate(wordlen, input, output)._replace(
        output_map=frozenset((k, reversed(vs)) for k, vs in circ.output_map))


def combine_gate(left_wordlen, left, right_wordlen, right, output):
    lefts = named_indexes(left_wordlen, left)
    rights = named_indexes(right_wordlen, right)
    outputs = named_indexes(left_wordlen + right_wordlen, output)

    aig = identity_gate(left_wordlen, left, left) \
        | identity_gate(right_wordlen, right, right)
    return aig >> aigbv.aig2aigbv(
        aiger.identity(inputs=lefts + rights, outputs=outputs))


def split_gate(left_wordlen, left, right_wordlen, right, input):
    inputs = named_indexes(left_wordlen + right_wordlen, input)
    lefts = named_indexes(left_wordlen, left)
    rights = named_indexes(right_wordlen, right)

    aig = identity_gate(left_wordlen, left, left) \
        | identity_gate(right_wordlen, right, right)
    return aigbv.aig2aigbv(
        aiger.identity(inputs=inputs, outputs=lefts + rights)) >> aig


def unsigned_right_shift_gate(wordlen, shift, input='x', output='x'):
    assert 0 <= shift < wordlen

    if shift == 0:
        return identity_gate(wordlen, input, output)

    inputs = named_indexes(wordlen, input)
    outputs = named_indexes(wordlen, output)
    _, shifted_i = inputs[:shift], inputs[shift:]
    const_o, shifted_o = outputs[:shift], outputs[shift:]

    aig = aiger.identity(shifted_i, shifted_o)
    aig |= aiger.source({n: False for n in const_o})
    return aigbv.AIGBV(
        aig=aig,
        input_map=frozenset([(input, inputs)]),
        output_map=frozenset([output, outputs]),
        latch_map=frozenset(),
    )


def unsigned_left_shift_gate(wordlen, shift, input='x', output='x'):
    return reverse_gate(wordlen, input, 'tmp') \
        >> unsigned_right_shift_gate(wordlen, shift, 'tmp', 'tmp') \
        >> reverse_gate(wordlen, 'tmp', output)


def _full_adder(x, y, carry_in, result, carry_out):
    # TODO: Rewrite in aiger.
    return aiger.parse(
        "aag 10 3 0 2 7\n2\n4\n6\n18\n21\n8 4 2\n10 5 3\n"
        "12 11 9\n14 12 6\n16 13 7\n18 17 15\n20 15 9\n"
        f"i0 {x}\ni1 {y}\ni2 {carry_in}\no0 {result}\no1 {carry_out}\n")


def add_gate(wordlen, left='x', right='y', output='x+y', has_carry=False):
    carry_name = f'{output}_carry'
    assert left != carry_name and right != carry_name

    adder_aig = aiger.source({carry_name: False})

    lefts = named_indexes(wordlen, 'left')
    rights = named_indexes(wordlen, 'right')
    outputs = named_indexes(wordlen, 'output')

    for lname, rname, oname in zip(lefts, rights, outputs):
        adder_aig >>= _full_adder(
            x=lname,
            y=rname,
            carry_in=carry_name,
            result=oname,
            carry_out=carry_name)

    if not has_carry:
        adder_aig >>= aiger.sink([output + '_carry'])

    return aigbv.AIGBV(
        aig=adder_aig,
        input_map=frozenset([(left, lefts), (right, rights)]),
        output_map=frozenset([(output, outputs)]),
        latch_map=frozenset(),
    )


def negate_gate(wordlen, input='x', output='~x'):
    """Implements two's complement negation."""
    inc = source(wordlen, 1, 'const1') \
        >> add_gate(wordlen, 'tmp', 'const1', output)
    return bitwise_negate(wordlen, input, 'tmp') >> inc


def subtract_gate(wordlen, left='x', right='y', output='x-y'):
    tmp = _fresh()
    return negate_gate(wordlen, right, tmp) \
        >> add_gate(wordlen, left, tmp, output)


def signed_lt_gate(wordlen, left, right, output):
    raise NotImplementedError


def signed_le_gate(wordlen, left, right, output):
    tmp = _fresh()
    return signed_lt_gate(wordlen, right, left, tmp) \
        >> bitwise_negate(1, tmp, output)


def abs_gate():
    raise NotImplementedError
