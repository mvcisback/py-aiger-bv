import operator as op
from functools import reduce
from itertools import product
from uuid import uuid1

import aiger
import funcy as fn

from aigerbv import aigbv


def _fresh():
    return str(uuid1())


def _name_idx(root, i):
    return f"{root}[{i}]"


def named_indexes(wordlen, root):
    return tuple(_name_idx(root, i) for i in range(wordlen))


def encode_int(wordlen, value, signed=True):
    N = 1 << wordlen
    if signed:
        N2 = 1 << (wordlen - 1)
        assert N2 > value >= -N2
    else:
        assert N - 1 > value >= 0

    if value < 0:
        value = N + value

    return [bool((value >> i) & 1) for i in range(wordlen)]    


def decode_int(bits, signed=True):
    # Interpret result
    last = bits[-1]*(1 << (len(bits) - 1))
    last *= -1 if signed else 1
    return sum(val << idx for idx, val in enumerate(bits[:-1])) + last


def bitwise_binop(binop, wordlen, left='x', right='y', output='x&y'):
    lefts = named_indexes(wordlen, left)
    rights = named_indexes(wordlen, right)
    outputs = named_indexes(wordlen, output)

    aig = reduce(op.or_,
        (binop([l, r], o) for l, r, o in zip(lefts, rights, outputs))
    )
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


def source(wordlen, value, name='x', signed=True):
    names = named_indexes(wordlen, name)
    bits = encode_int(wordlen, value, signed)
    aig = aiger.source({name: bit for name, bit in zip(names, bits)})
    return aigbv.AIGBV(
        aig=aig,
        input_map=frozenset(),
        output_map=frozenset([(name, names)]),
        latch_map=frozenset(),
    )


def tee(wordlen, iomap):
    input_map = frozenset((i, named_indexes(wordlen, i)) for i in iomap)
    output_map = frozenset(
        (o, named_indexes(wordlen, o)) for o in fn.cat(iomap.values())
    )
    blasted_iomap = fn.merge(
        *({_name_idx(iname, idx): [_name_idx(o, idx) for o in iomap[iname]]}
          for iname, idx in product(iomap, range(wordlen)))
    )

    return aigbv.AIGBV(
        aig=aiger.tee(blasted_iomap),
        input_map=input_map,
        output_map=output_map,
        latch_map=frozenset()
    )


def identity_gate(wordlen, input='x', output='x'):
    inputs = named_indexes(wordlen, input)
    outputs = named_indexes(wordlen, output)
    return aigbv.AIGBV(
        aig=aiger.identity(inputs=inputs, outputs=outputs),
        input_map=frozenset([(input, inputs)]),
        output_map=frozenset([(output, outputs)]),
        latch_map=frozenset()
    )


def reverse_gate(wordlen, input='x', output='rev(x)'):
    circ = identity_gate(wordlen, input, output)
    return identity_gate(wordlen, input, output)._replace(
        output_map=frozenset((k, reversed(vs)) for k, vs in circ.output_map))


def combine_gate(left_wordlen, left, right_wordlen, right, output):
    lefts = named_indexes(left_wordlen, left)
    rights = named_indexes(right_wordlen, right)
    outputs = named_indexes(left_wordlen + right_wordlen, output)

    aigbv = identity_gate(left_wordlen, left, left) \
        | identity_gate(right_wordlen, right, right)
    return aigbv._replace(
        output_map=frozenset([(output, lefts+rights)])
    )


def split_gate(input, left_wordlen, left, right_wordlen, right):
    inputs = named_indexes(left_wordlen + right_wordlen, input)
    lefts = named_indexes(left_wordlen, left)
    rights = named_indexes(right_wordlen, right)

    aigbv = identity_gate(left_wordlen, left, left) \
        | identity_gate(right_wordlen, right, right)
    return aigbv._replace(
        input_map=frozenset([(input, lefts+rights)])
    )


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


def inc_gate(wordlen, input='x', output='inc'):
    tmp = _fresh()
    return source(wordlen, 1, tmp) >> add_gate(wordlen, tmp, input, output)


def negate_gate(wordlen, input='x', output='~x'):
    """Implements two's complement negation."""
    neg = bitwise_negate(wordlen, input, "tmp")
    inc = inc_gate(wordlen, "tmp", output)
    return neg >> inc


def dec_gate(wordlen, input='x', output='inc'):
    tmp = _fresh()
    return source(wordlen, -1, tmp) >> add_gate(wordlen, tmp, input, output)


def subtract_gate(wordlen, left='x', right='y', output='x-y'):
    tmp = _fresh()
    return negate_gate(wordlen, right, tmp) \
        >> add_gate(wordlen, left, tmp, output)


def unsigned_lt_gate(wordlen, left, right, output):
    # out = ~x /\ y /\ active; active' = (x == y) /\ active .
    bit_comparer = aiger.bit_flipper(['x']) \
        >> aiger.and_gate(['x', 'y'], 'tmp') \
        >> aiger.and_gate(['active', 'tmp'], 'out')
    check_active = aiger.parity_gate(['x', 'y'], 'tmp') \
        >> aiger.bit_flipper(['tmp']) \
        >> aiger.and_gate(['tmp', 'active'], 'active')
    _gadget = bit_comparer | check_active

    def gadget(params):
        x, y, active, out = params
        subs = {'x': x, 'y':y, 'active': active, 'out': out}
        return _gadget[('i', subs)][('o', subs)]

    # Create gadget for each pair of bits.
    active = _fresh()
    lefts = named_indexes(wordlen, left)
    rights = named_indexes(wordlen, right)
    outputs = named_indexes(wordlen, output)
    actives = fn.repeat(active)

    # Work GSB to LSB.
    zipped = zip(lefts[::-1], rights[::-1], actives, outputs[::-1])
    gadgets = map(gadget, zipped)

    # Sequentially compose gadgets and take disjunction.
    aig = aiger.source({active: True}) \
        >> reduce(op.rshift, gadgets) \
        >> aiger.or_gate(outputs, output) \
        >> aiger.sink([active])

    return aigbv.AIGBV(
        aig=aig,
        input_map=frozenset([(left, lefts), (right, rights)]),
        output_map=frozenset([(output, (output,))]),
        latch_map=frozenset(),
    )


def unsigned_le_gate(wordlen, left, right, output):
    fresh = [_fresh() for _ in range(4)]
    lt = unsigned_lt_gate(wordlen, fresh[0], fresh[2], 'lt')
    eq = eq_gate(wordlen, fresh[1], fresh[3], 'eq')
    return tee(wordlen, {left: fresh[:2], right: fresh[2:]}) \
        >> (lt | eq) \
        >> aigbv.aig2aigbv(aiger.or_gate(['lt', 'eq'], output))


def unsigned_gt_gate(wordlen, left, right, output):
    return unsigned_le_gate(wordlen, left, right, 'le') \
        >> bitwise_negate(1, 'le', output)


def unsigned_ge_gate(wordlen, left, right, output):
    return unsigned_lt_gate(wordlen, left, right, 'ge') \
        >> bitwise_negate(1, 'ge', output)


def signed_lt_gate(wordlen, left, right, output):
    raise NotImplementedError


def signed_le_gate(wordlen, left, right, output):
    raise NotImplementedError


def abs_gate():
    raise NotImplementedError
