import operator as op
from functools import reduce

import aiger
import funcy as fn

from aigerbv import aigbv


def _named_indexes(wordlen, root):
    return tuple(f"{root}[{i}]" for i in range(wordlen))


def bitwise_binop(binop, wordlen, left='x', right='y', output='x&y'):
    lefts = _named_indexes(wordlen, left)
    rights = _named_indexes(wordlen, right)
    outputs = _named_indexes(wordlen, output)

    aig = reduce(
        op.or_,
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
    inputs = _named_indexes(wordlen, input)
    outputs = _named_indexes(wordlen, output)
    return aigbv.AIGBV(
        aig=aiger.bit_flipper(inputs=inputs, outputs=outputs),
        input_map=frozenset([(input, inputs)]),
        output_map=frozenset([(output, outputs)]),
        latch_map=frozenset(),
    )


def is_nonzero_gate(wordlen, input='x', output='is_nonzero'):
    inputs = _named_indexes(wordlen, input)
    outputs = _named_indexes(1, output)
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
    names = _named_indexes(wordlen, name)
    return aigbv.AIGBV(
        aig=aiger.source({name: bit for name, bit in zip(names, bits)}),
        input_map=frozenset([(name, names)]),
        output_map=frozenset([(name, names)]),
        latch_map=frozenset(),
    )


def _full_adder(x, y, carry_in, result, carry_out):
    # TODO: Rewrite in aiger. 
    return aiger.parse(
        "aag 10 3 0 2 7\n2\n4\n6\n18\n21\n8 4 2\n10 5 3\n"
        "12 11 9\n14 12 6\n16 13 7\n18 17 15\n20 15 9\n"
        f"i0 {x}\ni1 {y}\ni2 {carry_in}\no0 {result}\no1 {carry_out}\n")


def add_gate(wordlen, output='x+y', left='x', right='y', has_carry=False):
    carry_name = f'{output}_carry'
    assert left != carry_name and right != carry_name

    adder_aig = aiger.source({carry_name: False})

    lefts = _named_indexes(wordlen, 'left')
    rights = _named_indexes(wordlen, 'right')
    outputs = _named_indexes(wordlen, 'output')

    for lname, rname, oname in zip(lefts, rights, outputs):
        adder_aig >>= _full_adder(
            x=lname,
            y=rname,
            carry_in=carry_name,
            result=oname,
            carry_out=carry_name
        )

    if not has_carry:
        adder_aig >>= aiger.sink([output + '_carry'])
    
    return aigbv.AIGBV(
        aig=adder_aig,
        input_map=frozenset([(left, lefts), (right, rights)]),
        output_map=frozenset([(output, outputs)]),
        latch_map=frozenset(),
    )


def leq_gate():
    raise NotImplementedError


def geq_gate():
    raise NotImplementedError


def lt_gate():
    raise NotImplementedError


def gt_gate():
    raise NotImplementedError


def abs_gate():
    raise NotImplementedError


def concat_gate(wordlen, input='x', output='rev(x)'):
    raise NotImplementedError


def repeat_gate(wordlen, input='x', output='rev(x)'):
    raise NotImplementedError


def reverse_gate(wordlen, input='x', output='rev(x)'):
    raise NotImplementedError


def negate_gate(wordlen, input='x', output='~x'):
    raise NotImplementedError


def subtract_gate(wordlen, output='x-y', left='x', right='y'):
    raise NotImplementedError


def shift_gate(wordlen, shift, input='x', output='x', signed=True):
    raise NotImplementedError
