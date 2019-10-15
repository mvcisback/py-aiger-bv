import operator as op
from functools import reduce
from itertools import product, starmap
from uuid import uuid1

import attr
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
        assert N > value >= 0

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

    bw_aigs = (binop([l, r], o) for l, r, o in zip(lefts, rights, outputs))
    aig = reduce(op.or_, bw_aigs)
    return aigbv.AIGBV(
        aig=aig,
        input_map=frozenset([(left, lefts), (right, rights)]),
        output_map=frozenset([(output, outputs)]),
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
    )


def _apply_pairwise(func, seq):
    return list(starmap(func, zip(seq[::2], seq[1::2])))


def reduce_binop(wordlen, inputs, output, op):
    def join(left, right):
        (o1, *_), (o2, *_) = left.outputs, right.outputs # noqa
        return (left | right) >> op(wordlen, o1, o2, _fresh())

    inputs = list(inputs)
    queue = [identity_gate(wordlen, i) for i in inputs]
    while len(queue) > 1:
        queue = _apply_pairwise(join, queue)

    circ = queue[0]
    if len(inputs) & 1:  # Odd number of elements.
        circ = join(circ, identity_gate(wordlen, inputs[-1]))
    return circ


def is_nonzero_gate(wordlen, input='x', output='is_nonzero'):
    inputs = named_indexes(wordlen, input)
    outputs = named_indexes(1, output)
    return aigbv.AIGBV(
        aig=aiger.or_gate(inputs, outputs[0]),
        input_map=frozenset([(input, inputs)]),
        output_map=frozenset([(output, outputs)]),
    )


def ne_gate(wordlen, left='x', right='y', output='x!=y'):
    return bitwise_xor(wordlen, left, right, left+right) >> \
        is_nonzero_gate(wordlen, left+right, output)


def is_zero_gate(wordlen, input='x', output='is_zero'):
    return is_nonzero_gate(wordlen, input, input+'out') >> \
        bitwise_negate(1, input+'out', output)


def eq_gate(wordlen, left='x', right='y', output='x=y'):
    return ne_gate(wordlen, left, right, left+right) >> \
        bitwise_negate(1, left+right, output)


def source(wordlen, value, name='x', signed=True):
    names = named_indexes(wordlen, name)
    bits = encode_int(wordlen, value, signed)
    aig = aiger.source({name: bit for name, bit in zip(names, bits)})
    return aigbv.AIGBV(
        aig=aig,
        output_map=frozenset([(name, names)]),
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
    )


def repeat(wordlen, input, output=None):
    if output is None:
        output = input

    outputs = named_indexes(wordlen, output)
    return aigbv.AIGBV(
        aig=aiger.tee({input: outputs}),
        input_map=frozenset([(input, (input,))]),
        output_map=frozenset([(input, outputs)]),
    )


def identity_gate(wordlen, input='x', output=None):
    if output is None:
        output = input

    inputs = named_indexes(wordlen, input)
    outputs = named_indexes(wordlen, output)
    return aigbv.AIGBV(
        aig=aiger.identity(inputs=inputs, outputs=outputs),
        input_map=frozenset([(input, inputs)]),
        output_map=frozenset([(output, outputs)]),
    )


def reverse_gate(wordlen, input='x', output='rev(x)'):
    circ = identity_gate(wordlen, input, output)
    output_map = frozenset(
        (k, tuple(reversed(vs))) for k, vs in circ.output_map
    )
    return attr.evolve(circ, output_map=output_map)


def combine_gate(left_wordlen, left, right_wordlen, right, output):
    lefts = named_indexes(left_wordlen, left)
    rights = named_indexes(right_wordlen, right)

    circ = identity_gate(left_wordlen, left, left) \
        | identity_gate(right_wordlen, right, right)
    return attr.evolve(circ, output_map=frozenset([(output, lefts+rights)]))


def split_gate(input, left_wordlen, left, right_wordlen, right):
    inputs = named_indexes(left_wordlen + right_wordlen, input)
    lefts, rights = inputs[:left_wordlen], inputs[left_wordlen:]

    circ = identity_gate(left_wordlen + right_wordlen, input, input)
    output_map = frozenset([(left, lefts), (right, rights)])
    return attr.evolve(circ, output_map=output_map)


def sink(wordlen, inputs):
    blasted_inputs = [named_indexes(wordlen, i) for i in inputs]
    return aigbv.AIGBV(
        aig=aiger.sink(fn.lcat(blasted_inputs)),
        input_map=frozenset(fn.lzip(inputs, blasted_inputs)),
    )


def __full_adder():
    x, y, cin = map(aiger.atom, ('x', 'y', 'ci'))
    tmp = x ^ y
    res = (tmp ^ cin)
    cout = ((tmp & cin) | (x & y))
    circ = res.aig | cout.aig
    relabels = {res.output: 'res', cout.output: 'co'}
    return (circ)['o', relabels]


FULL_ADDER_GADGET = __full_adder()


def _full_adder(x, y, carry_in, result, carry_out):
    relabels = {'x': x, 'y': y, 'ci': carry_in, 'res': result, 'co': carry_out}
    return FULL_ADDER_GADGET['i', relabels]['o', relabels]


def even_popcount_gate(wordlen, input, output):
    inputs = named_indexes(wordlen, input)
    return aigbv.AIGBV(
        aig=aiger.parity_gate(inputs, output),
        input_map=frozenset([(input, inputs)]),
        output_map=frozenset([(output, (output,))])
    )


def dot_mod2_gate(wordlen, left='x', right='y', output='x@y'):
    return bitwise_and(wordlen, left, right, 'tmp') >> \
        even_popcount_gate(wordlen, 'tmp', output)


def add_gate(wordlen, left='x', right='y', output='x+y', has_carry=False):
    carry_name = f'{output}_carry'
    assert left != carry_name and right != carry_name

    adder_aig = aiger.source({carry_name: False})

    lefts = named_indexes(wordlen, left)
    rights = named_indexes(wordlen, right)
    outputs = named_indexes(wordlen, output)

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


def index_gate(wordlen, idx, input, output=None):
    assert 0 <= idx < wordlen
    if output is None:
        output = input

    inputs = named_indexes(wordlen, input)
    outputs = (inputs[idx],)
    aig = aiger.sink(set(inputs) - set(outputs)) \
        | aiger.identity(outputs)
    return aigbv.AIGBV(
        aig=aig,
        input_map=frozenset([(input, inputs)]),
        output_map=frozenset([(output, outputs)]),
    )


def unsigned_lt_gate(wordlen, left, right, output):
    left_names = named_indexes(wordlen, left)
    right_names = named_indexes(wordlen, right)

    lefts = map(aiger.atom, left_names)
    rights = map(aiger.atom, right_names)

    def test_bit(expr, lr):
        l, r = lr
        expr &= ~(l ^ r)  # l == r.
        expr |= ~l & r  # l < r.
        return expr

    expr = reduce(test_bit, zip(lefts, rights), aiger.atom(False))
    return aigbv.AIGBV(
        aig=expr.aig,
        input_map=frozenset([(left, left_names), (right, right_names)]),
        output_map=frozenset([(output, (expr.output,))]),
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
    msb = _fresh()
    msb2 = _fresh()
    get_msb = index_gate(wordlen, wordlen-1, msb, msb)
    get_msb2 = index_gate(wordlen, wordlen-1, msb2, msb2)

    circ1 = get_msb >> combine_gate(wordlen, left, 1, msb, left)
    circ2 = get_msb2 >> combine_gate(wordlen, right, 1, msb2, right)

    return tee(wordlen, {left: (left, msb), right: (right, msb2)}) \
        >> (circ1 | circ2) \
        >> subtract_gate(wordlen+1, left, right, output) \
        >> index_gate(wordlen+1, wordlen, output)


def signed_gt_gate(wordlen, left, right, output):
    return signed_lt_gate(wordlen, right, left, output)


def signed_ge_gate(wordlen, left, right, output):
    return signed_lt_gate(wordlen, left, right, 'tmp') \
        >> bitwise_negate(1, 'tmp', output)


def signed_le_gate(wordlen, left, right, output):
    return signed_gt_gate(wordlen, left, right, 'tmp') \
        >> bitwise_negate(1, 'tmp', output)


def left_shift_gate(wordlen, shift, input='x', output='x'):
    return reverse_gate(wordlen, input, 'tmp') \
        >> logical_right_shift_gate(wordlen, shift, 'tmp', 'tmp') \
        >> reverse_gate(wordlen, 'tmp', output)


def _right_shift_gate(wordlen, shift, shiftin, input='x', output='x'):
    assert 0 <= shift
    shift = min(shift, wordlen)

    return repeat(shift, shiftin) \
        >> split_gate(input, shift, 'drop', wordlen - shift, output) \
        >> sink(shift, ['drop']) \
        >> combine_gate(wordlen - shift, output, shift, shiftin, output)


def logical_right_shift_gate(wordlen, shift, input='x', output='x'):
    tmp = _fresh()
    return source(1, 0, tmp) \
        >> _right_shift_gate(wordlen, shift, tmp, input, output)


def arithmetic_right_shift_gate(wordlen, shift, input, output):
    shiftin = _fresh()
    circ = index_gate(wordlen, wordlen - 1, shiftin, shiftin) \
        | identity_gate(wordlen, input, input)
    return tee(wordlen, {input: (input, shiftin)}) >> circ \
        >> _right_shift_gate(wordlen, shift, shiftin, input, output)


def abs_gate(wordlen, input, output):
    tmp, tmp2 = _fresh(), _fresh()
    return tee(wordlen, {input: (input, tmp)}) \
        >> arithmetic_right_shift_gate(wordlen, wordlen - 1, tmp, tmp) \
        >> tee(wordlen, {tmp: (tmp, tmp2)}) \
        >> add_gate(wordlen, input, tmp, output) \
        >> bitwise_xor(wordlen, output, tmp2, output)


def lookup(inlen, outlen, mapping, input, output, *,
           in_signed=True, out_signed=True):
    # [(i = a1) -> b] /\ [(i = a2) -> c] /\ [(i = a3) -> d]
    def guard(key, val):
        circ = identity_gate(inlen, _fresh(), 'input') \
            | source(inlen, key, 'key', in_signed) \
            | source(outlen, val, 'val', out_signed)
        circ >>= ne_gate(inlen, 'input', 'key', 'neq')
        circ >>= repeat(outlen, 'neq', 'neq')
        circ >>= bitwise_or(outlen, 'neq', 'val', _fresh())
        return circ

    circ = reduce(op.or_, starmap(guard, mapping.items()))
    circ = tee(inlen, {input: circ.inputs}) >> circ
    if len(circ.outputs) > 1:
        circ >>= reduce_binop(outlen, circ.outputs, output, bitwise_and)
    out, *_ = circ.outputs
    return circ['o', {out: output}]


def kmodels(wordlen: int, k: int, input=None, output=None):
    """Return a circuit taking a wordlen bitvector where only k
    valuations return True. Uses encoding from [1].

    Note that this is equivalent to (~x < k).
    - TODO: Add automated simplification so that the circuits
            are equiv.

    [1]: Chakraborty, Supratik, et al. "From Weighted to Unweighted Model
    Counting." IJCAI. 2015.
    """

    assert 0 <= k < 2**wordlen
    if output is None:
        output = _fresh()

    if input is None:
        input = _fresh()

    input_names = named_indexes(wordlen, input)
    atoms = map(aiger.atom, input_names)

    active = False
    expr = aiger.atom(False)
    for atom, bit in zip(atoms, encode_int(wordlen, k, signed=False)):
        active |= bit
        if not active:  # Skip until first 1.
            continue
        expr = (expr | atom) if bit else (expr & atom)

    return aigbv.AIGBV(
        aig=expr.aig,
        input_map=frozenset([(input, tuple(input_names))]),
        output_map=frozenset([(output, (expr.output,))]),
    )
