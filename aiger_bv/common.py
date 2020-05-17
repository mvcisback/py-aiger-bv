import operator as op
from collections import defaultdict
from functools import reduce
from itertools import starmap
from uuid import uuid1

import attr
import aiger
import funcy as fn

from aiger_bv import aigbv
from aiger_bv.bundle import BundleMap, Bundle


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
    imap = BundleMap({left: wordlen, right: wordlen})
    omap = BundleMap({output: wordlen})

    names = zip(imap[left], imap[right], omap[output])
    return aigbv.AIGBV(
        imap=imap, omap=omap,
        aig=reduce(op.or_, (binop([lft, rht], o) for lft, rht, o in names)),
    )


def bitwise_and(wordlen, left='x', right='y', output='x&y'):
    return bitwise_binop(aiger.and_gate, wordlen, left, right, output)


def bitwise_or(wordlen, left='x', right='y', output='x&y'):
    return bitwise_binop(aiger.or_gate, wordlen, left, right, output)


def bitwise_xor(wordlen, left='x', right='y', output='x&y'):
    return bitwise_binop(aiger.parity_gate, wordlen, left, right, output)


def bitwise_negate(wordlen, input='x', output='not x'):
    imap, omap = BundleMap({input: wordlen}), BundleMap({output: wordlen})
    return aigbv.AIGBV(
        imap=imap, omap=omap,
        aig=aiger.bit_flipper(inputs=imap[input], outputs=omap[output]),
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
    imap, omap = BundleMap({input: wordlen}), BundleMap({output: 1})
    return aigbv.AIGBV(
        imap=imap, omap=omap,
        aig=aiger.or_gate(imap[input], omap[output][0]),
    )


def ne_gate(wordlen, left='x', right='y', output='x!=y'):
    return bitwise_xor(wordlen, left, right, left+right) >> \
        is_nonzero_gate(wordlen, left+right, output)


def is_zero_gate(wordlen, input='x', output='is_zero'):
    return is_nonzero_gate(wordlen, input, input+'out') >> \
        bitwise_negate(1, input + 'out', output)


def eq_gate(wordlen, left='x', right='y', output='x=y'):
    return ne_gate(wordlen, left, right, left + right) >> \
        bitwise_negate(1, left + right, output)


def source(wordlen, value, name='x', signed=True):
    if isinstance(value, int):
        value = encode_int(wordlen, value, signed)

    omap = BundleMap({name: wordlen})
    aig = aiger.source({name: bit for name, bit in zip(omap[name], value)})
    return aigbv.AIGBV(aig=aig, omap=omap)


def tee(wordlen, iomap):
    imap = BundleMap({i: wordlen for i in iomap})
    omap = BundleMap({o: wordlen for o in fn.cat(iomap.values())})

    blasted = defaultdict(list)

    for i, outs in iomap.items():
        for o in outs:
            for k, v in zip(imap[i], omap[o]):
                blasted[k].append(v)

    return aigbv.AIGBV(imap=imap, omap=omap, aig=aiger.tee(blasted))


def repeat(wordlen, input, output=None):
    if output is None:
        output = input

    imap, omap = BundleMap({input: 1}), BundleMap({output: wordlen})
    return aigbv.AIGBV(
        imap=imap, omap=omap,
        aig=aiger.tee({imap[input][0]: list(omap[output])}),
    )


def identity_gate(wordlen, input='x', output=None):
    if output is None:
        output = input

    imap, omap = BundleMap({input: wordlen}), BundleMap({output: wordlen})
    return aigbv.AIGBV(
        imap=imap, omap=omap,
        aig=aiger.identity(inputs=imap[input], outputs=omap[output]),
    )


def reverse_gate(wordlen, input='x', output='rev(x)'):
    circ = identity_gate(wordlen, input, output=output)

    tmp, obdl = Bundle(_fresh(), wordlen), Bundle(output, wordlen)

    aig = circ.aig['o', dict(zip(obdl, reversed(tmp)))]
    aig = aig['o', dict(zip(tmp, obdl))]

    return attr.evolve(circ, aig=aig)


def combine_gate(left_wordlen, left, right_wordlen, right, output):
    circ = identity_gate(left_wordlen, left, left) \
        | identity_gate(right_wordlen, right, right)

    omap1 = circ.omap
    relabels = {
        k: f"{left}[{i + left_wordlen}]" for i, k in enumerate(omap1[right])
    }
    omap2 = BundleMap({left: left_wordlen + right_wordlen})
    circ = attr.evolve(circ, omap=omap2, aig=circ.aig['o', relabels])
    return circ if left == output else circ['o', {left: output}]


def split_gate(input, left_wordlen, left, right_wordlen, right):
    omap = BundleMap({left: left_wordlen, right: right_wordlen})

    circ = identity_gate(left_wordlen + right_wordlen, input, input)
    relabels = fn.merge(
        dict(zip(circ.omap[input][:left_wordlen], omap[left])),
        dict(zip(circ.omap[input][left_wordlen:], omap[right])),
    )

    return attr.evolve(circ, omap=omap, aig=circ.aig['o', relabels])


def sink(wordlen, inputs):
    imap = BundleMap({i: wordlen for i in inputs})
    return aigbv.AIGBV(imap=imap, aig=aiger.sink(fn.lmapcat(imap.get, inputs)))


def even_popcount_gate(wordlen, input, output):
    imap, omap = BundleMap({input: wordlen}), BundleMap({output: 1})
    return aigbv.AIGBV(
        imap=imap, omap=omap,
        aig=aiger.parity_gate(imap[input], omap[output][0]),
    )


def dot_mod2_gate(wordlen, left='x', right='y', output='x@y'):
    return bitwise_and(wordlen, left, right, 'tmp') >> \
        even_popcount_gate(wordlen, 'tmp', output)


def __full_adder():
    x, y, cin = aiger.atoms('x', 'y', 'ci')
    tmp = x ^ y
    res = (tmp ^ cin)
    cout = ((tmp & cin) | (x & y))
    circ = res.aig | cout.aig
    relabels = {res.output: 'res', cout.output: 'co'}
    return (circ)['o', relabels]


FULL_ADDER_GADGET = __full_adder()


def _full_adder(x, y, carry_in, result, carry_out):
    irelabels = {'x': x, 'y': y, 'ci': carry_in}
    orelabels = {'res': result, 'co': carry_out}
    return FULL_ADDER_GADGET.relabel('input', irelabels) \
                            .relabel('output', orelabels)


def add_gate(wordlen, left='x', right='y', output='x+y', has_carry=False):
    carry_name = f'{output}_carry'
    assert left != carry_name and right != carry_name

    adder_aig = aiger.source({carry_name: False})

    imap = BundleMap({left: wordlen, right: wordlen})
    omap = BundleMap(
        {output: wordlen, has_carry: 1} if has_carry else {output: wordlen}
    )

    for lname, rname, oname in zip(imap[left], imap[right], omap[output]):
        adder_aig >>= _full_adder(
            x=lname,
            y=rname,
            carry_in=carry_name,
            result=oname,
            carry_out=carry_name)

    if not has_carry:
        adder_aig >>= aiger.sink([output + '_carry'])

    return aigbv.AIGBV(imap=imap, omap=omap, aig=adder_aig)


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

    imap, omap = BundleMap({input: wordlen}), BundleMap({output: 1})
    inputs, outputs = imap[input], (imap[input][idx],)

    aig = aiger.sink(set(inputs) - set(outputs)) | aiger.identity(outputs)
    relabels = {outputs[0]: omap[output][0]}
    return aigbv.AIGBV(imap=imap, omap=omap, aig=aig['o', relabels])


def unsigned_lt_gate(wordlen, left, right, output):
    omap = BundleMap({output: 1})
    imap = BundleMap({left: wordlen, right: wordlen})

    lefts = map(aiger.atom, imap[left])
    rights = map(aiger.atom, imap[right])

    def test_bit(expr, lr):
        l, r = lr
        expr &= ~(l ^ r)  # l == r.
        expr |= ~l & r  # l < r.
        return expr

    expr = reduce(test_bit, zip(lefts, rights), aiger.atom(False))
    aig = expr.aig['o', {expr.output: omap[output][0]}]
    return aigbv.AIGBV(imap=imap, omap=omap, aig=aig)


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

    imap, omap = BundleMap({input: wordlen}), BundleMap({output: 1})
    atoms = map(aiger.atom, imap[input])

    active = False
    expr = aiger.atom(False)
    for atom, bit in zip(atoms, encode_int(wordlen, k, signed=False)):
        active |= bit
        if not active:  # Skip until first 1.
            continue
        expr = (expr | atom) if bit else (expr & atom)

    aig = expr.aig['o', {expr.output: omap[output][0]}]
    aig |= aiger.sink(imap[input])
    return aigbv.AIGBV(imap=imap, omap=omap, aig=aig)
