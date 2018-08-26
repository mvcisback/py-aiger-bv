from typing import Union

import attr
import funcy as fn

from aigerbv import aigbv
from aigerbv import common as cmn


@attr.s(frozen=True, slots=True, cmp=False, auto_attribs=True)
class UnsignedBVExpr:
    aigbv: aigbv.AIGBV

    def __call__(self, inputs=None):
        if inputs is None:
            inputs = {}
        return self.aigbv(inputs)[0][self.output]

    def __getitem__(self, idx: int):
        # TODO: support ranged indexing.
        indexer = cmn.index_gate(self.size, idx, self.output, cmn._fresh())
        return UnsignedBVExpr(self.aigbv >> indexer)

    def concat(self, other):
        combiner = cmn.combine_gate(
            output=cmn._fresh(),
            left_wordlen=self.size, left=self.output,
            right_wordlen=other.size, right=other.output,
        )
        circ = self.aigbv | other.aigbv
        return type(self)(circ >> combiner)

    def repeat(self, times):
        # TODO: support size != 1 via self concatenation.
        assert self.size == 1
        repeater = cmn.repeat(times, self.output, cmn._fresh())
        return type(self)(self.aigbv >> repeater)

    @property
    def output(self):
        return list(self.aigbv.outputs)[0]

    @property
    def inputs(self):
        return self.aigbv.inputs

    @property
    def size(self):
        return len(list(self.aigbv.output_map)[0][1])

    def __invert__(self):
        return _unary_gate(cmn.bitwise_negate, self)

    def __add__(self, other):
        return _binary_gate(cmn.add_gate, self, other)

    def __sub__(self, other):
        return _binary_gate(cmn.subtract_gate, self, other)

    def __and__(self, other):
        return _binary_gate(cmn.bitwise_and, self, other)

    def __matmul__(self, other):
        return _binary_gate(cmn.dot_mod2_gate, self, other)

    def __or__(self, other):
        return _binary_gate(cmn.bitwise_or, self, other)

    def __xor__(self, other):
        return _binary_gate(cmn.bitwise_xor, self, other)

    def __eq__(self, other):
        return _binary_gate(cmn.eq_gate, self, other)

    def __ne__(self, other):
        return _binary_gate(cmn.ne_gate, self, other)

    def __le__(self, other):
        return _binary_gate(cmn.unsigned_le_gate, self, other)

    def __lt__(self, other):
        return _binary_gate(cmn.unsigned_lt_gate, self, other)

    def __ge__(self, other):
        return _binary_gate(cmn.unsigned_ge_gate, self, other)

    def __gt__(self, other):
        return _binary_gate(cmn.unsigned_gt_gate, self, other)

    def __abs__(self):
        return self


class SignedBVExpr(UnsignedBVExpr):
    def __neg__(self):
        return _unary_gate(cmn.negate_gate, self)

    def __le__(self, other):
        return _binary_gate(cmn.signed_le_gate, self, other)

    def __lt__(self, other):
        return _binary_gate(cmn.signed_lt_gate, self, other)

    def __ge__(self, other):
        return _binary_gate(cmn.signed_ge_gate, self, other)

    def __gt__(self, other):
        return _binary_gate(cmn.signed_gt_gate, self, other)

    def __abs__(self):
        return _unary_gate(cmn.abs_gate, self)


Expr = Union[UnsignedBVExpr, SignedBVExpr]


def _binary_gate(gate, expr1, expr2):
    assert expr1.size == expr2.size
    wordlen = expr1.size
    circ1, circ2 = expr1.aigbv, expr2.aigbv
    circ3 = _parcompose(wordlen, circ1, circ2)
    circ3 >>= gate(wordlen=wordlen, output=cmn._fresh(),
                   left=expr1.output, right=expr2.output)
    return type(expr1)(aigbv=circ3)


def _unary_gate(gate, expr):
    circ = gate(expr.size, input=expr.output, output=cmn._fresh())
    return type(expr)(aigbv=expr.aigbv >> circ)


def _fresh_relabel(keys):
    return {k: cmn._fresh() for k in keys}


def _parcompose(wordlen, circ1, circ2):
    inputs_collide = circ1.inputs & circ2.inputs
    outputs_collide = circ1.outputs & circ2.outputs

    if outputs_collide:
        circ1 = circ1['o', _fresh_relabel(circ1.outputs)]
        circ2 = circ2['o', _fresh_relabel(circ2.outputs)]

    if not inputs_collide:
        return circ1 | circ2
    else:
        subs1 = _fresh_relabel(circ1.inputs)
        subs2 = _fresh_relabel(circ2.inputs)
        tee = cmn.tee(wordlen, fn.merge_with(tuple, subs1, subs2))
        return tee >> (circ2['i', subs1] | circ1['i', subs2])


def atom(wordlen: int, val: Union[str, int], signed: bool=True) -> Expr:
    output = cmn._fresh()
    if isinstance(val, str):
        aig = cmn.identity_gate(wordlen, val, output)
    else:
        aig = cmn.source(wordlen, val, output, signed)

    return (SignedBVExpr if signed else UnsignedBVExpr)(aig)
