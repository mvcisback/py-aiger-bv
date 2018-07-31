from typing import NamedTuple, Union

from aigerbv import aigbv
from aigerbv import common as cmn


class BVExpr(NamedTuple):
    aigbv: aigbv.AIGBV
    size: int

    def __call__(self, inputs):
        return self.aigbv(inputs)[0]

    @property
    def output(self):
        return list(self.aigbv.outputs)[0]

    @property
    def inputs(self):
        return self.aigbv.inputs

    def __invert__(self):
        return _unary_gate(cmn.bitwise_negate, self)

    def __neg__(self):
        return _unary_gate(cmn.negate_gate, self)

    def __add__(self, other):
        return _binary_gate(cmn.add_gate, self, other)

    def __sub__(self, other):
        return _binary_gate(cmn.subtract_gate, self, other)

    def __and__(self, other):
        return _binary_gate(cmn.bitwise_and, self, other)

    def __or__(self, other):
        return _binary_gate(cmn.bitwise_or, self, other)

    def __xor__(self, other):
        return _binary_gate(cmn.bitwise_xor, self, other)

    def __eq__(self, other):
        return _binary_gate(cmn.eq_gate, self, other, boolean=True)

    def __ne__(self, other):
        return _binary_gate(cmn.ne_gate, self, other, boolean=True)

    def __le__(self):
        # TODO.
        raise NotImplementedError

    def __lt__(self):
        # TODO.
        raise NotImplementedError

    def __gt__(self):
        # TODO.
        raise NotImplementedError

    def __ge__(self):
        # TODO.
        raise NotImplementedError

    def __abs__(self):
        # TODO.
        raise NotImplementedError
    

def _binary_gate(gate, expr1, expr2, boolean=False):
    assert expr1.size == expr2.size
    wordlen = expr1.size
    circ1, circ2 = expr1.aigbv, expr2.aigbv
    circ3 = _parcompose(wordlen, circ1, circ2)
    circ3 >>= gate(wordlen=wordlen, output=cmn._fresh(),
                   left=expr1.output, right=expr2.output)
    return BVExpr(aigbv=circ3, size=1 if boolean else wordlen)


def _unary_gate(gate, expr):
    circ = gate(expr.size, input=expr.output, output=cmn._fresh())
    return BVExpr(aigbv=expr.aigbv >> circ, size=expr.size)


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


def atom(wordlen: int, val: Union[str, int]) -> BVExpr:
    output = cmn._fresh()
    if isinstance(val, str):
        aig = cmn.identity_gate(wordlen, val, output)
    else:
        aig = cmn.source(wordlen, val, output)

    return BVExpr(aig, wordlen)
