<figure>
  <img src="logo_text.svg" alt="py-aiger-bv logo" width=300px>
  <figcaption>pyAiger-BV: Extension of pyAiger for manipulating
    sequential bitvector circuits.</figcaption>
</figure>


[![Build Status](https://cloud.drone.io/api/badges/mvcisback/py-aiger-bv/status.svg)](https://cloud.drone.io/mvcisback/py-aiger-bv)
[![Docs](https://img.shields.io/badge/API-link-color)](https://mvcisback.github.io/py-aiger-bv)
[![PyPI version](https://badge.fury.io/py/py-aiger-bv.svg)](https://badge.fury.io/py/py-aiger-bv)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
 
# Table of Contents
- [About](#about-py-aiger-bv)
- [Installation](#installation)
- [BitVector Expr DSL](#bitvector-expression-dsl)
- [Sequential Circuit DSL](#sequential-circuit-dsl)

# About Py-Aiger-BV

This library provides word level abstractions on top of
[py-aiger](https://github.com/mvcisback/py-aiger). This is done by the
`AIGBV` which groups inputs, outputs, and latches into named
**ordered** sequences, e.g. bitvectors.

The resulting objects can be turned into `AIG`s where each input,
output, or latches name has its index appended to its name. For example,
an bitvector input, `'x'` with 3 bits becomes inputs `'x[0]', 'x[1]', 'x[3]'`


# Installation

If you just need to use `aiger_bv`, you can just run:

`$ pip install py-aiger-bv`

For developers, note that this project uses the
[poetry](https://poetry.eustace.io/) python package/dependency
management tool. Please familarize yourself with it and then
run:

`$ poetry install`

# BitVector Expression DSL

As in py-aiger, when writing combinatorial circuits, the Sequential
Circuit DSL can be somewhat clumsy. For this common usecase, we have
developed the BitVector Expression DSL. This DSL actually consists of
two DSLs for signed and unsigned BitVectors.  All circuits generated
this way have a single output word. We use a **big-endian** encoding
where the most significant digit is the first element of the tuple
representing the word. For signed numbers, two's complement is used.

```python
import aiger_bv

# Create 16 bit variables.
x = aiger_bv.atom(16, 'x')
y = aiger_bv.atom(16, 'y', signed=True)  # Signed by default.
z = aiger_bv.uatom(16, 'z')              # Equiv to signed=False.

# bitwise ops.
expr1 = x & y  # Bitwise and.
expr2 = x | y  # Bitwise or.
expr3 = x ^ y  # Bitwise xor.
expr4 = ~x  # Bitwise negation.

# arithmetic
expr5 = x + y
expr6 = x - y
expr7 = x << y
expr8 = x >> y  # logical if unsigned, arithmetic if signed.
expr9 = -x  # Arithmetic negation. Only defined for signed expr.
expr10 = abs(x)
expr11 = x @ y  # inner product of bitvectors mod 2 (+ is xor).

# comparison
expr12 = x == y
expr13 = x != y
expr14 = x < y
expr15 = x <= y
expr16 = x > y
expr17 = x >= y

# Atoms can be constants.
expr18 = x & aiger_bv.atom(16, 3)
expr19 = x & aiger_bv.atom(16, 0xff)

# BitVector expressions can be concatenated.
expr20 = x.concat(y)

# Particular bits can be indexed to create new expressions.
expr21 = x[1]

# Single bit expressions can be repeated.
expr22 = x[1].repeat(10)

# And you can inspect the AIGBV if needed.
circ = x.aigbv

# And you can inspect the AIG if needed.
circ = x.aigbv.aig

# And of course, you can get a BoolExpr from a single output aig.
expr = aiger_bv.UnsignedBVExpr(circ)
```

# Sequential Circuit DSL

py-aiger-bv's Sequential Circuit DSL implements the same basic api as
py-aiger's Sequential Circuit DSL, but operates at the (variable
length) word level rather than the bit level.

```python
import aiger
import aiger_bv


circ = ... # Create a circuit (see below).

# We assume this circuit has word level
# inputs: x,y, outputs: z, w, q, latches: a, b
assert circ.inputs == {'x', 'y'}
assert circ.outputs == {'z', 'w', 'q'}
assert circ.latches == {'a', 'b'}
```

## Sequential composition
```python
circ3 = circ1 >> circ2
```

## Parallel composition
```python
circ3 = circ1 | circ2
```

## Adding Feedback (inserts a delay)
```python
# Connect output y to input x with delay (initialized to True).
# (Default initialization is False.)
cir2 = circ.feedback(
    inputs=['x'],
    outputs=['y'],
    initials=[True],
    keep_outputs=True
)
```

## Relabeling
```python
# Relabel input 'x' to 'z'.
circ2 = circ['i', {'x': 'z'}]

# Relabel output 'y' to 'w'.
circ2 = circ['o', {'y': 'w'}]

# Relabel latches 'l1' to 'l2'.
circ2 = circ['l', {'l1': 'l2'}]
```

## Evaluation
```python
# Combinatoric evaluation.
circ(inputs={'x':(True, False, True), 'y': (True, False)})

# Sequential evaluation.
circ.simulate([
        {'x': (True, False, True), 'y': (True, False)},
        {'x': (False, False, True), 'y': (False, False)},
    ])

# Simulation Coroutine.
sim = circ.simulator()  # Coroutine
next(sim)  # Initialize
print(sim.send({'x': (True, False, True), 'y': (True, False)}))
print(sim.send({'x': (False, False, True), 'y': (False, False)}))


# Unroll
circ2 = circ.unroll(steps=10, init=True)
```

## aiger.AIG to aiger.AIGBV

There are two main ways to take an object `AIG` from `aiger` and
convert it into an `AIGBV` object. The first is the `aig2aigbv`
command which simply makes all inputs words of size 1.


```python
# Create aiger_bv.AIGERBV object from aiger.AIG object.
circ  = ... # Some aiger.AIG object
word_circ = aiger_bv.aig2aigbv(circ)  # aiger_bv.AIGBV object


```

## Gadget Library

### General Manipulation

```python
# Copy outputs 'x' and 'y' to 'w1, w2' and 'z1, z2'.
circ1 = circ >> aiger_bv.tee(wordlen=3, iomap={
        'x': ('w1', 'w2'),
        'y': ('z1', 'z2')
    })

# Take 1 bit output, 'x', duplicate it 5 times, and group into
# a single 5-length word output, 'y'.
circ2 = circ >> aiger_bv.repeat(wordlen=5, input='x', output='z')

# Reverse order of a word.
circ3 = circ >> aiger_bv.reverse_gate(wordlen=5, input='x', output='z')

# Sink and Source circuits (see encoding section for encoding details).
## Always output binary encoding for 15. 
circ4 = aiger_bv.source(wordlen=4, value=15, name='x', signed=False)

## Absorb output 'y'
circ5 = circ >> aiger_bv.sink(wordlen=4, inputs=['y'])

# Identity Gate
circ6 = circ >> aiger_bv.identity_gate(wordlen=3, input='x')

# Combine/Concatenate words
circ7 = circ >> aiger_bv.combine_gate(
    left_wordlen=3, left='x',
    right_wordlen=3, right='y',
    output='z'
)

# Split words
circ8 = circ >> aiger_bv.split_gate(
    input='x',
    left_wordlen=1, left='z',
    right_wordlen=2, right='w'
)

# Select single index of circuit and make it a wordlen=1 output.
circ9 = circ >> aiger_bv.index_gate(wordlen=3, idx=1, input='x', output='x1')
```

## Bitwise Operations

- `aiger_bv.bitwise_and(3, left='x', right='y', output='x&y')`
- `aiger_bv.bitwise_or(3, left='x', right='y', output='x|y')`
- `aiger_bv.bitwise_xor(3, left='x', right='y', output='x^y')`
- `aiger_bv.bitwise_negate(3, left='x', output='~x')`

## Arithmetic

- `aiger_bv.add_gate(3, left='x', right='y', output='x+y')`
- `aiger_bv.subtract_gate_gate(3, left='x', right='y', output='x-y')`
- `aiger_bv.inc_gate(3, left='x', output='x+1')`
- `aiger_bv.dec_gate(3, left='x', output='x+1')`
- `aiger_bv.negate_gate(3, left='x', output='-x')`
- `aiger_bv.logical_right_shift(3, shift=1, input='x', output='x>>1')`
- `aiger_bv.arithmetic_right_shift(3, shift=1, input='x', output='x>>1')`
- `aiger_bv.left_shift(3, shift=1, input='x', output='x<<1')`

## Comparison

- `aiger_bv.is_nonzero_gate(3, input='x', output='is_nonzero')`
- `aiger_bv.is_zero_gate(3, input='x', output='is_zero')`
- `aiger_bv.eq_gate(3, left='x', right='y', output='x=y')`
- `aiger_bv.ne_gate(3, left='x', right='y', output='x!=y')`
- `aiger_bv.unsigned_lt_gate(3, left='x', right='y', output='x<y')`
- `aiger_bv.unsigned_gt_gate(3, left='x', right='y', output='x>y')`
- `aiger_bv.unsigned_le_gate(3, left='x', right='y', output='x<=y')`
- `aiger_bv.unsigned_ge_gate(3, left='x', right='y', output='x>=y')`
- `aiger_bv.signed_lt_gate(3, left='x', right='y', output='x<y')`
- `aiger_bv.signed_gt_gate(3, left='x', right='y', output='x>y')`
- `aiger_bv.signed_le_gate(3, left='x', right='y', output='x<=y')`
- `aiger_bv.signed_ge_gate(3, left='x', right='y', output='x>=y')`
