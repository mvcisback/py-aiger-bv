<figure>
  <img src="logo_text.svg" alt="py-aiger-bv logo" width=300px>
  <figcaption>pyAiger-BV: Extension of pyAiger for manipulating
    sequential bitvector circuits.</figcaption>
</figure>

[![Build Status](https://travis-ci.org/mvcisback/py-aiger-bv.svg?branch=master)](https://travis-ci.org/mvcisback/py-aiger-bv)
[![codecov](https://codecov.io/gh/mvcisback/py-aiger-bv/branch/master/graph/badge.svg)](https://codecov.io/gh/mvcisback/py-aiger-bv)
[![Updates](https://pyup.io/repos/github/mvcisback/py-aiger-bv/shield.svg)](https://pyup.io/repos/github/mvcisback/py-aiger-bv/)

[![PyPI version shields.io](https://img.shields.io/pypi/v/py-aiger-bv.svg)](https://pypi.python.org/pypi/py-aiger-bv/)
[![PyPI license](https://img.shields.io/pypi/l/py-aiger-bv.svg)](https://pypi.python.org/pypi/py-aiger-bv/)

# Table of Contents
- [About](#about-py-aiger-bv)
- [Installation](#installation)
- [BitVector Expr DSL](#bitvector-expression-dsl)
- [Sequential Circuit DSL](#sequential-circuit-dsl)

# About Py-Aiger-BV

This library provides word level abstractions on top of
[py-aiger](https://github.com/mvcisback/py-aiger). This is done by the
`AIGBV` which groups inputs, outputs, and latches into named
**ordered** sequences (tuples).

# Installation

TODO

# BitVector Expression DSL

## Unsigned

## Signed

# Sequential Circuit DSL

py-aiger-bv's Sequential Circuit DSL implements the same basic api as
py-aiger's Sequential Circuit DSL, but operates at the (variable
length) word level rather than the bit level.

```python
import aiger
import aigerbv


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

```python
# Create aigerbv.AIGERBV object from aiger.AIG object.
circ  = ... # Some aiger.AIG object
word_circ = aigerbv.aig2aigbv(circ)  # aigerbv.AIGBV object
```

## Gadget Library

### General Manipulation

```python
# Copy outputs 'x' and 'y' to 'w1, w2' and 'z1, z2'.
circ1 = circ >> aigerbv.tee(wordlen=3, iomap={
        'x': ('w1', 'w2'),
        'y': ('z1', 'z2')
    })

# Take 1 bit output, 'x', duplicate it 5 times, and group into
# a single 5-length word output, 'y'.
circ2 = circ >> aigerbv.repeat(wordlen=5, input='x', output='z')

# Reverse order of a word.
circ3 = circ >> aigerbv.reverse_gate(wordlen=5, input='x', output='z')

# Sink and Source circuits (see encoding section for encoding details).
## Always output binary encoding for 15. 
circ4 = aigerbv.source(wordlen=4, value=15, name='x', signed=False)

## Absorb output 'y'
circ5 = circ >> aigerbv.sink(wordlen=4, inputs=['y'])

# Identity Gate
circ6 = circ >> aigerbv.identity_gate(wordlen=3, input='x')

# Combine/Concatenate words
circ7 = circ >> aigerbv.combine_gate(
    left_wordlen=3, left='x',
    right_wordlen=3, right='y',
    output='z'
)

# Split words
circ8 = circ >> aigerbv.split_gate(
    input='x',
    left_wordlen=1, left='z',
    right_wordlen=2, right='w'
)

# Select single index of circuit and make it a wordlen=1 output.
circ9 = circ >> aigerbv.index_gate(wordlen=3, idx=1, input='x', output='x1')
```

## Encoding
TODO

## Bitwise Operations

- `aigerbv.bitwise_and(3, left='x', right='y', output='x&y')`
- `aigerbv.bitwise_or(3, left='x', right='y', output='x|y')`
- `aigerbv.bitwise_xor(3, left='x', right='y', output='x^y')`
- `aigerbv.bitwise_negate(3, left='x', output='~x')`

## Arithmetic

- `aigerbv.add_gate(3, left='x', right='y', output='x+y')`
- `aigerbv.subtract_gate_gate(3, left='x', right='y', output='x-y')`
- `aigerbv.inc_gate(3, left='x', output='x+1')`
- `aigerbv.dec_gate(3, left='x', output='x+1')`
- `aigerbv.negate_gate(3, left='x', output='-x')`

## Comparison

- `aigerbv.is_nonzero_gate(3, input='x', output='is_nonzero')`
- `aigerbv.is_zero_gate(3, input='x', output='is_zero')`
- `aigerbv.eq_gate(3, left='x', right='y', output='x=y')`
- `aigerbv.ne_gate(3, left='x', right='y', output='x!=y')`
- `aigerbv.unsigned_lt_gate(3, left='x', right='y', output='x<y')`
- `aigerbv.unsigned_gt_gate(3, left='x', right='y', output='x>y')`
- `aigerbv.unsigned_le_gate(3, left='x', right='y', output='x<=y')`
- `aigerbv.unsigned_ge_gate(3, left='x', right='y', output='x>=y')`
- `aigerbv.signed_lt_gate(3, left='x', right='y', output='x<y')`
- `aigerbv.signed_gt_gate(3, left='x', right='y', output='x>y')`
- `aigerbv.signed_le_gate(3, left='x', right='y', output='x<=y')`
- `aigerbv.signed_ge_gate(3, left='x', right='y', output='x>=y')`
