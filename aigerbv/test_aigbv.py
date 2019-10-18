import aiger

from aigerbv import aigbv
from aigerbv import bundle as bdl


def test_AIGBV_smoke():
    circ = aigbv.aig2aigbv(aiger.and_gate(['x', 'y'], output='z'))

    assert circ.inputs == {'x', 'y'}
    assert circ.outputs == {'z'}
    assert circ.latches == set()

    assert circ.aig.inputs == {'x[0]', 'y[0]'}
    assert circ.aig.outputs == {'z[0]'}

    assert circ.imap == bdl.BundleMap({'x': 1, 'y': 1})


def test_AIGBV_seq_compose():
    circ1 = aigbv.aig2aigbv(aiger.and_gate(['x1', 'x2'], output='tmp'))
    circ2 = aigbv.aig2aigbv(aiger.and_gate(['tmp', 'x3'], output='x4'))
    circ12 = circ1 >> circ2

    assert circ12.inputs == circ1.inputs | circ2.inputs - {'tmp'}
    assert circ12.outputs == circ2.outputs

    assert circ12({'x1': (True,), 'x2': (True,), 'x3': (True,)})[0]['x4'][0]


def test_AIGBV_or_compose():
    circ1 = aigbv.aig2aigbv(aiger.and_gate(['x1', 'x2'], output='tmp'))
    circ2 = aigbv.aig2aigbv(aiger.and_gate(['tmp', 'x3'], output='x4'))
    circ12 = circ1 | circ2

    assert circ12.inputs == circ1.inputs | circ2.inputs
    assert circ12.outputs == circ1.outputs | circ2.outputs

    vals = {'x1': (True,), 'x2': (True,), 'x3': (True,), 'tmp': (False,)}
    assert circ12(vals)[0] == {
        'tmp': (True,),
        'x4': (False,),
    }


def test_relabel():
    circ = aigbv.aig2aigbv(aiger.and_gate(['x', 'y'], output='out'))
    circ2 = circ['i', {'x': 'z'}]
    assert circ2.inputs == {'y', 'z'}
    assert circ2.aig.inputs == {'y[0]', 'z[0]'}
