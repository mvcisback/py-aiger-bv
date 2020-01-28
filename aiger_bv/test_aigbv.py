import aiger
import pytest
from pathlib import Path
from tempfile import TemporaryDirectory

from aiger_bv import aigbv
from aiger_bv import bundle as bdl


def test_AIGBV_smoke():
    circ = aigbv.aig2aigbv(aiger.and_gate(['x', 'y'], output='z'))

    assert circ.inputs == {'x', 'y'}
    assert circ.outputs == {'z'}
    assert circ.latches == set()

    assert circ.aig.inputs == {'x[0]', 'y[0]'}
    assert circ.aig.outputs == {'z[0]'}

    assert circ.imap == bdl.BundleMap({'x': 1, 'y': 1})

    circ = aigbv.aig2aigbv(aiger.delay(['x', 'y'], initials=[False, True]))
    assert circ.aig.inputs == {'x[0]', 'y[0]'}
    assert circ.aig.outputs == {'x[0]', 'y[0]'}
    assert circ.aig.latches == {'x[0]', 'y[0]'}


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


def test_write():
    circ = aigbv.aig2aigbv(aiger.and_gate(['x', 'y'], output='out'))
    with TemporaryDirectory() as tmpdir:
        circ.write(Path(tmpdir) / "test.aag")


def test_rebundle():
    circ = aigbv.aig2aigbv(aiger.and_gate(['x', 'y'], output='z'))
    circ2 = aigbv.rebundle_aig(circ.aig)
    assert circ.imap == circ2.imap
    assert circ.omap == circ2.omap
    assert circ.lmap == circ2.lmap
    assert circ2.aig.inputs == {'x[0]', 'y[0]'}
    assert circ2.aig.outputs == {'z[0]'}

    circ = aigbv.rebundle_aig(
        aiger.delay(['x[0]', 'y[0]'], initials=[False, True])
    )
    assert circ.inputs == circ.latches == circ.outputs == {'x', 'y'}
    assert circ.aig.inputs == circ.aig.outputs == circ.aig.latches \
        == {'x[0]', 'y[0]'}


def test_relabel():
    circ = aigbv.aig2aigbv(aiger.and_gate(['x', 'y'], output='out'))
    circ2 = circ['i', {'x': 'z'}]
    assert circ2.inputs == {'y', 'z'}
    assert circ2.aig.inputs == {'y[0]', 'z[0]'}

    circ2 = circ['o', {'out': 'tmp'}]
    assert circ2.outputs == {'tmp'}
    assert circ2.aig.outputs == {'tmp[0]'}

    with pytest.raises(AssertionError):
        circ['i', {'x': 'y', }]

    circ = aigbv.rebundle_aig(
        aiger.delay(['x[0]', 'y[0]'], initials=[False, True])
    )
    circ2 = circ['l', {'x': 'z'}]
    assert circ2.latches == {'z', 'y'}
    assert circ2.aig.latches == {'z[0]', 'y[0]'}


def test_latch2init():
    circ = aigbv.rebundle_aig(
        aiger.delay(['x[0]', 'y[0]'], initials=[False, True])
    )
    assert circ.latch2init == {'x': (False,), 'y': (True,)}

    circ = aigbv.rebundle_aig(
        aiger.delay(['x[0]', 'x[1]'], initials=[False, True])
    )
    assert circ.latch2init == {'x': (False, True)}


def test_feedback():
    circ = aigbv.rebundle_aig(aiger.identity(['x[0]', 'x[1]', 'y[0]', 'y[1]']))
    circ2 = circ.feedback(inputs=['x'], outputs=['y'])
    assert circ2.inputs == {'y'}

    circ = aigbv.rebundle_aig(aiger.identity(['x[0]', 'x[1]', 'y[0]', 'y[1]']))
    circ2 = circ.feedback(
        inputs=['x'], outputs=['y'], initials=[(True, False)]
    )
    assert circ2.latch2init['x'] == (True, False)


def test_loopback():
    circ = aigbv.rebundle_aig(aiger.identity(['x[0]', 'x[1]', 'y[0]', 'y[1]']))
    circ2 = circ.loopback({'input': 'x', 'output': 'y'})
    assert circ2.inputs == {'y'}

    circ = aigbv.rebundle_aig(aiger.identity(['x[0]', 'x[1]', 'y[0]', 'y[1]']))
    circ2 = circ.loopback(
        {'input': 'x', 'output': 'y', 'init': (True, False)}
    )
    assert circ2.latch2init['x'] == (True, False)


def test_unroll():
    circ = aigbv.rebundle_aig(aiger.identity(['x[0]', 'x[1]', 'y[0]', 'y[1]']))
    circ2 = circ.feedback(inputs=['x'], outputs=['y'])

    circ3 = circ2.unroll(3)
    assert circ3.inputs == {'y##time_0', 'y##time_1', 'y##time_2'}
    assert circ3.outputs == {'x##time_1', 'x##time_2', 'x##time_3'}


def test_cutlatches():
    circ = aigbv.rebundle_aig(aiger.identity(['x[0]', 'x[1]', 'y[0]', 'y[1]']))
    circ2 = circ.feedback(inputs=['x'], outputs=['y'])
    circ3, lmap = circ2.cutlatches(renamer=lambda x: f"{x}##time_0")

    assert circ3.inputs >= circ2.inputs
    assert circ3.outputs >= circ2.outputs
    assert len(lmap) == 1
