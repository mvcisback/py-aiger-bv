import aiger
import aigerbv
import aiger_pltl
from aiger_analysis import simplify
from aiger_analysis.count import count


def _gridworld1d(n, state_name='x', action='a', start=0):
    return aigerbv.add_gate(n, state_name, action, state_name).feedback(
        inputs=[state_name],
        outputs=[state_name],
        initials=[start],
        keep_outputs=True
    )


def gridworld(n, start=(0, 0)):
    circ = _gridworld1d(n, 'x', 'ax', start[0]) \
        | _gridworld1d(n, 'y', 'ay', start[1])

    ax = cmn.lookup(mapping={0: 0, 1: 1, 2: 0, 3: -1},
                    input='tmp1',
                    output='ax',
                    inlen=2,
                    outlen=n,
                    in_signed=False)

    ay = cmn.lookup(mapping={0: 1, 1: 0, 2: -1, 3: 0},
                    input='tmp2',
                    output='ay',
                    inlen=2,
                    outlen=n,
                    in_signed=False)

    action = cmn.tee(2, {'a': ('tmp1', 'tmp2')}) >> (ax | ay)
    return action >> circ


if __name__ == '__main__':
    g = gridworld(4)
    g.write('gridworld.aag')
