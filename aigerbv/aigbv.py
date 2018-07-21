from typing import Tuple, FrozenSet, NamedTuple, Union, Mapping, List

import aiger
import funcy as fn


BV_MAP = FrozenSet[Tuple[str, Tuple[str]]]


def _blast(bvname2vals, name_map):
    return fn.merge(
        *(dict(zip(names, bvname2vals[bvname])) for bvname, names in name_map)
    )


def _unblast(name2vals, name_map):
    def _collect(names):
        return tuple(name2vals[n] for n in names)

    return {bvname: collect(names) for bvname, names in name_map}


class AIGBV(NamedTuple):
    aig: aiger.AIG
    input_map: BV_MAP
    output_map: BV_MAP
    latch_map: BV_MAP

    def __rshift__(self, other):
        interface = self.outputs & other.inputs

        assert not self.latches & other.latches
        assert not (self.outputs - interface) & other.outputs

        input_map2 = {kv for kv in other.input_map if kv[0] not in interface}
        output_map2 = {kv for kv in self.output_map if kv[0] not in interface}
        return AIGBV(
            aig=self.aig >> other.aig,
            input_map=self.input_map | input_map2,
            output_map=output_map2 | other.output_map,
            latch_map=self.latch_map | other.latch_map,
        )

    def __or__(self, other):
        assert not self.inputs & other.inputs
        assert not self.outputs & other.outputs
        assert not self.latches & other.latches

        return AIGBV(
            aig=self.aig | other.aig,
            input_map=self.input_map | other.input_map,
            output_map=self.output_map | other.output_map,
            latch_map=self.latch_map | other.latch_map
        )

    def __call__(self, inputs, latches=None):
        if latches is None:
            latches = dict()
        
        out_vals, latch_vals = self.aig(
            inputs=_blast(inputs, self.input_map),
            latches=_blast(latches, self.latch_map)
        )
        outputs = _unblast(out_vals, self.output_map)
        latch_outs = _unblast(latch_vals, self.latch_map)
        return outputs, latch_outs

    
    # def __getitem__ <- TODO: implement renaming.
    # def unroll <- TODO
    # def feedback <- TODO
    # def write <- TODO
    # def simulator

    @property
    def inputs(self):
        return set(fn.pluck(0, self.input_map))

    @property
    def outputs(self):
        return set(fn.pluck(0, self.output_map))

    @property
    def latches(self):
        return set(fn.pluck(0, self.latch_map))
