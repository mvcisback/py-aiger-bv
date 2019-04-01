from typing import Tuple, FrozenSet

import aiger
import attr
import funcy as fn

from aigerbv import common


BV_MAP = FrozenSet[Tuple[str, Tuple[str]]]


def _blast(bvname2vals, name_map):
    """Helper function to expand (blast) str -> int map into str ->
    bool map. This is used to send word level inputs to aiger."""
    if len(name_map) == 0:
        return dict()
    return fn.merge(*(dict(zip(names, bvname2vals[bvname]))
                      for bvname, names in name_map))


def _unblast(name2vals, name_map):
    """Helper function to lift str -> bool maps used by aiger
    to the word level. Dual of the `_blast` function."""
    def _collect(names):
        return tuple(name2vals[n] for n in names)

    return {bvname: _collect(names) for bvname, names in name_map}


@attr.s(frozen=True, slots=True, cmp=False, auto_attribs=True)
class AIGBV:
    aig: aiger.AIG
    input_map: BV_MAP = frozenset()   # TODO: add converter
    output_map: BV_MAP = frozenset()  # from dict-like obj.
    latch_map: BV_MAP = frozenset()

    @property
    def inputs(self):
        return set(fn.pluck(0, self.input_map))

    @property
    def outputs(self):
        return set(fn.pluck(0, self.output_map))

    @property
    def latches(self):
        return set(fn.pluck(0, self.latch_map))

    def __getitem__(self, others):
        if not isinstance(others, tuple):
            return super().__getitem__(others)

        kind, relabels = others
        if kind not in {'i', 'o', 'l'}:
            raise NotImplementedError

        attr_name = {
            'i': 'input_map',
            'o': 'output_map',
            'l': 'latch_map',
        }.get(kind)

        attr_value = fn.walk_keys(lambda x: relabels.get(x, x),
                                  getattr(self, attr_name))
        return attr.evolve(self, **{attr_name: attr_value})

    def __rshift__(self, other):
        interface = self.outputs & other.inputs

        assert not self.latches & other.latches
        assert not (self.outputs - interface) & other.outputs

        # Relabel interface to match up.
        aig = self.aig
        if interface:
            imap, omap = dict(other.input_map), dict(self.output_map)
            relabels = fn.merge(*(
                dict(zip(omap[name], imap[name])) for name in interface
            ))
            aig = aig[('o', relabels)]

        # Create composed aigbv
        input_map2 = {kv for kv in other.input_map if kv[0] not in interface}
        output_map2 = {kv for kv in self.output_map if kv[0] not in interface}
        return AIGBV(
            aig=aig >> other.aig,
            input_map=self.input_map | input_map2,
            output_map=output_map2 | other.output_map,
            latch_map=self.latch_map | other.latch_map,
        )

    def __lshift__(self, other):
        return other >> self

    def __or__(self, other):
        assert not self.outputs & other.outputs
        assert not self.latches & other.latches

        shared_inputs = self.inputs & other.inputs
        input_map = dict(self.input_map)
        if shared_inputs:
            relabels1 = {n: common._fresh() for n in shared_inputs}
            relabels2 = {n: common._fresh() for n in shared_inputs}
            self, other = self['i', relabels1], other['i', relabels2]

        circ = AIGBV(
            aig=self.aig | other.aig,
            input_map=self.input_map | other.input_map,
            output_map=self.output_map | other.output_map,
            latch_map=self.latch_map | other.latch_map)

        if shared_inputs:
            for orig in shared_inputs:
                new1, new2 = relabels1[orig], relabels2[orig]
                circ <<= common.tee(len(input_map[orig]), {orig: [new1, new2]})

        return circ

    def __call__(self, inputs, latches=None):
        if latches is not None:
            latches = _blast(latches, self.latch_map)

        out_vals, latch_vals = self.aig(
            inputs=_blast(inputs, self.input_map),
            latches=latches)
        outputs = _unblast(out_vals, self.output_map)
        latch_outs = _unblast(latch_vals, self.latch_map)
        return outputs, latch_outs

    def simulator(self, latches=None):
        inputs = yield
        while True:
            outputs, latches = self(inputs, latches)
            inputs = yield outputs, latches

    def simulate(self, input_seq, latches=None):
        sim = self.simulator()
        next(sim)
        return [sim.send(inputs) for inputs in input_seq]

    def write(self, path):
        self.aig.write(path)

    def feedback(self, inputs, outputs, initials=None, latches=None,
                 keep_outputs=False, signed=False):
        if latches is None:
            latches = inputs

        idrop, imap = fn.lsplit(lambda x: x[0] in inputs, self.input_map)
        odrop, omap = fn.lsplit(lambda x: x[0] in outputs, self.output_map)

        wordlens = [len(vals) for i, vals in idrop]
        new_latches = [(n, common.named_indexes(k, n))
                       for k, n in zip(wordlens, latches)]

        if initials is None:
            initials = [0 for _ in inputs]
        assert len(inputs) == len(outputs) == len(initials) == len(latches)

        initials = fn.lcat(
            common.encode_int(k, i, signed) for k, i in zip(wordlens, initials)
        )

        def get_names(key_vals):
            return fn.lcat(fn.pluck(1, key_vals))

        aig = self.aig.feedback(
            inputs=get_names(idrop),
            outputs=get_names(odrop),
            latches=get_names(new_latches),
            initials=initials,
            keep_outputs=keep_outputs,
        )

        imap, odrop, omap = map(frozenset, [imap, odrop, omap])
        return AIGBV(
            aig=aig,
            input_map=imap,
            output_map=omap | (odrop if keep_outputs else frozenset()),
            latch_map=self.latch_map | set(new_latches),
        )

    def unroll(self, horizon, *, init=True, omit_latches=True,
               only_last_outputs=False):
        aig = self.aig.unroll(
            horizon, init=init,
            omit_latches=omit_latches,
            only_last_outputs=only_last_outputs
        )

        # TODO: generalize and apply to all maps.

        def extract_map(name_map, names):
            lookup_root = fn.merge(*(
                {v: k for v in vals} for k, vals in name_map)
            )
            mapping = fn.group_by(
                lambda x: lookup_root[x.split('##time_')[0]],
                names
            )
            mapping = fn.walk_values(tuple, mapping)  # Make hashable.
            return frozenset(mapping.items())

        circ = AIGBV(
            aig=aig,
            input_map=extract_map(self.input_map, aig.inputs),
            output_map=extract_map(self.output_map, aig.outputs),
            latch_map=extract_map(self.latch_map, aig.latches),
        )
        # PROBLEM: aigbv.unroll currently doesn't preserve variable
        #          order.
        # WORK AROUND: Sort input and output maps
        # TODO: Remove when fixed!

        def _fix_order(names):
            def to_key(x):
                name, time = x.split('##time_')
                return int(time), name

            return tuple(sorted(names, key=to_key))

        def fix_order(mapping):
            return frozenset(fn.walk_values(_fix_order, dict(mapping)).items())

        imap, omap = fix_order(circ.input_map), fix_order(circ.output_map)
        return attr.evolve(circ, input_map=imap, output_map=omap)


def _diagonal_map(keys, frozen=True):
    dmap = {k: (k,) for k in keys}
    return frozenset(dmap.items()) if frozen else dmap


def aig2aigbv(aig):
    return AIGBV(
        aig=aig,
        input_map=_diagonal_map(aig.inputs),
        output_map=_diagonal_map(aig.outputs),
        latch_map=_diagonal_map(aig.latches),
    )
