from typing import Tuple, FrozenSet

import aiger
import attr
import funcy as fn
from pyrsistent import pmap
from pyrsistent.typing import PMap

from aigerbv import common
from aigerbv.bundle import BundleMap


@attr.s(frozen=True, slots=True, eq=False, auto_attribs=True)
class AIGBV:
    aig: aiger.AIG
    imap: BundleMap = BundleMap()
    omap: BundleMap = BundleMap()
    lmap: BundleMap = BundleMap()

    simulate = aiger.AIG.simulate
    simulator = aiger.AIG.simulator

    def write(self, path): 
        self.aig.write(path)

    @property
    def inputs(self): return set(self.imap.keys())

    @property
    def outputs(self): return set(self.omap.keys())

    @property
    def latches(self): return set(self.lmap.keys())

    def __call__(self, inputs, latches=None):
        # TODO: Make this an extension.
        encoded_in = {i for i, v in inputs.items() if isinstance(v, int)}
        inputs.update({
            i: self._encode_val(i, inputs[i], self.imap) for i in encoded_in
        })

        out2val, latch2val = self.aig(
            inputs=self.imap.blast(inputs),
            latches=None if latches is None else self.lmap.blast(latches)
        )
        return self.omap.unblast(out2val), self.lmap.unblast(latch2val)

    def __lshift__(self, other):
        return other >> self

    def __rshift__(self, other):
        interface = self.outputs & other.inputs
        assert not self.latches & other.latches
        assert not (self.outputs - interface) & other.outputs

        return AIGBV(
            aig=self.aig >> other.aig,
            imap=self.imap + other.imap.omit(interface),
            omap=other.omap + self.omap.omit(interface),
            lmap=self.lmap + other.lmap,
        )

    def __or__(self, other):
        assert not self.outputs & other.outputs
        assert not self.latches & other.latches

        shared_inputs = self.inputs & other.inputs
        if shared_inputs:
            relabels1 = {n: common._fresh() for n in shared_inputs}
            relabels2 = {n: common._fresh() for n in shared_inputs}
            self, other = self['i', relabels1], other['i', relabels2]

        circ = AIGBV(
            aig=self.aig | other.aig,
            imap=self.imap + other.imap,
            omap=self.omap + other.omap,
            lmap=self.lmap + other.lmap)

        if shared_inputs:
            for orig in shared_inputs:
                new1, new2 = relabels1[orig], relabels2[orig]
                circ <<= common.tee(len(self.imap[orig]), {orig: [new1, new2]})

        return circ

    def __getitem__(self, others):
        kind, relabels = others
        if kind not in {'i', 'o', 'l'}:
            raise NotImplementedError

        attr_name = {'i': 'imap', 'o': 'omap', 'l': 'lmap'}.get(kind)
        bmap1 = getattr(self, attr_name)
        assert not set(relabels.values()) & set(bmap1.keys())
        bmap2 = bmap1.relabel(relabels)
        circ = attr.evolve(self, **{attr_name: bmap2})

        # Update AIG to match new interface.
        relabels_aig = fn.merge(*(
            dict(zip(bmap1[k], bmap2[v])) for k, v in relabels.items()
            if k in bmap1
        ))
        return attr.evolve(circ, aig=circ.aig[kind, relabels_aig])

    def _encode_val(self, i, word, imap):
        # TODO: figure out encoding for word type.
        return tuple(common.encode_int(len(imap[i]), word, signed=False))

    def feedback(self, inputs, outputs, initials=None, latches=None,
                 keep_outputs=False, signed=False):
        if latches is None:
            latches = inputs

        idrop, imap = fn.lsplit(lambda x: x[0] in inputs, self.imap.items())
        odrop, omap = fn.lsplit(lambda x: x[0] in outputs, self.omap.items())

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
            imap=imap,
            omap=omap | (odrop if keep_outputs else frozenset()),
            lmap=self.lmap + dict(new_latches),
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
            imap=extract_map(self.imap.items(), aig.inputs),
            omap=extract_map(self.omap.items(), aig.outputs),
            lmap=extract_map(self.lmap.items(), aig.latches),
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

        imap, omap = fix_order(circ.imap.items()), fix_order(circ.omap.items())
        return attr.evolve(circ, imap=imap, omap=omap)


######### Lifting AIG to AIGBVs


def _diagonal_map(keys):
    return BundleMap({k: 1 for k in keys})


def append_index(aig):
    for key in ['inputs', 'outputs', 'latches']:
        relabels = {name: f"{name}[0]" for name in  getattr(aig, key)}
        aig = aig[key[0], relabels]
    return aig


def aig2aigbv(aig):
    return AIGBV(
        aig=append_index(aig),
        imap=_diagonal_map(aig.inputs),
        omap=_diagonal_map(aig.outputs),
        lmap=_diagonal_map(aig.latches),
    )
