from itertools import islice

import attr
import funcy as fn
from pyrsistent import pmap
from pyrsistent.typing import PMap


@attr.s(frozen=True, slots=True, auto_attribs=True, repr=False)
class Bundle:
    name: str = attr.ib()
    size: int = attr.ib()

    @size.validator
    def _assert_positive(self, _, value):
        if value < 0:
            raise ValueError("size must be a positive number.")

    def __len__(self): return self.size
    def __repr__(self): return f"{self.name}[:{self.size}]"
    def __iter__(self): return (self[i] for i in range(self.size))

    def __add__(self, other):
        return Bundle(f"{self.name}#+#{other.name}", len(self) + len(other))

    def __getitem__(self, idx):
        if isinstance(idx, int):
            assert idx in range(self.size)
            return f"{self.name}[{idx}]"

        start = 0 if idx.start is None else idx.start
        stop = self.size if idx.stop is None else idx.stop
        step = 1 if idx.step is None else idx.step
        return tuple(islice(self, start, stop, step))

    def blast(self, val):
        assert len(val) == self.size
        return dict(zip(self, val))


@attr.s(frozen=True, slots=True, auto_attribs=True)
class BundleMap:
    mapping: PMap[str, int] = attr.ib(default=pmap(), converter=pmap)

    def __getitem__(self, idx):
        return Bundle(name=idx, size=self.mapping[idx])

    def keys(self): return self.mapping.keys()
    def values(self): return self.mapping.values()
    def items(self): return self.mapping.items()
    def __iter__(self): return iter(self.mapping)

    def __add__(self, other):
        assert not (set(self.keys()) & set(other.keys()))
        mapping2 = other.mapping if isinstance(other, BundleMap) else other
        return BundleMap(self.mapping + mapping2)

    def blast(self, idx2vals):
        idx2vals = idx2vals.items()
        return fn.merge(*(self[idx].blast(val) for idx, val in idx2vals))
