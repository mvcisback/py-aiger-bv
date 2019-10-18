from itertools import islice

import attr


@attr.s(frozen=True, slots=True, auto_attribs=True, repr=False)
class Bundle:
    name: str = attr.ib()
    size: int = attr.ib()

    @size.validator
    def _assert_positive(self, _, value):
        if value < 0:
            raise ValueError("size must be a positive number.")

    def __len__(self):
        return self.size

    def __add__(self, other):
        return Bundle(f"{self.name}#+#{other.name}", len(self) + len(other))

    def __repr__(self):
        return f"{self.name}[:{self.size}]"

    def __iter__(self):
        return (self[i] for i in range(self.size))

    def __getitem__(self, idx):
        if isinstance(idx, int):
            assert idx in range(self.size)
            return f"{self.name}[{idx}]"

        start = 0 if idx.start is None else idx.start
        stop = self.size if idx.stop is None else idx.stop
        step = 1 if idx.step is None else idx.step
        return tuple(islice(self, start, stop, step))
