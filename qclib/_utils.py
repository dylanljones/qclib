# coding: utf-8
#
# This code is part of qclib.
#
# Copyright (c) 2021, Dylan Jones
import os.path
import pickle
from collections.abc import MutableMapping
from typing import Iterable, Any


def cachekey(*args, **kwargs):
    argstr = "; ".join(str(x) for x in args)
    kwargstr = "; ".join(f"{k}={v}" for k, v in kwargs.items())
    return "; ".join(s for s in [argstr, kwargstr] if s)


class Cache(MutableMapping):

    def __init__(self, file: str = "", autosave=True):
        super().__init__()
        self._data = dict()
        self.file = file
        self.auotsave = autosave

        self.load()

    @staticmethod
    def key(*args, **kwargs):
        return cachekey(*args, **kwargs)

    def on_change(self):
        if self.auotsave:
            self.save()

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> Iterable[Any]:
        return iter(self._data)

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._data[key] = value
        self.on_change()

    def __delitem__(self, key: str) -> None:
        del self._data[key]
        self.on_change()

    def clear(self) -> None:
        self._data.clear()
        self.on_change()

    def save(self) -> None:
        if self.file:
            with open(self.file, "wb") as fh:
                pickle.dump(self._data, fh)

    def load(self) -> None:
        if os.path.isfile(self.file):
            with open(self.file, "rb") as fh:
                self._data = pickle.load(fh)


cache = Cache("cache.pkl", autosave=True)
