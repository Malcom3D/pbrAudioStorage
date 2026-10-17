# Copyright (C) 2025 Malcom3D <malcom3d.gpl@gmail.com>
#
# This file is part of pbrAudio.
#
# pbrAudio is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# pbrAudio is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with pbrAudio.  If not, see <https://www.gnu.org/licenses/>.
# SPDX-License-Identifier: GPL-3.0-or-later

import os
import blosc2
import numpy as np
from collections import OrderedDict
from typing import Dict, Optional


class HandleCache:
    """
    Process-local LRU of opened blosc2.DictStore handles.

    blosc2 opens the file once and mmaps it. Re-opening for every read/write
    is expensive (syscalls + index reload). This cache keeps a bounded number
    of open stores per process, keyed by absolute path.
    """

    def __init__(self, capacity: int = 8) -> None:
        self._cap = capacity
        self._lru: "OrderedDict[str, blosc2.DictStore]" = OrderedDict()

    # ------------------------------------------------------------------
    def get(self, path: str, mode: str = "a") -> blosc2.DictStore:
        path = os.path.abspath(path)
        if path in self._lru:
            self._lru.move_to_end(path)
            return self._lru[path]

        # evict if over capacity
        while len(self._lru) >= self._cap:
            _, store = self._lru.popitem(last=False)
            try:
                store.close()
            except Exception:
                pass

        cparams = blosc2.CParams(codec=blosc2.Codec.LZ4, clevel=1, typesize=np.dtype(np.dtype).itemsize, nthreads=16)
        dparams = blosc2.DParams(nthreads=8)
        if not os.path.exists(path):
            store = blosc2.DictStore(localpath=path, mode='w', cparams=cparams, dparams=dparams)
            store.close()
        wstore = blosc2.DictStore(localpath=path, mode=mode, cparams=cparams, dparams=dparams)
        rstore = blosc2.DictStore(localpath=path, mmap_mode='r', mode='r', cparams=cparams, dparams=dparams)
        self._lru[path] = store
        return rstore, wstore

    # ------------------------------------------------------------------
    def clear(self) -> None:
        for _, store in self._lru.items():
            try:
                store.close()
            except Exception:
                pass
        self._lru.clear()

    def __enter__(self) -> "HandleCache":
        return self

    def __exit__(self, *exc) -> None:
        self.clear()

