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

import blosc2
import numpy as np
from .base import ArrayBackend, ArrayHandle

class Blosc2Handle(ArrayHandle):
    def __init__(self, arr: blosc2.NDArray):
        self._arr = arr
        self.name = arr.name or ""
        self.shape = arr.shape
        self.dtype = np.dtype(arr.dtype)

    def write(self, data: np.ndarray, slices=...) -> None:
        self._arr[slices] = data

    def read(self, slices=...) -> np.ndarray:
        return self._arr[slices]

    def apply_jit(self, op_name: str, **params) -> None:
        # Blosc2 JIT via blosc2.jit — compiled expression applied lazily
        expr = params.pop("expr", None)
        if expr is None:
            raise ValueError("apply_jit requires 'expr'")
        self._arr = blosc2.jit(expr, self._arr, **params)

class Blosc2Backend(ArrayBackend):
    def __init__(self, root_path: str, cparams: dict | None = None):
        self.root = root_path
        self.cparams = cparams or {
            "codec": blosc2.Codec.ZSTD,
            "clevel": 5,
            "filters": [blosc2.Filter.SHUFFLE],
        }

    def create(self, name, shape, dtype, chunks=None, blocks=None, **kw):
        if chunks is None:
            chunks = self._auto_chunks(shape, dtype)
        arr = blosc2.empty(
            shape=shape,
            dtype=dtype,
            chunks=chunks,
            blocks=blocks,
            cparams=self.cparams,
            urlpath=f"{self.root}/{name}.b2nd",
            mode="w",
            **kw,
        )
        arr.name = name
        return Blosc2Handle(arr)

    def open(self, name):
        arr = blosc2.open(f"{self.root}/{name}.b2nd", mode="r")
        return Blosc2Handle(arr)

    @staticmethod
    def _auto_chunks(shape, dtype, target_bytes=1 << 20):
        """Pick chunk size so a chunk is ~1 MiB."""
        itemsize = np.dtype(dtype).itemsize
        # last axis (time) gets full length; earlier axes chunked at 1
        chunks = [1] * (len(shape) - 1) + [shape[-1]]
        # shrink last axis if a single chunk exceeds target
        while (chunks[-1] * itemsize * int(np.prod(chunks[:-1]))) > target_bytes and chunks[-1] > 1:
            chunks[-1] //= 2
        return tuple(chunks)

