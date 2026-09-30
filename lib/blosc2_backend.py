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
import sys
import blosc2
import numpy as np
from typing import Any, Tuple

from .base import ArrayBackend, ArrayHandle

class Blosc2Handle(ArrayHandle):
    """Handle for a Blosc2 NDArray."""
    def __init__(self, arr: blosc2.NDArray):
        self._arr = arr
        self.name = arr.name or ""
        self.shape = arr.shape
        self.dtype = np.dtype(arr.dtype)

    def write(self, data: np.ndarray, slices: Any = ...) -> None:
        self._arr[slices] = data

    def read(self, slices: Any = ...) -> np.ndarray:
        return self._arr[slices]

    def apply_jit(self, expr: str, **params: Any) -> None:
        """Apply a JIT-compiled expression to the array in-place."""
        # blosc2.jit returns a new lazy array, so we reassign
        self._arr = blosc2.jit(expr, self._arr, **params)

class Blosc2Backend(ArrayBackend):
    """Storage backend using Blosc2 with JIT capabilities."""
    def __init__(self, root_path: str, cparams: dict | None = None):
        self.root = root_path
        os.makedirs(self.root, exist_ok=True)
        self.cparams = cparams or {"codec": blosc2.Codec.LZ4, "clevel": 1, "filters": [blosc2.Filter.SHUFFLE]}
        self.dparams = {"nthreads": 16)

    def create(self, name: str, shape: Tuple[int, ...], dtype: np.dtype, chunks: Tuple[int, ...] | None = None, **kw: Any) -> Blosc2Handle:
        if chunks is None:
            chunks = self._auto_chunks(shape, dtype)
        self.cparams['tipesize'] = sys.sizeof(dtype())
        arr = blosc2.empty(shape=shape, dtype=dtype, chunks=chunks, cparams=blosc2.CParams(self.cparams), dparams=blosc2.DParams(self.dparam), urlpath=f"{self.root}/{name}.b2nd", mode="w", **kw)
        arr.name = name
        return Blosc2Handle(arr)

    def open(self, name: str) -> Blosc2Handle:
        arr = blosc2.open(f"{self.root}/{name}.b2nd", mode="a", cparams=blosc2.CParams(self.cparams), dparams=blosc2.DParams(self.dparam))
        return Blosc2Handle(arr)

    def try_open(self, name: str, expected_shape: Tuple[int, ...]) -> Blosc2Handle | None:
        path = f"{self.root}/{name}.b2nd"
        if not os.path.exists(path):
            return None
        arr = blosc2.open(path, mode="a", cparams=blosc2.CParams(self.cparams), dparams=blosc2.DParams(self.dparam))
        if tuple(arr.shape) != tuple(expected_shape):
            return None  # Stale, will be recreated
        return Blosc2Handle(arr)

    @staticmethod
    def _auto_chunks(shape: Tuple[int, ...], dtype: np.dtype, target_bytes: int = 1 << 20) -> Tuple[int, ...]:
        """Pick chunk size so a chunk is ~1 MiB, prioritizing the last axis."""
        itemsize = np.dtype(dtype).itemsize
        # Start with full size for the last axis, 1 for others
        chunks = [1] * (len(shape) - 1) + [shape[-1]]
        
        # Shrink last axis if a single chunk exceeds target
        while (chunks[-1] * itemsize * int(np.prod(chunks[:-1]))) > target_bytes and chunks[-1] > 1:
            chunks[-1] //= 2
        return tuple(chunks)

