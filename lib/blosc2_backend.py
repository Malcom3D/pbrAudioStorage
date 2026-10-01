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
        self.name = arr.name if hasattr(arr, "name") else ""
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
    def __init__(self, root_path: str, cparams: dict | None = None, dparams_nthreads: int = 16, chunk_size_samples: int = 1 << 16):
        self.root = root_path
        os.makedirs(self.root, exist_ok=True)
        self.chunk_size_samples = int(chunk_size_samples)

        processed_cparams = {}
        if cparams:
            for key, value in cparams.items():
                if key == "codec" and isinstance(value, blosc2.Codec):
                    processed_cparams[key] = value.value
                elif key == "filters" and isinstance(value, list):
                    processed_cparams[key] = [f.value if isinstance(f, blosc2.Filter) else f for f in value]
                else:
                    processed_cparams[key] = value
        self.cparams = processed_cparams or {
            "codec": blosc2.Codec.LZ4.value,
            "clevel": 1,
            "filters": [blosc2.Filter.SHUFFLE.value],
        }
        self.dparams = blosc2.DParams(nthreads=dparams_nthreads)


    def create(self, name: str, shape: Tuple[int, ...], dtype: np.dtype, chunks: Tuple[int, ...] | None = None, **kw: Any) -> Blosc2Handle:
        if shape is None or any(int(d) <= 0 for d in shape):
            raise ValueError(f"Refusing to create array '{name}' with invalid shape {shape}")

        if chunks is None:
            chunks = self._auto_chunks(shape, dtype)

        arr = blosc2.zeros(shape=shape, dtype=dtype, chunks=chunks, cparams=self.cparams, dparams=self.dparams, urlpath=f"{self.root}/{name}.b2nd", mode="w", **kw)
        return Blosc2Handle(arr)

    def open(self, name: str) -> Blosc2Handle:
        arr = blosc2.open(f"{self.root}/{name}.b2nd", mode="a")
        return Blosc2Handle(arr)

    def try_open(self, name: str, expected_shape: tuple[int, ...]) -> Blosc2Handle | None:
        path = f"{self.root}/{name}.b2nd"
        if not os.path.exists(path):
            return None
        try:
            arr = blosc2.open(path, mode="a")
        except Exception:
            return None
        if tuple(arr.shape) != tuple(expected_shape):
            return None  # Stale, will be recreated

        if len(arr.chunks) != len(expected_shape):
            return None
        last_chunk = int(arr.chunks[-1])
        last_dim = int(expected_shape[-1])
        if last_chunk <= 0 or last_dim <= 0:
            return None
        if min(last_dim, self.chunk_size_samples) != last_chunk:
            return None
        return Blosc2Handle(arr)

    def _auto_chunks(self, shape: Tuple[int, ...], dtype: np.dtype) -> Tuple[int, ...]:
        """
        Pick chunk sizes.
        """
        itemsize = np.dtype(dtype).itemsize
        if not shape or any(d <= 0 for d in shape):
            raise ValueError(f"Cannot compute chunks for invalid shape: {shape}")

        last = min(int(shape[-1]), self.chunk_size_samples)

        chunks = [1] * (len(shape) - 1) + [last]
        chunks = [max(1, int(c)) for c in chunks]
        return tuple(chunks)
