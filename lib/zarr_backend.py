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
import zarr
import numpy as np
from typing import Any, Tuple

from .base import ArrayBackend, ArrayHandle

class ZarrHandle(ArrayHandle):
    """Handle for a Zarr array."""
    def __init__(self, arr: zarr.Array):
        self._arr = arr
        self.name = os.path.basename(arr.store.path).replace('.zarr', '')
        self.shape = arr.shape
        self.dtype = np.dtype(arr.dtype)

    def write(self, data: np.ndarray, slices: Any = ...) -> None:
        self._arr[slices] = data

    def read(self, slices: Any = ...) -> np.ndarray:
        return self._arr[slices]

    def apply_jit(self, expr: str, **params: Any) -> None:
        raise NotImplementedError("JIT is only supported on the Zarr backend.")

class ZarrBackend(ArrayBackend):
    """Storage backend using Zarr."""
    def __init__(self, root_path: str, **store_kw: Any):
        self.root = root_path
        os.makedirs(self.root, exist_ok=True)
        self.store_kw = store_kw

    def create(self, name: str, shape: Tuple[int, ...], dtype: np.dtype, chunks: Tuple[int, ...] | None = None, **kw: Any) -> ZarrHandle:
        if chunks is None:
            chunks = (1,) * (len(shape) - 1) + (shape[-1],)
        
        z = zarr.open_array(
            store=f"{self.root}/{name}.zarr",
            mode="w",
            shape=shape,
            dtype=dtype,
            chunks=chunks,
            **self.store_kw,
        )
        return ZarrHandle(z)

    def open(self, name: str) -> ZarrHandle:
        z = zarr.open_array(f"{self.root}/{name}.zarr", mode="a")
        return ZarrHandle(z)

    def try_open(self, name: str, expected_shape: Tuple[int, ...]) -> ZarrHandle | None:
        path = f"{self.root}/{name}.zarr"
        if not os.path.exists(path):
            return None
        z = zarr.open_array(path, mode="a")
        if z.shape != expected_shape:
            return None
        return ZarrHandle(z)

