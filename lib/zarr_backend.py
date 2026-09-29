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

import zarr
import zarrs
import numpy as np
from ..lib.base import ArrayBackend, ArrayHandle

zarr.config.set({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"})

class ZarrHandle(ArrayHandle):
    def __init__(self, arr):
        self._arr = arr
        self.name = arr.name
        self.shape = arr.shape
        self.dtype = np.dtype(arr.dtype)

    def write(self, data, slices=...):
        self._arr[slices] = data

    def read(self, slices=...):
        return self._arr[slices]

    def apply_jit(self, op_name, **params):
        raise NotImplementedError("JIT only supported on Blosc2 backend")


class ZarrBackend(ArrayBackend):
    def __init__(self, root_path, **store_kw):
        self.root = root_path
        self.store_kw = store_kw

    def create(self, name, shape, dtype, chunks=None, **kw):
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

    def open(self, name):
        z = zarr.open_array(f"{self.root}/{name}.zarr", mode="r")
        return ZarrHandle(z)

