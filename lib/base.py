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

from abc import ABC, abstractmethod
import numpy as np

class ArrayHandle(ABC):
    name: str
    shape: tuple
    dtype: np.dtype

    @abstractmethod
    def write(self, data: np.ndarray, slices=...) -> None: ...

    @abstractmethod
    def read(self, slices=...) -> np.ndarray: ...

    @abstractmethod
    def apply_jit(self, op_name: str, **params) -> None: ...

class ArrayBackend(ABC):
    @abstractmethod
    def create(self, name: str, shape, dtype, **kw) -> ArrayHandle: ...

    @abstractmethod
    def open(self, name: str) -> ArrayHandle: ...
