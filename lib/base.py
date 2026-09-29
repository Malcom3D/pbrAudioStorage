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
from typing import Any, Tuple
import numpy as np

class ArrayHandle(ABC):
    """Abstract base class for a handle to a persistent array."""
    name: str
    shape: Tuple[int, ...]
    dtype: np.dtype

    @abstractmethod
    def write(self, data: np.ndarray, slices: Any = ...) -> None:
        """Write data to the array."""
        ...

    @abstractmethod
    def read(self, slices: Any = ...) -> np.ndarray:
        """Read data from the array."""
        ...

    @abstractmethod
    def apply_jit(self, expr: str, **params: Any) -> None:
        """Apply a JIT-compiled expression to the array."""
        ...

class ArrayBackend(ABC):
    """Abstract base class for a storage backend."""
    @abstractmethod
    def create(self, name: str, shape: Tuple[int, ...], dtype: np.dtype, **kw: Any) -> ArrayHandle:
        """Create a new persistent array."""
        ...

    @abstractmethod
    def open(self, name: str) -> ArrayHandle:
        """Open an existing persistent array."""
        ...

    @abstractmethod
    def try_open(self, name: str, expected_shape: Tuple[int, ...]) -> ArrayHandle | None:
        """Try to open an array, returning returning None if it doesn't exist or has the wrong shape."""
        ...
