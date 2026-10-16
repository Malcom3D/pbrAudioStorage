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

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


class BackendBase(ABC):
    """
    Abstract I/O backend for pbrAudioStorage.

    Concrete implementations: Blosc2Backend, (future) ZarrBackend, HDF5Backend,...
    All backends receive the *raw* dicts produced by StorageEngine, and are
    responsible for storing them without modifying schema semantics.
    """

    # -- schema ---------------------------------------------------------
    @abstractmethod
    def register_object(self, obj_idx: int, schema_dict: Dict[str, Any]) -> None:
        """Create the top-level node and its attrs for one object."""

    @abstractmethod
    def materialize(self, schema_dict: Dict[str, Any]) -> None:
        """Pre-allocate/validate the storage for the registered schema."""

    # -- writes ---------------------------------------------------------
    @abstractmethod
    def write(self,
              engine: str,
              obj_idx: int,
              track_name: str,
              audio_data: np.ndarray,
              sample_start: int = 0,
              signal_name: Optional[str] = None,
              signal_metadata: Optional[Dict[str, Any]] = None,
              metadata: Optional[Dict[str, Any]] = None) -> None:
        """Store one chunk/signal into the given track."""

    # -- reads ----------------------------------------------------------
    @abstractmethod
    def read(self,
             engine: str,
             obj_idx: Optional[int] = None,
             track_name: Optional[str] = None,
             signal_name: Optional[str] = None,
             unprocessed: bool = False,
             side_channels: Optional[Dict[str, np.ndarray]] = None) -> tuple[np.ndarray, Dict[str, Any]]:
        """Read raw or DAG-processed audio data together with the attrs tree."""

    # -- tree ----------------------------------------------------------
    @abstractmethod
    def read_tree(self) -> Dict[str, Any]:
        """Return the full attrs metadata tree."""

    @abstractmethod
    def edit_tree(self, payload: Any) -> Dict[str, Any]:
        """Apply a JSON-Patch subset to the attrs metadata tree."""

    # -- lifecycle -----------------------------------------------------
    def close(self) -> None:
        """Release resources; overridden by concrete backends."""

