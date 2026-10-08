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

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from dask import delayed, compute
from dask import config as dask_config

from pbrAudioCommon import EntityManager, debug_print, set_debug, set_debug_prefix

from ..lib.backend import Blosc2Backend

@dataclass
class StorageEngine:
    entity_manager: EntityManager

    # populated by register()
    engine: Optional[str] = None
    collection: Optional[str] = None
    objs_type: Optional[str] = None
    track_group: Optional[str] = None
    track_names: List[str] = field(default_factory=list)
    signal_type: Optional[str] = None
    total_samples: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # populated by materialize()
    backend: Optional[Blosc2Backend] = None
    tree_store_path: Optional[str] = None
    obj_indices: List[int] = field(default_factory=list)

    def __post_init__(self):
        config = self.entity_manager.get("config")
        set_debug(config.system.debug)
        set_debug_prefix(self.__class__.__name__)
        self.config = config

    def register(self, collection: str, objs_type: str, engine: str, track_group: str, track_names: List[str], signal_type: str, total_samples: int, metadata: Dict[str, Any] = None) -> None:
        """
        Record the schema for a group of tracks. Does NOT allocate storage.
        Call `materialize()` afterwards to pre-allocate the NDArrays.
        """
        self.collection = collection
        self.objs_type = objs_type
        self.engine = engine
        self.track_group = track_group
        self.track_names = list(track_names)
        self.signal_type = signal_type
        self.total_samples = int(total_samples)
        self.metadata = dict(metadata or {})

        # collect object indices from the entity manager
        objs_list = getattr(self.config, objs_type)
        self.obj_indices = [objs_list[k].idx for k in range(len(objs_list))]

        # resolve the on-disk TreeStore path
        self.tree_store_path = f"{self.config.storage.root_path}/{self.collection}.b2d"

        debug_print(
            f"register: engine={engine} collection={collection} "
            f"objects={len(self.obj_indices)} tracks={len(self.track_names)} "
            f"total_samples={self.total_samples}"
        )

    def materialize(self) -> None:
        """
        Pre-allocate the blosc2.NDArrays for objects registered so far.
        Each object is created with a single signal slot per track.
        Idempotent — safe to call again after adding objects.
        """
        if self.engine is None or self.total_samples is None:
            raise RuntimeError("materialize() called before register().")

        self.backend = Blosc2Backend.get(self.tree_store_path, mode="a")

        # discover any newly-added objects
        objs_list = getattr(self.config, self.objs_type)
        current = [objs_list[k].idx for k in range(len(objs_list))]
        self.obj_indices = current

        self.backend.materialize(
            engine=self.engine,
            obj_indices=self.obj_indices,
            track_names=self.track_names,
            total_samples=self.total_samples,
            signal_type=self.signal_type,
            metadata=self.metadata,
        )

        debug_print(
            f"materialize: allocated {len(self.obj_indices)} object(s) "
            f"at {self.tree_store_path}"
        )

    def write(self, audio_data: np.ndarray, obj_idx: int, track_name: str, signal_name: Optional[Any] = None, metadata: Optional[Dict[str, Any]] = None, sample_start: Optional[int] = None) -> None:
        """
        Offload a write to the backend via dask.delayed.

        audio_data : (C, S) or (S,) float array. If `sample_start` is given,
                     the chunk is zero-padded to `total_samples` and placed
                     at [sample_start : sample_start + S]. Otherwise it is
                     written from index 0.
        signal_name : if None, appends a new signal slot; otherwise, finds or
                      appends a signal slot with that name.
        """
        if self.backend is None:
            self.materialize()

        if audio_data is None:
            return

        arr = np.ascontiguousarray(audio_data, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        # capture a snapshot so the delayed task doesn't close over `self`
        backend = self.backend
        engine = self.engine
        start = sample_start

        task = delayed(backend.write_signal)(
            engine=engine,
            obj_idx=int(obj_idx),
            track_name=track_name,
            data=arr,
            metadata=metadata,
            sample_start=start,
            signal_name=signal_name,
        )

        compute(task)

    def read(self, engine: str, obj_idx: int, track_name: str, signal_name: str, start: int = 0, stop: Optional[int] = None) -> np.ndarray:
        if engine == self.engine and self.backend is None:
            self.materialize()

        track_index = self._resolve_track_index(track_name)

        if track_index is not None:
            backend = self.backend
            if backend is None:
                 backend = Blosc2Backend.get(self.tree_store_path, mode="r")
            
            signal = backend.read_signal(
                engine=engine,
                obj_idx=obj_idx,
                track_index=track_index,
                signal_name=signal_name,
                start=start,
                stop=stop,
            )
            return signal
        return None

    def _resolve_track_index(self, track_name: Any) -> Optional[int]:
        for i, name in enumerate(self.track_names):
            if name == track_name:
                return i
            # allow tuple/list ranges to match by containment
            if isinstance(name, (list, tuple)) and track_name in name:
                return i
        return None

    def close(self) -> None:
        if self.backend is not None:
            self.backend.close()
            self.backend = None

