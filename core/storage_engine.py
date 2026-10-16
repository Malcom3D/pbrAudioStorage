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

import os
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type

from ..lib.schema import AudioSchema, validate_schema
from ..lib.audio_data_type import AudioDataType, DataTypeKind, DefaultBehavior
from ..lib.backend import BackendBase
from ..lib.blosc2_backend import Blosc2Backend
from ..utils.errors import StorageError, BackendError


# ---------------------------------------------------------------------------
_BACKENDS: Dict[str, Type[BackendBase]] = {
    "blosc2": Blosc2Backend,
}


def register_backend(name: str, cls: Type[BackendBase]) -> None:
    _BACKENDS[name] = cls


# ---------------------------------------------------------------------------
@dataclass
class StorageEngine:
    """
    Backend-agnostic front-end for pbrAudioStorage.

    Parameters
    ----------
    entity_manager : EntityManager
        Used to read SystemConfig at init.
    backend : str | BackendBase
        Either a registered backend name ("blosc2", "zarr", ...) or an
        already-constructed BackendBase instance.
    urlpath : str, optional
        Storage file/dir. Defaults to ``<cache_path>/storage``.
    """
    entity_manager: Any
    backend: Any = "blosc2"
    urlpath: Optional[str] = None

    # runtime
    _backend: Optional[BackendBase] = field(default=None, init=False)
    tree_store_path: Optional[str] = field(default=None, init=False)

    # ------------------------------------------------------------------
    def __post_init__(self):
        config = self.entity_manager.get('config')

        # resolve url
        if self.urlpath is None:
            storage_cfg = getattr(config, "storage", None)
            root = getattr(storage_cfg, "root_path", None) or \
                   f"{config.system.cache_path}/storage"
            backend_name = getattr(storage_cfg, "backend", "blosc2")
            self.urlpath = root
            if isinstance(self.backend, str):
                self.backend = backend_name
        self.tree_store_path = self.urlpath

        # collection label
        self.collection = getattr(config.system, "collection", None) or "scene"

        # instantiate backend
        if isinstance(self.backend, str):
            cls = _BACKENDS.get(self.backend)
            if cls is None:
                raise StorageError(f"unknown backend {self.backend!r}; "
                                   f"known: {sorted(_BACKENDS)}")
            self._backend = cls(urlpath=self.urlpath, collection=self.collection)
        elif isinstance(self.backend, BackendBase):
            self._backend = self.backend
        else:
            raise StorageError("backend must be a name or BackendBase instance")

    # ------------------------------------------------------------------
    def register(self,
                 collection: str,
                 objs_type: str,
                 engine: str,
                 track_group: str,
                 track_names: Sequence[str],
                 signal_type: str,
                 total_samples: int,
                 metadata: Optional[Dict[str, Any]] = None,
                 track_behaviors: Optional[Dict[str, str]] = None,
                 track_kinds: Optional[Dict[str, str]] = None,
                 track_force_source: Optional[Dict[str, str]] = None,
                 ) -> AudioSchema:
        """
        Register an audio data schema.

        Called by physicsEngine._register_storage_schema, rigidBodyEngine,
        and any future engine. Returns the validated AudioSchema.
        """
        schema = AudioSchema(
            collection=collection,
            engine=engine,
            objs_type=objs_type,
            track_group=track_group,
            track_names=list(track_names),
            signal_type=signal_type,
            total_samples=int(total_samples),
            metadata=dict(metadata or {}),
            track_behaviors=dict(track_behaviors or {}),
            track_kinds=dict(track_kinds or {}),
            track_force_source=dict(track_force_source or {}),
        )
        validate_schema(schema)

        # register one schema descriptor per object of objs_type
        config = self.entity_manager.get('config')
        objs = getattr(config, schema.objs_type, [])
        for obj in objs:
            obj_schema_dict = {
                "collection": collection,
                "engine": engine,
                "obj_idx": int(obj.idx),
                "objs_type": objs_type,
                "name": str(obj.name),
                "signal_type": signal_type,
                "total_samples": int(total_samples),
                "track_names": list(track_names),
                "track_behaviors": dict(track_behaviors or {}),
                "track_kinds": dict(track_kinds or {}),
                "metadata": dict(metadata or {}),
            }
            self._backend.register_object(int(obj.idx), obj_schema_dict)

        return schema

    # ------------------------------------------------------------------
    def materialize(self) -> None:
        """
        Physically pre-allocate NDArrays for every registered object.
        Idempotent: objects already materialized are skipped by the backend.
        """
        if self._backend is None:
            raise StorageError("no backend initialized")

        config = self.entity_manager.get('config')
        # Walk every registered schema by re-reading from the backend.
        # We keep the schemas on the backend side; iterate per object here.
        for obj in getattr(config, "objects", []):
            obj_schema_dict = {
                "collection": self.collection,
                "engine": None,  # filled per-engine below
                "obj_idx": int(obj.idx),
            }
            # We must know the engines under which this object was registered.
            # Backend tracks them via (engine, obj_idx) keys -> walk store.
            prefix = f"{self.collection}/"
            for key in list(self._backend._store.keys()):
                if not key.startswith(prefix):
                    continue
                tail = key[len(prefix):]
                if "/" not in tail:
                    continue
                eng, rest = tail.split("/", 1)
                if rest.lstrip("-").isdigit() and int(rest) == int(obj.idx):
                    # already materialized
                    pass
            # If not materialized at all, materialize using the *last* schema
            # registered for this object:
            for (eng, o_idx), schema in self._backend._schemas.items():
                if int(o_idx) != int(obj.idx):
                    continue
                try:
                    self._backend.materialize({
                        "collection": self.collection,
                        "engine": eng,
                        "obj_idx": int(obj.idx),
                        "name": str(obj.name),
                        "signal_type": schema.signal_type,
                        "total_samples": schema.total_samples,
                        "track_names": schema.track_names,
                        "metadata": schema.metadata,
                    })
                except BackendError:
                    pass

    # ------------------------------------------------------------------
    def write(self,
              audio_data: np.ndarray,
              obj_idx: int,
              track_name: str,
              sample_start: int = 0,
              engine: str = None,
              signal_name: Optional[str] = None,
              signal_metadata: Optional[Dict[str, Any]] = None,
              metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Store one chunk or one full signal.

        The engine defaults to the single registered engine for this obj_idx.
        """
        if self._backend is None:
            raise StorageError("no backend initialized")

        if engine is None:
            engine = self._infer_engine(int(obj_idx))
            if engine is None:
                raise BackendError(f"no engine registered for obj {obj_idx}")

        self._backend.write(
            engine=engine,
            obj_idx=int(obj_idx),
            track_name=track_name,
            audio_data=audio_data,
            sample_start=int(sample_start),
            signal_name=signal_name,
            signal_metadata=signal_metadata,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    def read(self,
             engine: Optional[str] = None,
             obj_idx: Optional[int] = None,
             track_name: Optional[str] = None,
             signal_name: Optional[str] = None,
             unprocessed: bool = False,
             side_channels: Optional[Dict[str, np.ndarray]] = None,
             ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Read raw (unprocessed=True) or DAG-processed (unprocessed=False).
        Returns (audio_ndarray, attrs_tree_dict).
        """
        if self._backend is None:
            raise StorageError("no backend initialized")
        return self._backend.read(
            engine=engine,
            obj_idx=obj_idx,
            track_name=track_name,
            signal_name=signal_name,
            unprocessed=unprocessed,
            side_channels=side_channels,
        )

    # ------------------------------------------------------------------
    def read_tree(self) -> Dict[str, Any]:
        if self._backend is None:
            raise StorageError("no backend initialized")
        return self._backend.read_tree()

    # ------------------------------------------------------------------
    def edit_tree(self, payload: Any) -> Dict[str, Any]:
        if self._backend is None:
            raise StorageError("no backend initialized")
        return self._backend.edit_tree(payload)

    # ------------------------------------------------------------------
    def close(self) -> None:
        if self._backend is not None:
            self._backend.close()
            self._backend = None

    def __enter__(self) -> "StorageEngine":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------
    def _infer_engine(self, obj_idx: int) -> Optional[str]:
        """Return the unique engine registered for obj_idx, if any."""
        engines = set()
        for (eng, o_idx), _ in self._backend._schemas.items():
            if int(o_idx) == int(obj_idx):
                engines.add(eng)
        if not engines:
            return None
        if len(engines) > 1:
            raise BackendError(
                f"obj_idx {obj_idx} registered under multiple engines "
                f"{sorted(engines)}; pass engine= explicitly"
            )
        return engines.pop()

