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
import weakref
import numpy as np
import blosc2
from collections import deque
from typing import List, Tuple, Any, Dict, Optional
from dataclasses import dataclass, field

from ..lib.audio_data_type import AudioDataType
from ..lib.schema import AudioSchema
from ..utils.errors import BackendError

# urlpath (abspath) -> live Blosc2Backend instance
#
# If unused backends must GC'd, swap for:
#   _BACKEND_REGISTRY: "weakref.WeakValueDictionary[str, Blosc2Backend]" = weakref.WeakValueDictionary()
_BACKEND_REGISTRY: Dict[str, "Blosc2Backend"] = {}

@dataclass
class Blosc2Backend:
    """
    Concrete backend that stores one blosc2.NDArray per object's track,
    inside a single blosc2.TreeStore with all metadata in the attrs
    """
    urlpath: str
    _wstore: Optional[blosc2.TreeStore] = field(default=None, init=False, repr=False)
    _rstore: Optional[blosc2.TreeStore] = field(default=None, init=False, repr=False)
    _wpaths: List[str] = field(default_factory=list)
    _queue: deque = field(default_factory=deque)

    def __post_init__(self):
        self.urlpath = os.path.abspath(self.urlpath)

        existing = _BACKEND_REGISTRY.get(self.urlpath)
        if existing is not None:
            # Re-use the already-initialized instance:
            #   * overwrite self's state with the registry owner's state
            #   * re-register self so that `self is existing` semantically
            #     (both names now refer to a fully-live backend)
            self.__dict__.update(existing.__dict__)
            return

        self._wstore = self.open(self.urlpath, mode='a')
        _BACKEND_REGISTRY[self.urlpath] = self

    def __del__(self):
        Blosc2Backend.close_all()

    @classmethod
    def close_all(cls):
        for backend in list(_BACKEND_REGISTRY.values()):
            try:
                if backend._wstore is not None:
                    backend._wstore.close()
            finally:
                pass
        _BACKEND_REGISTRY.clear()

    def open(self, path: str = None, cparams: blosc2.CParams = None, dparams: blosc2.DParams = None, mode: str = 'a') -> (blosc2.TreeStore, blosc2.TreeStore):
        cparams = cparams or blosc2.CParams(codec=blosc2.Codec.LZ4, clevel=1, typesize=np.dtype(np.dtype).itemsize, nthreads=16)
        dparams = dparams or blosc2.DParams(nthreads=8)
        path = os.path.abspath(path) or self.urlpath
        if not os.path.exists(path):
            store = blosc2.TreeStore(localpath=path, mode='w', cparams=cparams, dparams=dparams)
            store.attrs['metadata'] = []
            store.close()
        if mode in ['w', 'a']:
            store = blosc2.TreeStore(localpath=path, mode=mode, cparams=cparams, dparams=dparams)
        elif mode == 'r':
            store = blosc2.TreeStore(localpath=path, mmap_mode=mode, mode=mode, cparams=cparams, dparams=dparams)
        return store

    def register(self, schema: AudioSchema) -> None:
        # save the AudioSchema to the blosc2.TreeStore attrs schemas metadata
        schema_dict = schema.to_dict()
        schema_registry = self._wstore.attrs['metadata']
        for _schema in schema_registry:
            if _schema == schema_dict:
                return
        self._wstore.attrs['metadata'] += [schema_dict]

    def materialize(self, collection: str, engine: str, obj_idx: int, obj_name: str) -> None:
        schema = self.get_schema(collection, engine)
        signal_names = schema.signal_names or []
        total_samples = int(schema.total_samples)

        np_dtype = np.float32 if schema.signal_type == "float32" else np.float64
        n_signals = len(signal_names) if len(signal_names) > 0 else 1

        # create object's tracks array
        for t_idx, track_name in enumerate(schema.track_names):
            track_arr = blosc2.zeros((n_signals,total_samples), dtype=np_dtype)

            # write track's storage metadata
            track_arr.attrs['track'] = AudioDataType(
                name=track_name,
                id=t_idx,
                data_type=schema.default_kinds['track'],
                default_behavior=schema.default_behaviors['track'],
                sample_rate=schema.sample_rate,
                bit_depth=schema.bit_depth,
                subtype=schema.subtype,
                metadata=schema.metadata,
                child_names=signal_names or [],
                child_indices=np.arange(len(signal_names)).tolist() if signal_names is not None else []
            ).to_dict()

            # write signal's storage metadata
            track_arr.attrs['signals'] = []
            for s_idx in range(n_signals):
                track_arr.attrs['signals'].append(AudioDataType(
                    name=signal_names[s_idx] if len(signal_names) > 0 else s_idx,
                    id=s_idx,
                    data_type=schema.default_kinds['signal'],
                    default_behavior=schema.default_behaviors['signal'],
                    sample_rate=schema.sample_rate,
                    bit_depth=schema.bit_depth,
                    subtype=schema.subtype,
                    metadata=schema.metadata,
                    child_names=[],
                    child_indices=[]
                ).to_dict())

            # store the track's array
            self._wstore[f"/{collection}/{engine}/{obj_idx}/{track_name}"] = track_arr

        # write the object's metadata
        self._wstore[f"/{collection}/{engine}/{obj_idx}"].attrs['metadata'] = AudioDataType(
            name=obj_name,
            id=obj_idx,
            data_type=schema.default_kinds['object'],
            default_behavior=schema.default_behaviors['object'],
            sample_rate=schema.sample_rate,
            bit_depth=schema.bit_depth,
            subtype=schema.subtype,
            metadata=schema.metadata,
            child_names=[],
            child_indices=[]
        ).to_dict()

        # check for engine's metadata
        _rstore = self.open(self.urlpath, mode='r')
        engine_id = len(_rstore.get_children(f"/{collection}")) - 1
        try:
            check = _rstore[f"/{collection}/{engine}"].attrs['metadata']
        except:
            # write the engine's metadata
            self._wstore[f"/{collection}/{engine}"].attrs['metadata'] = AudioDataType(
                name=engine,
                id=engine_id,
                data_type=schema.default_kinds['engine'],
                default_behavior=schema.default_behaviors['engine'],
                sample_rate=schema.sample_rate,
                bit_depth=schema.bit_depth,
                subtype=schema.subtype,
                metadata=schema.metadata,
                child_names=schema.track_names,
                child_indices=np.arange(len(schema.track_names)).tolist()
            ).to_dict()

        # check for collection's metadata
        _rstore = self.open(self.urlpath, mode='r')
        collection_id = len(_rstore.get_children(f"/")) - 1
        try:
            check = _rstore[f"/{collection}"].attrs['metadata']
            check.add_child(idx=engine_id, name=engine)
        except:
            # write the engine's metadata
            self._wstore[f"/{collection}"].attrs['metadata'] = AudioDataType(
                name=collection,
                id=collection_id,
                data_type=schema.default_kinds['collection'],
                default_behavior=schema.default_behaviors['collection'],
                sample_rate=schema.sample_rate,
                bit_depth=schema.bit_depth,
                subtype=schema.subtype,
                metadata=schema.metadata,
                child_names=[engine],
                child_indices=[engine_id]
            ).to_dict()

    def _drain(self):
        paths = ([f"{p[1][0]}/{p[1][1]}/{p[1][2]}/{p[1][3]}"] for p in self._queue)
        for path in paths:
            while self._queue and not path in self._wpaths:
                fn, args = self._queue.popleft()
                fn(*args)
            if self._queue:
                self._drain()

    def write(self, collection: str, engine: str, obj_idx: int, track_name: str, audio_data: np.ndarray, sample_start: int = 0, sample_stop: int = None, signal_name: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        path = f"/{collection}/{engine}/{obj_idx}/{track_name}"
        if path not in self._wstore:
            raise BackendError(f"Node {path} not materialized") 

        if path in self._wpaths:
            self._queue.append((self._write, (collection, engine, obj_idx, track_name, audio_data, sample_start, sample_stop, signal_name, metadata)))
        else:
            self._write(collection, engine, obj_idx, track_name, audio_data, sample_start, sample_stop, signal_name, metadata)
        self._drain()

    def _write(self, collection: str, engine: str, obj_idx: int, track_name: str, audio_data: np.ndarray, sample_start: int = 0, sample_stop: int = None, signal_name: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Write a track_name's signal to the blosc2.TreeStore
        """
        schema = self.get_schema(collection, engine)
        path = f"/{collection}/{engine}/{obj_idx}/{track_name}"
        self._wpaths.append(path)
        track_arr = self._wstore[path]
        track_meta = AudioDataType.from_dict(track_arr.attrs['track'])
        signals_meta = [AudioDataType.from_dict(signal_meta) for signal_meta in track_arr.attrs['signals']]
#        signals_meta: List[AudioDataType] = list(track_arr.attrs['signals'])
#        track_meta = track_arr.attrs['track']
#        signals_meta = track_arr.attrs['signals']

        new_meta = False 
        signal_idx = None

        if signal_name is not None:
            # search signal_name in track_meta.child_names
            if signal_name in track_meta.child_names:
                signal_idx = track_meta.child_names.index(signal_name)

        if signal_name is not None and signal_idx is None:
            # search signal_name or signals_meta
            for signal_meta in signals_meta:
                signal_meta = AudioDataType.from_dict(signal_meta)
                if signal_name in signal_meta.name:
                    signal_idx = signal_meta.id if signal_idx is None else signal_idx
                    break
                # Search signal_name in signal_meta child
                elif signal_name in signal_meta.child_names:
                    child_idx = signal_meta.child_names.index(signal_name)
                    signal_idx = signal_meta.child_indices[child_idx]
                    break

        if signal_idx is None:
            # Check schema default behaviors for signal (override single signal?)
            if len(signals_meta) == 1 and schema.default_behaviors['signal'] == 'single':
                signal_idx = 0

        new_meta = True if signal_idx is None or signal_idx == 0 else False
        signal_idx = len(signals_meta) if signal_idx is None else signal_idx
        signal_name = signal_name if signal_name is not None else signal_idx

        # resize for new signal
        if not track_arr.shape[0] > signal_idx:
            new_shape = (signal_idx + 1, track_arr.shape[1])
            track_arr.resize(new_shape)

        if new_meta:
            # add signal info to track_meta
            track_meta.add_child(signal_idx, signal_name)
            track_arr.attrs['track'] = track_meta.to_dict()

            # write signal's metadata to signals_meta
            signals_meta.append(AudioDataType(
                name=signal_name,
                id=signal_idx,
                data_type=schema.default_kinds['signal'],
                default_behavior=schema.default_behaviors['signal'],
                sample_rate=schema.sample_rate,
                bit_depth=schema.bit_depth,
                subtype=schema.subtype,
                metadata=metadata or {},
                child_names=[],
                child_indices=[]
            ).to_dict())
            track_arr.attrs['signals'] = signals_meta

        # check audio_data length
        audio_data = audio_data if audio_data.ndim == 1 else audio_data.T
        np_dtype = np.float32 if schema.signal_type == "float32" else np.float64
        if sample_stop is None:
            sample_stop = schema.total_samples
        if not audio_data.shape[0] == schema.total_samples:
            data_pad = np.zeros((schema.total_samples,), dtype=np_dtype)
            data_pad[start_sample:stop_sample] = audio_data
            audio_data = data_pad
        # write signal's audio_data to storage
        track_arr[signal_idx] = audio_data

        # rewrite track_arr to storage
        self._wstore[path] = track_arr

        self._wpaths.remove(path)

    def read(self, collection: str, engine: str, obj_idx: int, track_name: str, signal_idx: int) -> np.ndarray:
        """
        Read a signal
        """
        _rstore = self.open(self.urlpath, mode='r')

        track_path = f"/{collection}/{engine}/{obj_idx}/{track_name}"
        track_arr = _rstore[track_path]
        return track_arr[signal_idx].reshape(1,-1)

    def get_child(self, paths: List[str] = ['/']) -> List[str]:
        _rstore = self.open(self.urlpath, mode='r')
        for path in paths:
            childs = _rstore.get_descendants(path)
            valid_childs = []
            for child in childs:
                if len(child.strip('/').split('/')) >= 4:
                    track_path = "/" + "/".join(child.strip('/').split('/')[0:4])
                    track_arr = _rstore[track_path]
                    audio_data_type = track_arr.attrs['track']

#                    if len(child.strip('/').split('/')) > 3:
#                        signal_name = child.strip('/').split('/')[4:][0]
#                        signal_idx = audio_data_type.child_names.index(signal_name)
#                        if signal_idx is None:
#                            if str(signal_name).isdigit():
#                                signal_idx = int(signal_name)
#                        if signal_idx is not None:
#                            audio_data_type = track_arr.attrs['signals'][signal_idx]
                else:
                    audio_data_type = _rstore[child].attrs['metadata']
                if not audio_data_type['default_behavior'] == "none":
                    valid_childs.append(child)
#                if len(valid_childs) > 0:
#                    valid_childs += self.get_child(valid_childs)
        return valid_childs

    def list(self, path: str = '/') -> List[str]:
        _rstore = self.open(self.urlpath, mode='r')
        path_parts = path.strip('/').split('/')
        if len(path_parts) >= 4:
            track_path = "/" + "/".join(path_parts[0:4]) if path_parts else "/"
            track_arr = _rstore[track_path]
            if len(path_parts) == 4:
                return track_arr.attrs['track'].child_names
            elif len(path_parts) > 4 and path_parts[5] in track_arr.attrs['track'].child_names:
                return track_arr.attrs['signals'][path_parts[5]].child_names
        return _rstore.get_children(path)

    def get_meta(self, path: str = '/', signal_name: str = None, signal_idx: int = None) -> AudioDataType:
        _rstore = self.open(self.urlpath, mode='r')
        path_parts = path.strip('/').split('/')
        if len(path_parts) == 4:
            track_arr = _rstore[path]
            if signal_name is not None or signal_idx is not None:
                if signal_name in track_arr.attrs['track']['child_names'] or signal_idx in track_arr.attrs['track']['child_indices']:
                    signal_idx = track_arr.attrs['track']['child_indices'][track_arr.attrs['track']['child_names'].index(signal_name)] if signal_name is not None else signal_idx
                    return AudioDataType.from_dict(track_arr.attrs['signals'][signal_idx])
            else:
                return AudioDataType.from_dict(track_arr.attrs['track'])
        return AudioDataType.from_dict(_rstore[path].attrs['metadata'])

    def get_schema(self, collection: str, engine: str) -> AudioSchema:
        _rstore = self.open(self.urlpath, mode='r')
        for schema in _rstore.attrs['metadata']:
            if schema['collection'] == collection and schema['engine'] == engine:
                return AudioSchema.from_dict(schema)
