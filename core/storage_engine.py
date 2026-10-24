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
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type

from ..lib.schema import AudioSchema
from ..lib.audio_data_type import AudioDataType, DataTypeKind, DefaultBehavior
from ..lib.blosc2_backend import Blosc2Backend

from ..dag.dag_builder import DagBuilder
from ..dag.dag_evaluator import DagEvaluator, Props

from pbrAudioCommon import EntityManager

_BACKENDS: Dict[str, Type[BackendBase]] = {
    "blosc2": Blosc2Backend,
    "zarr": ZarrBackend,
}

@dataclass
class StorageEngine:
    """
    Backend-agnostic front-end for pbrAudioStorage.

    Parameters
    ----------
    entity_manager : EntityManager
        Used to read SystemConfig at init.
    backend : str
        A registered backend name ("blosc2", "zarr", ...).
    urlpath : str, optional
        Storage file/dir. Defaults to ``<cache_path>/storage``.
    """
    entity_manager: Any
    backend: Any = "blosc2"
    urlpath: Optional[str] = None

    # runtime
    _backend: Optional[BackendBase] = field(default=None, init=False)

    def __post_init__(self):
        config = self.entity_manager.get('config')

        # Determine total samples
        fps = config.system.fps
        fps_base = config.system.fps_base
        subframes = config.system.subframes
        sample_rate = config.system.sample_rate
        sfps = ( fps / fps_base ) * subframes # subframes per seconds
        self.total_samples = config.system.total_frames * sample_rate / sfps

        # resolve url
        if self.urlpath is None:
            self.urlpath = config.storage.root_path or f"{config.system.cache_path}/storage"
            self.backend = config.storage.backend

        # collection label
        self.collection = config.system.collection

        # instantiate backend
        cls = _BACKENDS[self.backend]
        if cls is None:
            raise StorageError(f"unknown backend {self.backend!r}; known: {sorted(_BACKENDS)}")
        self._backend = cls(urlpath=self.urlpath)

    def register(self, collection: str, objs_type: str, engine: str, track_group: str, track_names: Sequence[str], signal_type: str, default_behaviors: Dict[str, str], default_kinds: Dict[str, str], total_samples: int = None, sample_rate: int = None, bit_depth: int = None, subtype: str = None, metadata: Optional[Dict[str, Any]] = None):
        """
        Register an audio data schema.

        Called by engine's _register_storage_schema method.
        Physically pre-allocate NDArrays and write AudioDataType for every schema's objects type.
        """
        config = self.entity_manager.get('config')

        # Determine total samples
        fps = config.system.fps
        fps_base = config.system.fps_base
        subframes = config.system.subframes
        sample_rate = int(sample_rate or config.system.sample_rate)
        bit_depth = int(bit_depth or config.system.bit_depth)
        subtype = subtype or 'FLOAT' if signal_type == 'float32' else 'DOUBLE'
        sfps = ( fps / fps_base ) * subframes # subframes per seconds
        total_samples = total_samples or self.total_samples
        metadata = dict(metadata or {})

        # validate default_behaviors
        behaviors = {}
        for entity, behavior in default_behaviors.items():
            behaviors[entity] = DefaultBehavior(behavior).value

        # validate default_kinds
        kinds = {}
        for entity, kind in default_kinds.items():
            kinds[entity] = DataTypeKind(kind).value

        schema = AudioSchema(
            collection=collection,
            objs_type=objs_type,
            engine=engine,
            track_group=track_group,
            track_names=track_names,
            signal_type=signal_type,
            total_samples=total_samples,
            sample_rate=sample_rate,
            bit_depth=bit_depth,
            subtype=subtype,
            metadata=metadata,
            default_behaviors=behaviors,
            default_kinds=kinds
        )

        # validate schema
        schema.validate()

        self._backend.register(schema=schema)

        # Materialize schema
        objs = getattr(config, schema.objs_type, [])
        for obj in objs:
            self._backend.materialize(collection=collection, engine=engine, obj_idx=obj.idx, obj_name=obj.name)

    def write(self, collection: str, engine: str, obj_idx: int, track_name: str, audio_data: np.ndarray, sample_start: int = 0, sample_stop: int = None, signal_name: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Write signal to the backend offloading the caller class. The caller class call write with:
        from concurrent.futures import ThreadPoolExecutor
        _executor = ThreadPoolExecutor(max_workers=config.system.physical_core)
        _futures = []  # keep refs so tasks aren't GC'd
        fut = _executor.submit(storage_engine.write(collection, engine, obj_idx, track_name, audio_data, signal_name, sample_start, sample_stop, metadata))
        _futures.append(fut)
        _futures = [f for f in self._futures if not f.done()]
        _executor.shutdown(wait=True)
        """
        self._backend.write(collection=collection, engine=engine, obj_idx=obj_idx, track_name=track_name, audio_data=audio_data, sample_start=sample_start, sample_stop=sample_stop, signal_name=signal_name, metadata=metadata)

    def read(self, collection: Optional[str], engine: Optional[str] = None, obj_idx: Optional[int] = None, track_name: Optional[str] = None, signal_name: Optional[str] = None, unprocessed: bool = False) -> np.ndarray:
        """
        Read audio data from the backend

        Traverse the tree and use the DAG to evaluate the AudioDataType to process the raw float32 arrays from the signals.
        If ``unprocessed`` is True, returns raw float32 arrays evaluating the only the AudioDataType.default_behavior, without the signal processing chains.
        """
        if collection is None:
            # read the full graph, collections will be stacked on multitrack
            paths = self._backend.get_child()
                        
        elif engine is None:
            path = f"/{collection}"
            paths = self._backend.get_child([path])
                            
        elif obj_idx is None:
            path = f"/{collection}/{engine}"
            paths = self._backend.get_child([path])
                    
        elif track_name is None:
            path = f"/{collection}/{engine}/{obj_idx}"
            paths = self._backend.get_child([path])

        elif signal_name is None:
            path = f"/{collection}/{engine}/{obj_idx}/{track_name}"
            paths = self._backend.get_child([path])

        child_meta, audio_meta = ({} for _ in range(2))
        child_done, track_paths, object_paths, engine_paths, collection_paths = ([] for _ in range(5))
        for path in paths:
            audio_meta[path] = self._backend.get_meta(path=path)
            path_parts = path.strip('/').split('/')
            if len(path_parts) == 5:
                _collection, _engine, _obj_idx, _track_name, _signal_name = path_parts
                track_meta = self._backend.get_meta(path=f"/{_collection}/{_engine}/{_obj_idx}/{_track_name}")
                signal_idx = track_meta.child_names.index(_signal_name)
                # check for signal childs
                signal_childs = audio_meta[path].child_indices
                if len(signal_childs) > 0:
                    child_paths = ([path] + [f"/{_collection}/{_engine}/{_obj_idx}/{_track_name}/{child_idx}" for child_idx in signal_childs])
                    for child_path in child_paths:
                        if child_path in child_done:
                            continue
                        child_meta[child_path] = self._backend.get_meta(path=child_path)
                        child_done.append(child_path)
                    signal_datas = self._traverse_child(child_paths, child_meta, unprocessed)
                else:
                    audio_data = self._backend.read(collection=_collection, engine=_engine, obj_idx=_obj_idx, track_name=_track_name, signal_idx=_signal_idx)
                    props = self._get_props(audio_meta[path])
                    signal_datas[path] = self._apply_chain(audio_meta[path], audio_data, props) if not unprocessed else audio_data
            elif len(path_parts) == 4:
                track_paths.append(path)
            elif len(path_parts) == 3:
                object_paths.append(path)
            elif len(path_parts) == 2:
                engine_paths.append(path)
            elif len(path_parts) == 1:
                collection_paths.append(path)

        if signal_name is not None:
            return signal_datas

        track_datas = self._traverse(track_paths, signal_datas, audio_meta, unprocessed)
        if track_name is not None:
            return track_datas

        object_datas = self._traverse(object_paths, track_datas, audio_meta, unprocessed)
        if obj_idx is not None:
            return object_datas

        engine_datas = self._traverse(engine_paths, object_datas, audio_meta, unprocessed)
        if engine is not None:
            return engine_datas

        collection_datas = self._traverse(collection_paths, engine_datas, audio_meta, unprocessed)
        if collection is not None:
            return collection_datas
        else:
            return self._traverse(['/'], collection_datas, audio_meta, unprocessed)

    def _traverse_child(self, paths: List[str], audio_meta: Dict[str, Dict[str, Any]], unprocessed: bool = False):
        # read signals -> chain the signal -> apply behavior -> apply kind
        datas = {}
        for path in paths:
            path_parts = path.strip('/').split('/')
            _collection, _engine, _obj_idx, _track_name, _signal_idx = path_parts
            audio_data = self._backend.read(collection=_collection, engine=_engine, obj_idx=_obj_idx, track_name=_track_name, signal_idx=_signal_idx)
            props = self._get_props(audio_meta[path])
            datas[path] = self._apply_chain(audio_meta[path], audio_data, props) if not unprocessed else audio_data

            father_path = [f"{_collection}/{_engine}/{_obj_idx}/{_track_name}"]
            main_path = audio_meta.keys()[0]
            audio_data = {main_path: self._apply_behavior(father_path, datas, audio_meta)}
            audio_data = self._apply_kinds(audio_data, audio_meta)
        return audio_data[main_path]

    def _traverse(self, paths: List[str], audio_datas: Dict[str, np.ndarray], audio_meta: Dict[str, Dict[str, Any]], unprocessed: bool = False):
        datas = {}
        for path in paths:
            data = self._apply_behavior(path, audio_datas, audio_meta)
            props = self._get_props(audio_meta[path])
            datas[path] = self._apply_chain(audio_meta[path], data, props) if not unprocessed else data
        datas = self._apply_kinds(datas, audio_meta)
        return datas

    def _apply_behavior(self, path: str, audio_datas: Dict[str, np.ndarray], audio_meta: Dict[str, Dict[str, Any]]):
        datas = np.zeros((self.total_samples), dtype=np.float32)
        multi = False
        for child_path in audio_datas.keys():
            if path in child_path:
                if audio_meta[child_path].default_behavior == DefaultBehavior.NONE.value:
                    continue
                elif audio_meta[child_path].default_behavior == DefaultBehavior.SINGLE.value:
                    datas = audio_datas[child_path]
                    return datas
                elif audio_meta[child_path].default_behavior == DefaultBehavior.MERGE.value:
                    datas += audio_datas[child_path]
                elif audio_meta[child_path].default_behavior == DefaultBehavior.MULTI.value:
                    if not multi:
                        datas += audio_datas[child_path]
                        multi = True
                    else:
                        datas = np.stack((datas, audio_datas[child_path]), axis=0)
        return datas

    def _apply_kinds(self, audio_datas: Dict[str, np.ndarray], audio_meta: Dict[str, Dict[str, Any]]):
        datas = {}
        for path in audio_datas.keys():
            if audio_meta[path].data_type == DataTypeKind.MONO.value:
               if audio_datas[path].shape[0] == self.total_samples or audio_datas[path].shape[0] == 1 and audio_datas[path].ndim > 1:
                   datas[path] = audio_datas[path].reshape(1,-1) if audio_datas[path].shape[0] == self.total_samples else audio_datas[path]
               else:
                   datas[path] = np.sum(audio_datas[path], axis=0).reshape(1,-1) # fallback to merge behavior
            elif audio_meta[path].data_type == DataTypeKind.MULTI.value:
               datas[path] = audio_datas[path]
            elif audio_meta[path].data_type == DataTypeKind.CUSTOM.value:
               # custom encoder stub - fallback to MULTI
               # encoder = audio_meta[path].metadata[kind_encoder]
               datas[path] = audio_datas[path]
            elif audio_meta[path].data_type == DataTypeKind.SURROUND.value:
               # surround encoder stub - fallback to MULTI
               # encoder = audio_meta[path].metadata[kind_encoder]
               datas[path] = audio_datas[path]
            elif audio_meta[path].data_type == DataTypeKind.AMBISONIC.value:
               # ambisonic encoder stub - fallback to MULTI
               # encoder = audio_meta[path].metadata[kind_encoder]
               datas[path] = audio_datas[path]
        return datas

    def _build_chains_for_node(self, adt: AudioDataType) -> Optional[DagBuilder]:
        """
        Return a compiled DagBuilder for an AudioDataType's chain, or None
        when the node has no chain.

        The DAG is a linear pipeline: source = input signal,
        each processor = one chain step.
        """
        if not adt.chains:
            return None

        try:
            from pbrAudioSignals import get_processor
        except ImportError:
            return None

        builder = DagBuilder()
        prev = builder.add_source("__input__", None, metadata={"role": "input"})

        for step in adt.chains.get("processors", []):
            cls = get_processor(step["type"])
            proc = cls(**(step.get("params") or {}))
            prev = builder.add_processor(
                name=step["type"],
                processor=proc,
                inputs=[prev],
                metadata={"role": "processor"},
            )
        builder.compile(roots=[prev])
        return builder

    def _get_props(self, audio_meta: AudioDataType) -> List[Any]:
        props = []
        chains = audio_meta.chains
        for chain in chains:
            if chain.props.custom and chain.props.custom['entity']:    # read from EntityManager/KleptoEntityStore
                entities = self.entity_manager.get(chain.props.custom['entity'])
                for entity in entities:
                    if (entity.idx and entity.idx == chain.props.custom['entity_idx']) or (entity.obj_idx and entity.obj_idx == chain.props.custom['entity_idx']):
                        props.append(entity)
            elif chain.props.custom and chain.props.custom['path']:    # read from pbrAudioStorage
                data = self.read(chain.props.custom['path'])
                props.append(data)
            elif chain.props.custom and chain.props.custom['file']:    # read from filesystem
                file = np.fromfile(chain.props.custom['file'], dtype=chain.props.custom['dtype']
                props.append(file)
        return props

    def read(self, collection: Optional[str], engine: Optional[str] = None, obj_idx: Optional[int] = None, track_name: Optional[str] = None, signal_name: Optional[str] = None, unprocessed: bool = False) -> np.ndarray:

    def _apply_chain(self, adt: AudioDataType, x: np.ndarray, props: Props) -> np.ndarray:
        """
        Apply `adt.chains` to `x` using a linear DagBuilder + DagEvaluator.

        Any other DAG-aware processor output that lives in `props.custom`
        is not modified — this is purely non-destructive.
        """
        builder = self._build_chains_for_node(adt)
        if builder is None:
            return x

        if builder._order is None:
            builder.compile(roots=[list(builder.nodes().keys())[-1]])

        source_id = builder._order[0]
        root_id = builder._order[-1]

        x2d = x.reshape(1, -1) if x.ndim == 1 else np.ascontiguousarray(x, dtype=np.float32)
        builder.node(source_id).audio = x2d.astype(np.float32, copy=False)

        evaluator = DagEvaluator(builder, props=props)
        try:
            out = evaluator.evaluate(root_id)
        except Exception:
            return x2d   # never destroy audio on a broken chain

        if out is None:
            return x2d
        if out.ndim == 1:
            out = out.reshape(1, -1)
        return out.astype(np.float32, copy=False)
