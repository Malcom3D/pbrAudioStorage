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
import json
import copy
import threading  # only for a factory-created lock object, no threads spawned
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import blosc2
from dask import delayed, compute

# Dask scheduler configuration
from dask import config as dask_config
dask_config.set({'num_workers': 1024, 'optimization.fuse.active': True, 'optimization.fuse.max_depth': 10,})

from ..lib.audio_data_type import (
    AudioDataType,
    DataTypeKind,
    DefaultBehavior,
)
from ..dag.dag_builder import DagBuilder
from ..dag.dag_evaluator import DagEvaluator, Props
from ..utils.errors import BackendError, PatchError
from .schema import AudioSchema, validate_schema
from .backend import BackendBase
from .handle_cache import HandleCache


# ---------------------------------------------------------------------------
# Helpers - numba-free; all called inside main thread or dask synchronous graph
# ---------------------------------------------------------------------------

def _as_2d_f32(x: np.ndarray) -> np.ndarray:
    a = np.ascontiguousarray(x, dtype=np.float32)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    return a


def _stack_signals(signals: List[np.ndarray],
                   behavior: str) -> Optional[np.ndarray]:
    """
    Aggregate a list of 1-D or 2-D signal arrays according to *behavior*.

    Returns None if behavior is NONE or the list is empty.
    """
    if behavior == DefaultBehavior.NONE.value or not signals:
        return None

    s = [_as_2d_f32(x) for x in signals]

    if behavior == DefaultBehavior.SINGLE.value:
        return s[0].copy()

    if behavior == DefaultBehavior.MERGE.value:
        out = np.array(s[0], copy=True)
        for a in s[1:]:
            if a.shape != out.shape:
                # pad to shorter length
                n = min(out.shape[1], a.shape[1])
                out = out[:, :n] + a[:, :n]
            else:
                out += a
        return out

    if behavior == DefaultBehavior.MULTI.value:
        # ensure same length
        n = min(a.shape[1] for a in s)
        s = [a[:, :n] for a in s]
        return np.concatenate(s, axis=0)

    raise BackendError(f"unknown default_behavior {behavior!r}")


class _DummyLock:
    """
    No-op lock used when the caller guarantees single-writer discipline.

    We do NOT use threading.Lock() to avoid accidentally creating any
    threaded synchronization primitive that some hosts (Blender?) may
    police. The lock exists to serialize *in-process* concurrent writes,
    but you've said all writes are single-threaded in the caller.
    """
    def __enter__(self): return self
    def __exit__(self, *exc): return False


class Blosc2Backend(BackendBase):
    """
    Concrete backend that stores one blosc2.NDArray per
    ``(collection, engine, obj_idx)`` inside a blosc2.TreeStore,
    with all metadata in the NDArray's ``attrs`` dict.

    Layout on disk (inside the store's blosc2.DictStore):
      store["<collection>/<engine>/<obj_idx>"] -> blosc2.NDArray
        shape      = (1, n_signals, total_samples)
        dtype      = float32 | float64
        attrs:
          "object"  -> AudioDataType dict (obj-level)
          "tracks"  -> List[AudioDataType dict] (track-level)
          "signals" -> List[List[AudioDataType dict]] indexable [track_idx][signal_idx]
          "schema"  -> AudioSchema dict (as registered)
          "engine"  -> engine name
          "collection" -> collection name
    """

    # ------------------------------------------------------------------
    def __init__(self,
                 urlpath: str,
                 collection: str = None,
                 mode: str = "a",
                 handle_capacity: int = 8) -> None:
        self.urlpath = os.path.abspath(urlpath)
        self.collection = collection
        os.makedirs(os.path.dirname(self.urlpath) or ".", exist_ok=True)

        self._handle_cache = HandleCache(capacity=handle_capacity)
        self._store = self._handle_cache.get(self.urlpath, mode=mode)

        # We do NOT spawn any thread. The _lock attribute exists so callers
        # with an out-of-band reason to serialize can use it, but the default
        # is a no-op.
        self._lock: Any = _DummyLock()

        # schema cache: obj_idx -> AudioSchema
        self._schemas: Dict[Tuple[str, int], AudioSchema] = {}

        # caches for the DAG evaluator, keyed by (engine, obj_idx)
        self._chains_cache: Dict[Tuple[str, int], DagBuilder] = {}

    # ------------------------------------------------------------------
    # Store helpers
    # ------------------------------------------------------------------
    def _node_path(self, engine: str, obj_idx: int) -> str:
        return f"{self.collection}/{engine}/{int(obj_idx)}"

    def _engine_path(self, engine: str) -> str:
        return f"{self.collection}/{engine}"

    def _ensure_engine_container(self, engine: str) -> None:
        """
        Ensure the parent tree containers exist. We store an empty dict
        at engine level; blosc2.TreeStore creates intermediate nodes lazily.
        """
        ep = self._engine_path(engine)
        if ep not in self._store:
            # blosc2.DictStore doesn't like empty containers; use a marker
            self._store[ep] = {"_kind": "engine_container", "engine": engine}

    # ------------------------------------------------------------------
    # Registration / materialize
    # ------------------------------------------------------------------
    def register_object(self, obj_idx: int, schema_dict: Dict[str, Any]) -> None:
        schema = AudioSchema(**{k: v for k, v in schema_dict.items()
                                if k in AudioSchema.__dataclass_fields__})
        validate_schema(schema)
        self._schemas[(schema.engine, obj_idx)] = schema

    def materialize(self, schema_dict: Dict[str, Any]) -> None:
        """
        Pre-allocate the top-level NDArray for one object described by *schema_dict*.
        Expects keys: collection, engine, obj_idx, name, signal_type, total_samples,
        track_names.
        """
        engine = schema_dict["engine"]
        obj_idx = int(schema_dict["obj_idx"])
        name = schema_dict["name"]
        signal_type = schema_dict.get("signal_type", "float32")
        total_samples = int(schema_dict["total_samples"])
        track_names: List[str] = list(schema_dict["track_names"])
        collection = schema_dict.get("collection", self.collection)

        if collection != self.collection:
            raise BackendError(
                f"materialize() collection mismatch: got {collection}, "
                f"backend bound to {self.collection}"
            )

        np_dtype = np.float32 if signal_type == "float32" else np.float64

        n_tracks = len(track_names)
        # n_signals starts at n_tracks (one signal per track, unnamed).
        # The signal name is stored in the AudioDataType so a "renamedamed"
        # signal is just metadata.
        n_signals = n_tracks

        with self._lock:
            self._ensure_engine_container(engine)
            path = self._node_path(engine, obj_idx)

            if path in self._store:
                arr = self._store[path]
                # already materialized; leave alone
                return

            shape = (1, n_signals, total_samples)
            cparams = blosc2.CParams(
                codec=blosc2.Codec.LZ4,
                clevel=1,
                typesize=np.dtype(np_dtype).itemsize,
                nthreads=16,
            )
            dparams = blosc2.DParams(nthreads=8)

            arr = blosc2.empty(
                shape,
                dtype=np_dtype,
                chunks=(1, 1, min(total_samples, 1 << 16)),
                blocks=(1, 1, min(total_samples, 1 << 14)),
                cparams=cparams,
                dparams=dparams,
                urlpath=None,   # in-memory NDArray; DictStore serializes
                contiguous=False,
                fill_value=0.0,
            )

            # ---- attrs: object node ----
            obj_node = AudioDataType(
                name=name,
                id=obj_idx,
                data_type=DataTypeKind.MULTI_TRACK.value,
                default_behavior=DefaultBehavior.MULTI.value,
                sample_rate=48000,
                bit_depth=32,
                subtype="FLOAT",
                channels=1,
                metadata={
                    "signal_type": signal_type,
                    "total_samples": total_samples,
                },
            )
            arr.attrs["object"] = obj_node.to_dict()

            # ---- attrs: track nodes ----
            tracks = []
            for ti, tname in enumerate(track_names):
                t = AudioDataType(
                    name=tname,
                    id=ti,
                    data_type=DataTypeKind.CUSTOM.value,
                    default_behavior=DefaultBehavior.MERGE.value,
                                       channels=1,
                    child_indices=[ti],
                    child_names=[tname],
                )
                tracks.append(t)
            arr.attrs["tracks"] = [t.to_dict() for t in tracks]

            # ---- attrs: signal nodes, initialized one per track ----
            signals: List[List[Dict[str, Any]]] = []
            for ti, tname in enumerate(track_names):
                sig = AudioDataType(
                    name=tname,   # default signal name == track name
                    id=0,
                    data_type=DataTypeKind.MONO.value,
                    default_behavior=DefaultBehavior.SINGLE.value,
                    channels=1,
                    metadata={"total_samples": total_samples},
                )
                signals.append([sig.to_dict()])
            arr.attrs["signals"] = signals

            # ---- attrs: schema, engine, collection ----
            arr.attrs["engine"] = engine
            arr.attrs["collection"] = collection
            arr.attrs["schema"] = dict(schema_dict)

            # save it
            self._store[path] = arr
            # blosc2.DictStore may or may not flush on __setitem__; call flush
            try:
                self._store.flush()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------
    def write(self,
              engine: str,
              obj_idx: int,
              track_name: str,
              audio_data: np.ndarray,
              sample_start: int = 0,
              signal_name: Optional[str] = None,
              signal_metadata: Optional[Dict[str, Any]] = None,
              metadata: Optional[Dict[str, Any]] = None) -> None:
        path = self._node_path(engine, obj_idx)
        if path not in self._store:
            raise BackendError(f"object node {path} not materialized")

        data_1d = np.ascontiguousarray(audio_data, dtype=np.float32).ravel()

        with self._lock:
            arr: blosc2.NDArray = self._store[path]
            obj_meta = arr.attrs["object"]
            total_samples = int(obj_meta["metadata"]["total_samples"])
            signal_type = obj_meta["metadata"].get("signal_type", "float32")
            np_dtype = np.dtype(np.float32 if signal_type == "float32" else np.float64)

            tracks_meta_list: List[Dict[str, Any]] = list(arr.attrs["tracks"])
            signals_meta_list: List[List[Dict[str, Any]]] = list(arr.attrs["signals"])

            # ---- locate track ----
            track_idx = next(
                (ti for ti, t in enumerate(tracks_meta_list) if t["name"] == track_name),
                None,
            )
            if track_idx is None:
                raise BackendError(f"track {track_name!r} not found in {path}")

            track_signals = signals_meta_list[track_idx]

            # ---- resolve target signal idx ----
            if signal_name is None:
                target_signal_idx = len(track_signals)
                new_signal = True
            else:
                target_signal_idx = next(
                    (si for si, s in enumerate(track_signals) if s["name"] == signal_name),
                    None,
                )
                if target_signal_idx is None:
                    target_signal_idx = len(track_signals)
                    new_signal = True
                else:
                    new_signal = False

            # ---- signal metadata ----
            if new_signal:
                new_name = signal_name if signal_name is not None else f"{track_name}_{target_signal_idx}"
                sig_meta = AudioDataType(
                    name=new_name,
                    id=target_signal_idx,
                    data_type=DataTypeKind.MONO.value,
                    default_behavior=DefaultBehavior.SINGLE.value,
                    channels=1,
                    sample_rate=int((metadata or {}).get("sample_rate", 48000)),
                    bit_depth=int((metadata or {}).get("bit_depth", 32)),
                    subtype=str((metadata or {}).get("subtype", "FLOAT")),
                    metadata={"total_samples": total_samples, **(signal_metadata or {})},
                )
            else:
                sig_meta = AudioDataType.from_dict(track_signals[target_signal_idx])
                if signal_metadata:
                    sig_meta.metadata.update(signal_metadata)

            # ---- if new signal, rebuild NDArray with one extra slot ----
            if new_signal:
                arr.resize((arr.shape[0], arr.shape[1] + 1, arr.shape[2]))

            # ---- padded chunk ----
            buf = np.zeros(total_samples, dtype=np_dtype)
            n = min(len(data_1d), max(0, total_samples - sample_start))
            if n > 0:
                buf[sample_start:sample_start + n] = data_1d[:n]

            # ---- write to slot ----
            # NOTE: for chunked writes without a new signal, honour overwrite
            # of just the chunk range.
            if new_signal or sample_start == 0 and n == total_samples:
                arr[0, target_signal_idx, :] = buf
            else:
                cur = arr[0, target_signal_idx, :]
                cur[sample_start:sample_start + n] = data_1d[:n]
                arr[0, target_signal_idx, :] = cur

            # ---- persist signal / track meta ----
            if new_signal:
                track_signals.append(sig_meta.to_dict())
            else:
                track_signals[target_signal_idx] = sig_meta.to_dict()
            signals_meta_list[track_idx] = track_signals
            arr.attrs["signals"] = signals_meta_list

            t = tracks_meta_list[track_idx]
            t["child_indices"] = [s["id"] for s in track_signals]
            t["child_names"] = [s["name"] for s in track_signals]
            arr.attrs["tracks"] = tracks_meta_list

            self._store[path] = arr
            try:
                self._store.flush()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------
    def read(self,
             engine: str,
             obj_idx: Optional[int] = None,
             track_name: Optional[str] = None,
             signal_name: Optional[str] = None,
             unprocessed: bool = False,
             side_channels: Optional[Dict[str, np.ndarray]] = None
             ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Read audio data + attrs tree.

        If ``unprocessed`` is True, returns raw float32 arrays (no DAG).
        Otherwise the DAG described by the AudioDataType.chains attrs is
        evaluated.
        """
        # -------- resolve scope by walking down the tree --------
        attrs_tree = self.read_tree()

        # No engine => walk to the top: whole scene.
        if engine is None:
            return self._read_scope(
                scope="scene",
                attrs_tree=attrs_tree,
                unprocessed=unprocessed,
                side_channels=side_channels,
            )

        if obj_idx is None:
            return self._read_scope(
                scope="engine",
                engine=engine,
                attrs_tree=attrs_tree,
                unprocessed=unprocessed,
                side_channels=side_channels,
            )

        if track_name is None:
            return self._read_scope(
                scope="object",
                engine=engine,
                obj_idx=int(obj_idx),
                attrs_tree=attrs_tree,
                unprocessed=unprocessed,
                side_channels=side_channels,
            )

        if signal_name is None:
            return self._read_scope(
                scope="track",
                engine=engine,
                obj_idx=int(obj_idx),
                track_name=track_name,
                attrs_tree=attrs_tree,
                unprocessed=unprocessed,
                side_channels=side_channels,
            )

        return self._read_scope(
            scope="signal",
            engine=engine,
            obj_idx=int(obj_idx),
            track_name=track_name,
            signal_name=signal_name,
            attrs_tree=attrs_tree,
            unprocessed=unprocessed,
            side_channels=side_channels,
        )

    # ------------------------------------------------------------------
    def _read_scope(self, scope: str,
                    attrs_tree: Dict[str, Any],
                    engine: Optional[str] = None,
                    obj_idx: Optional[int] = None,
                    track_name: Optional[str] = None,
                    signal_name: Optional[str] = None,
                    unprocessed: bool = False,
                    side_channels: Optional[Dict[str, np.ndarray]] = None
                    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Dispatch correctly to the right reader for the scope."""
        if scope == "scene":
            return self._read_scene(attrs_tree, unprocessed, side_channels)
        if scope == "engine":
            return self._read_engine(engine, attrs_tree, unprocessed, side_channels)
        if scope == "object":
            return self._read_object(engine, obj_idx, attrs_tree, unprocessed, side_channels)
        if scope == "track":
            return self._read_track(engine, obj_idx, track_name, attrs_tree,
                                    unprocessed, side_channels)
        if scope == "signal":
            return self._read_signal(engine, obj_idx, track_name, signal_name,
                                     attrs_tree, unprocessed, side_channels)
        raise BackendError(f"unknown scope {scope!r}")

    # ------------------------------------------------------------------
    # Low-level: read the raw (1, S, N) NDArray for one object
    # ------------------------------------------------------------------
    def _read_ndarray(self, engine: str, obj_idx: int) -> Tuple[np.ndarray, blosc2.NDArray]:
        path = self._node_path(engine, obj_idx)
        if path not in self._store:
            raise BackendError(f"object node {path} not found")
        arr = self._store[path]
        # force-load into a contiguous numpy array
        data = np.ascontiguousarray(arr[:])
        return data, arr

    # ------------------------------------------------------------------
    # Signal scope: (1, S, N) -> (n_tracks... index track, index signal)
    # ------------------------------------------------------------------
    def _read_signal(self, engine, obj_idx, track_name, signal_name,
                     attrs_tree, unprocessed, side_channels):
        data, arr = self._read_ndarray(engine, obj_idx)
        tracks_meta = list(arr.attrs["tracks"])
        signals_meta = list(arr.attrs["signals"])

        track_idx = next((ti for ti, t in enumerate(tracks_meta)
                          if t["name"] == track_name), None)
        if track_idx is None:
            raise BackendError(f"track {track_name!r} not found")
        sig_meta_track = signals_meta[track_idx]
        sig_idx = next((si for si, s in enumerate(sig_meta_track)
                        if s["name"] == signal_name), None)
        if sig_idx is None:
            raise BackendError(f"signal {signal_name!r} not found in track {track_name!r}")

        sig_arr = data[0, sig_idx, :].reshape(1, -1).astype(np.float32)

        # Build pruned attrs (signal-level)
        pruned = {
            "scope": "signal",
            "engine": engine,
            "obj_idx": int(obj_idx),
            "track_name": track_name,
            "signal_name": signal_name,
            "signal": dict(sig_meta_track[sig_idx]),
        }

        if unprocessed:
            return sig_arr, self._prune_chains(pruned)

        # DAG path: run chains for this specific signal
        sig_meta = AudioDataType.from_dict(sig_meta_track[sig_idx])
        out = self._evaluate_chains(sig_arr, sig_meta, props_side=sideside_channels)
        return out, pruned

    # ------------------------------------------------------------------
    # Track scope: aggregate track's signals signals per track AudioDataType
    # ------------------------------------------------------------------
    def _read_track(self, engine, obj_idx, track_name, attrs_tree,
                    unprocessed, side_channels):
        data, arr = self._read_ndarray(engine, obj_idx)
        tracks_meta = list(arr.attrs["tracks"])
        signals_meta = list(arr.attrs["signals"])

        track_idx = next((ti for ti, t in enumerate(tracks_meta)
                          if t["name"] == track_name), None)
        if track_idx is None:
            raise BackendError(f"track {track_name!r} not found")

        track_node = AudioDataType.from_dict(tracks_meta[track_idx])
        sigs = signals_meta[track_idx]

        sig_arrays = [
            data[0, s["id"], :].reshape(1, -1).astype(np.float32)
            for s in sigs
        ]

        combined = _stack_signals(sig_arrays, track_node.default_behavior)
        if combined is None:
            combined = np.zeros((1, data.shape[2]), dtype=np.float32)

        pruned = {
            "scope": "track",
            "engine": engine,
            "obj_idx": int(obj_idx),
            "track_name": track_name,
            "track": tracks_meta[track_idx],
            "signals": [dict(s) for s in sigs],
        }

        if unprocessed:
            return combined, self._prune_chains(pruned)

        # chain at track level
        out = self._evaluate_chains(combined, track_node, props_side=side_channels)
        return out, pruned

    # ------------------------------------------------------------------
    # Object scope: aggregate tracks per object AudioDataType
    # ------------------------------------------------------------------
    def _read_object(self, engine, obj_idx, attrs_tree, unprocessed, side_channels):
        data, arr = self._read_ndarray(engine, obj_idx)
        obj_node = AudioDataType.from_dict(arr.attrs["object"])
        tracks_meta = list(arr.attrs["tracks"])
        signals_meta = list(arr.attrs["signals"])

        track_arrays = []
        for ti, t_meta in enumerate(tracks_meta):
            t_node = AudioDataType.from_dict(t_meta)
            sigs = signals_meta[ti]
            sig_arrays = [data[0, s["id"], :].reshape(1, -1).astype(np.float32)
                          for s in sigs]
            combined_track = _stack_signals(sig_arrays, t_node.default_behavior)
            if combined_track is None:
                continue
            track_arrays.append(combined_track)

        combined = _stack_signals(track_arrays, obj_node.default_behavior)
        if combined is None:
            combined = np.zeros((1, data.shape[2]), dtype=np.float32)

        pruned = {
            "scope": "object",
            "engine": engine,
            "obj_idx": int(obj_idx),
            "object": arr.attrs["object"],
            "tracks": [dict(t) for t in tracks_meta],
            "signals": [list(s) for s in signals_meta],
        }

        if unprocessed:
            return combined, self._prune_chains(pruned)

        out = self._evaluate_chains(combined, obj_node, props_side=side_channels)
        return out, pruned

    # ------------------------------------------------------------------
    # Engine scope: aggregate objects per engine AudioDataType
    # ------------------------------------------------------------------
    def _read_engine(self, engine, attrs_tree, unprocessed, side_channels):
        engine_node = attrs_tree.get("engines", {}).get(engine, {})
        engine_behavior = engine_node.get("default_behavior", DefaultBehavior.MULTI.value)

        if engine_behavior == DefaultBehavior.NONE.value:
            return None, self._prune_chains({
                "scope": "engine", "engine": engine, "engine_node": engine_node,
            })

        arrays = []
        attrs_objects = []
        # We must know which obj_idx live under this engine.
        # Walk the store: keys are "<collection>/<engine>/<obj_idx>"
        prefix = f"{self.collection}/{engine}/"
        for key in list(self._store.keys()):
            if not key.startswith(prefix):
                continue
            tail = key[len(prefix):]
            if "/" in tail:
                continue
            if not tail.lstrip("-").isdigit():
                continue
            o_idx = int(tail)
            arr_o, attrs_o = self._read_object(engine, o_idx, attrs_tree,
                                               unprocessed=True, side_channels=None)
            if arr_o is not None:
                arrays.append(arr_o)
            attrs_objects.append(attrs_o)

        combined = _stack_signals(arrays, engine_behavior)
        if combined is None:
            combined = np.zeros((1, 0), dtype=np.float32)

        pruned = {
            "scope": "engine",
            "engine": engine,
            "engine_node": engine_node,
            "objects": attrs_objects,
        }

        if unprocessed:
            return combined, self._prune_chains(pruned)

        eng_adt = AudioDataType.from_dict({
            "name": engine,
            "data_type": engine_node.get("data_type", DataTypeKind.MULTI_TRACK.value),
            "default_behavior": engine_behavior,
            "chains": engine_node.get("chains"),
        })
        out = self._evaluate_chains(combined, eng_adt, props_side=side_channels)
        return out, pruned

    # ------------------------------------------------------------------
    # Scene scope: aggregate collections
    # ------------------------------------------------------------------
    def _read_scene(self, attrs_tree, unprocessed, side_channels):
        top = attrs_tree.get("scene", {})
        top_behavior = top.get("default_behavior", DefaultBehavior.MULTI.value)

        arrays = []
        engines_attrs = []
        engines = attrs_tree.get("engines", {})
        for eng_name in engines.keys():
            arr_e, attrs_e = self._read_engine(eng_name, attrs_tree, True, None) \
                if False else self._read_engine(eng_name, attrs_tree, unprocessed=True, side_channels=None)
            engines_attrs.append(attrs_e)
            if arr_e is not None:
                arrays.append(arr_e)

        combined = _stack_signals(arrays, top_behavior)
        if combined is None:
            combined = np.zeros((1, 0), dtype=np.float32)

        pruned = {
            "scope": "scene",
            "scene": top,
            "engines": engines_attrs,
        }

        if unprocessed:
            return combined, self._prune_chains(pruned)

        scene_adt = AudioDataType.from_dict({
            "name": "scene",
            "data_type": top.get("data_type", DataTypeKind.MULTI_TRACK.value),
            "default_behavior": top_behavior,
            "chains": top.get("chains"),
        })
        out = self._evaluate_chains(combined, scene_adt, props_side=side_channels)
        return out, pruned

    # ------------------------------------------------------------------
    # DAG evaluation
    # ------------------------------------------------------------------
    def _evaluate_chains(self, x: np.ndarray,
                         adt: AudioDataType,
                         props_side: Optional[Dict[str, np.ndarray]]) -> np.ndarray:
        """
        Apply the AudioDataType.chains to the incoming signal.
        """
        if not adt.chains:
            return x

        # Lazy import: pbrAudioSignals depends on nothing from pbrAudioStorage,
        # so importing here keeps storage-only users free of the dep.
        try:
            from pbrAudioSignals import build_chain
        except ImportError:
            # No signal processors installed -> return the input unchanged.
            return x

        chain = build_chain(adt.chains.get("processors", []))

        # Injection channel: side_channels go into props.custom
        props = Props(custom={})
        if props_side:
            for k, v in props_side.items():
                props.custom[k] = v
        # Also inject any chain-level metadata the caller put on the node
        if adt.metadata:
            props.custom.setdefault("metadata", {}).update(adt.metadata)

        out = chain.process(x, props)
        return np.ascontiguousarray(out, dtype=np.float32)

    # ------------------------------------------------------------------
    # Chain pruning (produces the "unprocessed" attrs tree)
    # ------------------------------------------------------------------
    def _prune_chains(self, attrs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Return a copy of *attrs* with all chains definitions stripped
        and marked as unprocessed.
        """
        def _walk(o):
            if isinstance(o, dict):
                out = {}
                for k, v in o.items():
                    if k == "chains":
                        out["chains_present"] = bool(v)
                        continue
                    out[k] = _walk(v)
                return out
            if isinstance(o, list):
                return [_walk(v) for v in o]
            return o
        pruned = _walk(copy.deepcopy(attrs))
        pruned["unprocessed"] = True
        return pruned

    # ------------------------------------------------------------------
    # attrs tree reads
    # ------------------------------------------------------------------
    def read_tree(self) -> Dict[str, Any]:
        """
        Return the metadata tree of all registered nodes.
        Structure:
          {
            "scene": {...},
            "collections": {"<name>": {...}},
            "engines": {"<engine>": {...}},
            "objects": {"<engine>/<obj_idx>": {attrs}}
          }
        """
        tree: Dict[str, Any] = {
            "scene": {
                "collection": self.collection,
                "data_type": DataTypeKind.MULTI_TRACK.value,
                "default_behavior": DefaultBehavior.MULTI.value,
                "chains": None,
            },
            "collections": {self.collection: {}},
            "engines": {},
            "objects": {},
        }

        prefix = f"{self.collection}/"
        for key in list(self._store.keys()):
            if not key.startswith(prefix):
                continue
            tail = key[len(prefix):]
            parts = tail.split("/")
            if len(parts) < 2:
                continue
            eng = parts[0]
            o_str = parts[-1]
            if not o_str.lstrip("-").isdigit():
                continue
            o_idx = int(o_str)

            arr = self._store[key]
            tree["objects"][f"{eng}/{o_idx}"] = {
                "object": arr.attrs.get.get("object"),
                "tracks": arr.attrs.get("tracks"),
                "signals": arr.attrs.get("signals"),
                "schema": arr.attrs.get("schema"),
                "engine": arr.attrs.get("engine"),
                "collection": arr.attrs.get("collection"),
            }
            tree["engines"].setdefault(eng, {
                "name": eng,
                "data_type": DataTypeKind.MULTI_TRACK.value,
                "default_behavior": DefaultBehavior.MULTI.value,
                "chains": None,
            })

        return tree

    # ------------------------------------------------------------------
    # edit_tree (JSON-Patch subset)
    # ------------------------------------------------------------------
    def edit_tree(self, payload: Any) -> Dict[str, Any]:
        """
        Apply a JSON-Patch subset to the metadata tree stored in attrs.

        Supported ops: "replace", "add", "remove".
        Paths are dotted with the first segment being:
          - "scene"
          - "engines.<engine>"
          - "objects.<engine>.<obj_idx>.object"
          - "objects.<engine>.<obj_idx>.tracks.<track_idx>.<field>"
          - "objects.<engine>.<obj_idx>.signals.<track_idx>.<signal_idx>.<field>"

        Returns
        -------
        {"status": "ok", "applied": [...]} or {"status": "error", "detail": "..."}
        """
        if not isinstance(payload, list):
            return {"status": "error", "detail": "payload must be a list of patches"}

        applied: List[Dict[str, Any]] = []
        for i, patch in enumerate(payload):
            if not isinstance(patch, dict) or "op" not in patch or "path" not in patch:
                return {"status": "error", "detail": f"patch #{i}: malformed",
                        "applied": applied}
            try:
                self._apply_patch(patch)
                applied.append(patch)
            except PatchError as e:
                return {"status": "error", "detail": f"patch #{i}: {e}",
                        "applied": applied}

        try:
            self._store.flush()
        except Exception:
            pass
        return {"status": "ok", "applied": applied}

    # ------------------------------------------------------------------
    def _parse_path(self, path: str) -> List[Any]:
        if not path.startswith("/"):
            raise PatchError("path must start with '/'")
        parts = [p for p in path[1:].split("/") if p != ""]
        if not parts:
            raise PatchError("empty path")
        return parts

    def _apply_patch(self, patch: Dict[str, Any]) -> None:
        op = patch["op"]
        path = self._parse_path(patch["path"])
        if op not in ("replace", "add", "remove"):
            raise PatchError(f"unsupported op {op!r}")

        head = path[0]
        if head == "scene":
            # scene lives under attrs at the first object we find; we
            # keep a synthetic "scene" dict in the collection entry.
            self._edit_scene(path[1:], op, patch.get("value"))
            return

        if head == "engines":
            self._edit_engine(path[1:], op, patch.get("value"))
            return

        if head == "objects":
            self._edit_object(path[1:], op, patch.get("value"))
            return

        raise PatchError(f"unknown path head {head!r}")

    def _edit_scene(self, rest: List[str], op: str, value: Any) -> None:
        # scene metadata is not persisted in blosc2 attrs directly;
        # we store it under the special key "<collection>/__scene__".
        key = f"{self.collection}/__scene__"
        if key not in self._store:
            self._store[key] = {
                "data_type": DataTypeKind.MULTI_TRACK.value,
                "default_behavior": DefaultBehavior.MULTI.value,
                "chains": None,
            }
        scene = self._store[key]
        if not isinstance(scene, dict):
            raise PatchError("scene entry is not a dict")
        self._set_dict_path(scene, rest, op, value)
        self._store[key] = scene

    def _edit_engine(self, rest: List[str], op: str, value: Any) -> None:
        if not rest:
            raise PatchError("missing engine name")
        eng = rest[0]
        key = f"{self.collection}/__engine__/{eng}"
        if key not in self._store:
            self._store[key] = {
                "name": eng,
                "data_type": DataTypeKind.MULTI_TRACK.value,
                "default_behavior": DefaultBehavior.MULTI.value,
                "chains": None,
            }
        ent = self._store[key]
        self._set_dict_path(ent, rest[1:], op, value)
        self._store[key] = ent

    def _edit_object(self, rest: List[str], op: str, value: Any) -> None:
        if len(rest) < 2:
            raise PatchError("object path must be /objects/<engine>/<obj_idx>/...")
        eng = rest[0]
        try:
            obj_idx = int(rest[1])
        except ValueError:
            raise PatchError(f"obj_idx must be integer, got {rest[1]!r}")
        path = self._node_path(eng, obj_idx)
        if path not in self._store:
            raise PatchError(f"object {eng}/{obj_idx} not found")
        arr = self._store[path]
        rest2 = rest[2:]
        if not rest2:
            raise PatchError("missing object-level field")
        section = rest2[0]
        if section == "object":
            node = dict(arr.attrs["object"])
            self._set_dict_path(node, rest2[1:], op, value)
            arr.attrs["object"] = node
        elif section == "tracks":
            self._edit_object_track(arr, rest2[1:], op, value)
        elif section == "signals":
            self._edit_object_signal(arr, rest2[1:], op, value)
        else:
            raise PatchError(f"unknown section {section!r}")

        self._store[path] = arr

    def _edit_object_track(self, arr, rest: List[str], op: str, value: Any) -> None:
        if not rest:
            raise PatchError("missing track index")
        try:
            track_idx = int(rest[0])
        except ValueError:
            raise PatchError(f"track index must be integer, got {rest[0]!r}")
        tracks = [dict(t) for t in arr.attrs["tracks"]]
        if track_idx < 0 or track_idx >= len(tracks):
            raise PatchError(f"track index {track_idx} out of range [0,{len(tracks)})")
        t = tracks[track_idx]
        self._set_dict_path(t, rest[1:], op, value)
        tracks[track_idx] = t
        arr.attrs["tracks"] = tracks

    def _edit_object_signal(self, arr, rest: List[str], op: str, value: Any) -> None:
        if len(rest) < 2:
            raise PatchError("signal path must be /signals/<track_idx>/<signal_idx>/...")
        try:
            track_idx = int(rest[0])
            sig_idx = int(rest[1])
        except ValueError:
            raise PatchError(f"track/signal indices must be integers, got {rest[:2]!r}")
        signals = [list(s) for s in arr.attrs["signals"]]
        if track_idx < 0 or track_idx >= len(signals):
            raise PatchError(f"track index {track_idx} out of range")
        track_sigs = signals[track_idx]
        if sig_idx < 0 or sig_idx >= len(track_sigs):
            raise PatchError(f"signal index {sig_idx} out of range for track {track_idx}")
        s = dict(track_sigs[sig_idx])
        self._set_dict_path(s, rest[2:], op, value)
        track_sigs[sig_idx] = s
        signals[track_idx] = track_sigs
        arr.attrs["signals"] = signals

    # ------------------------------------------------------------------
    @staticmethod
    def _set_dict_path(node: Dict[str, Any], parts: List[str],
                       op: str, value: Any) -> None:
        """
        Walk ``parts`` on the dict ``node`` and apply the patch op at the
        final key. Intermediate dicts are auto-created for "add".
        """
        cur: Any = node
        for i, p in enumerate(parts[:-1]):
            if isinstance(cur, dict):
                if p not in cur:
                    if op == "add":
                        cur[p] = {}
                    else:
                        raise PatchError(f"path segment {p!r} missing at depth {i}")
                cur = cur[p]
            elif isinstance(cur, list):
                try:
                    idx = int(p)
                    cur = cur[idx]
                except (ValueError, IndexError):
                    raise PatchError(f"invalid list index {p!r}")
            else:
                raise PatchError(f"cannot descend into non-container at {p!r}")

        if not parts:
            raise PatchError("empty patch path")
        last = parts[-1]

        if isinstance(cur, dict):
            if op in ("replace", "remove") and last not in cur:
                raise PatchError(f"key {last!r} not present for {op}")
            if op == "remove":
                del cur[last]
            else:
                cur[last] = value
        elif isinstance(cur, list):
            try:
                idx = int(last)
            except ValueError:
                raise PatchError(f"invalid list index {last!r}")
            if op == "remove":
                if idx < 0 or idx >= len(cur):
                    raise PatchError(f"index {idx} out of range")
                del cur[idx]
            elif op == "add":
                if idx < 0 or idx > len(cur):
                    raise PatchError(f"index {idx} out of range for add")
                cur.insert(idx, value)
            else:  # replace
                if idx < 0 or idx >= len(cur):
                    raise PatchError(f"index {idx} out of range for replace")
                cur[idx] = value
        else:
            raise PatchError(f"cannot patch leaf at {last!r}")

    # ------------------------------------------------------------------
    # DAG builder cache / public helper used by StorageEngine
    # ------------------------------------------------------------------
    def build_chains_for_node(self, adt: AudioDataType) -> Optional[DagBuilder]:
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

    # ------------------------------------------------------------------
    def close(self) -> None:
        self._handle_cache.clear()
        self._schemas.clear()
        self._chains_cache.clear()

