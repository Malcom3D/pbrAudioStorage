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
import threading
from typing import Any, Dict, List, Optional

import blosc2
import numpy as np
from numba import njit

# Numba kernels — SIMD friendly, no parallel=True (no threads allowed).
@njit(cache=True, fastmath=True, nogil=True)
def _pad_into(dst: np.ndarray, src: np.ndarray, start: int) -> None:
    """
    dst : (C, T)  float32, zero-initialised
    src : (C, S)  float32, S <= T - start
    Copies src into dst[:, start:start+S]. Truncates if S > T - start.
    """
    C, T = dst.shape
    Sc = src.shape[0]
    S = src.shape[1]
    c = C if C < Sc else Sc
    avail = T - start
    n = S if S < avail else avail
    for ch in range(c):
        for i in range(n):
            dst[ch, start + i] = src[ch, i]


@njit(cache=True, fastmath=True, nogil=True)
def _copy_into(dst: np.ndarray, src: np.ndarray) -> None:
    """dst[:, :] = src[:, :] with shape check on the min."""
    C, T = dst.shape
    Sc, Ss = src.shape
    c = C if C < Sc else Sc
    n = T if T < Ss else Ss
    for ch in range(c):
        for i in range(n):
            dst[ch, i] = src[ch, i]


class Blosc2Backend:
    """
    Single owner of a blosc2.TreeStore handle. All StorageEngine instances
    that target the same path go through the same backend instance.

    Layout inside the TreeStore (matches StorageEngine.register):
        /<engine>/<obj_idx>   ->  NDArray (n_tracks, n_signals, total_samples)
                                   attrs: 'object'   -> object meta
                                          'tracks'   -> list of track meta
                                          '<track_name>' -> list of signal meta
    """

    _instances: Dict[str, "Blosc2Backend"] = {}
    _instances_lock = threading.Lock()

    def __init__(self, path: str, mode: str = "a"):
        self.path = path
        self.mode = mode
        self._lock = threading.RLock()
        self._store: Optional[blosc2.TreeStore] = None
        self._open()

    def _open(self) -> None:
        # blosc2.TreeStore(path, mode=...)  — 'a' creates if missing.
        # We open lazily and keep the handle for the process lifetime.
        with self._lock:
            if self._store is None:
                if not os.path.exists(f"{self.path}/embed.b2e"):
                    if not os.path.dirname(self.path):
                        os.makedirs(self.path, exist_ok=True)
                    _store = blosc2.TreeStore(self.path, mode='w')
                    _store.close()
                if os.path.dirname(self.path) and os.path.exists(f"{self.path}/embed.b2e"):
                    self._store = blosc2.TreeStore(self.path, mode=self.mode)

    @classmethod
    def get(cls, path: str, mode: str = "a") -> "Blosc2Backend":
        key = os.path.abspath(path)
        with cls._instances_lock:
            inst = cls._instances.get(key)
            if inst is None:
                inst = cls(path, mode=mode)
                cls._instances[key] = inst
            return inst

    @classmethod
    def close_all(cls) -> None:
        with cls._instances_lock:
            for inst in cls._instances.values():
                inst.close()
            cls._instances.clear()

    def close(self) -> None:
        with self._lock:
            if self._store is not None:
                try:
                    self._store.close()
                finally:
                    self._store = None

    @property
    def store(self) -> blosc2.TreeStore:
        if self._store is None:
            self._open()
        return self._store

    @staticmethod
    def _node_path(engine: str, obj_idx: int) -> str:
        return f"/{engine}/{obj_idx}"

    def materialize(self, engine: str, obj_indices: List[int], track_names: List[str], signal_names: List[Any], total_samples: int, signal_type: str, metadata: Optional[Dict[str, Any]] = None, dtype: np.dtype = np.float32) -> None:
        """
        Pre-allocate one NDArray per object: (n_tracks, n_signals, total_samples).
        Idempotent: if the node already exists we leave it alone.
        """
        n_tracks = len(track_names)
        n_signals = len(signal_names)
        shape = (n_tracks, n_signals, total_samples)

        meta = dict(metadata or {})
        meta.setdefault("signal_type", signal_type)

        with self._lock:
            store = self.store
            for obj_idx in obj_indices:
                node = self._node_path(engine, obj_idx)
                if node in store:
                    continue

                arr = blosc2.zeros(shape, dtype=dtype)

                # object-level meta
                arr.attrs["object"] = {
                    "obj_idx": obj_idx,
                    "track_names": list(track_names),
                    "total_samples": int(total_samples),
                    "n_tracks": n_tracks,
                    "n_signals": n_signals,
                }

                # track-level meta
                tracks_meta = []
                for t_idx, t_name in enumerate(track_names):
                    tracks_meta.append(
                        {
                            "track_idx": t_idx,
                            "track_name": t_name,
                            "n_signals": n_signals,
                            "signal_names": list(signal_names),
                        }
                    )
                    # signal-level meta, keyed by track name
                    arr.attrs[t_name] = [
                        {
                            "signal_idx": s_idx,
                            "signal_name": signal_names[s_idx],
                            "signal_type": signal_type,
                            **meta,
                        }
                        for s_idx in range(n_signals)
                    ]
                arr.attrs["tracks"] = tracks_meta

                store[node] = arr

    def write_signal(self, data: np.ndarray, engine: str, obj_idx: int, track_name: str, signal_index: int, metadata: Optional[Dict[str, Any]] = None, sample_start: Optional[int] = None) -> None:
        """
        Write `data` (shape (C, S) or (S,)) into
            /<engine>/<obj_idx>[track_idx, signal_index, :]
        If sample_start is None, the data is written from index 0.
        Otherwise the signal is zero-padded to total_samples and the chunk
        is placed at [sample_start : sample_start + S].
        """
        signal_saved = False
        with self._lock:
            store = self.store
            node = self._node_path(engine, obj_idx)
            arr = store[node]

            # resolve track index
            obj_meta = arr.attrs["object"]
            try:
                track_index = list(obj_meta["track_names"]).index(track_name)
            except ValueError as exc:
                raise KeyError(
                    f"track_name {track_name!r} not registered for "
                    f"engine={engine} obj_idx={obj_idx}"
                ) from exc

            C, T, S_total = arr.shape[1], arr.shape[2], arr.shape[2]
            # arr shape is (n_tracks, n_signals, total_samples)
            total_samples = arr.shape[2]

            # normalise input to (C, S)
            if data.ndim == 1:
                data = data.reshape(1, -1)
            data = np.ascontiguousarray(data, dtype=np.float32)
            C_in, S_in = data.shape

            if sample_start is None:
                # contiguous write from 0; must fit in total_samples
                if S_in > total_samples:
                    data = data[:, :total_samples]
                    S_in = total_samples
                # read existing row, patch, write back
                row = np.asarray(arr[track_index, signal_index, :], dtype=np.float32).reshape(1, -1)
                # row shape is (1, total_samples); broadcast if C_in > 1
                if C_in == 1:
                    _copy_into(row, data)
                else:
                    # multichannel into mono storage: downmix by first channel
                    _copy_into(row, data[:1, :])
                arr[track_index, signal_index, :] = row[0]
                signal_saved = True
            else:
                # padded chunk write
                if sample_start < 0:
                    raise ValueError("sample_start must be >= 0")
                if sample_start >= total_samples:
                    return  # nothing to write

                # build a zeroed (C_in, total_samples) buffer
                buf = np.zeros((C_in, total_samples), dtype=np.float32)
                _pad_into(buf, data, int(sample_start))

                # write each channel; storage is mono per signal, so we take
                # channel 0 unless the caller explicitly passes multichannel
                # and storage was allocated multichannel — but our schema is
                # mono-per-signal, so we downmix by channel 0.
                arr[track_index, signal_index, :] = buf[0]
                signal_saved = True

            if signal_saved:
                signal_name = arr.attrs['tracks']['signal_names'][signal_index]
                arr.attrs['tracks'][signal_name] = metadata

    def get_ndarray(self, engine: str, obj_idx: int) -> blosc2.NDArray:
        with self._lock:
            return self.store[self._node_path(engine, obj_idx)]

    def read_signal(self, engine: str, obj_idx: int, track_index: int, signal_index: int)-> np.ndarray:
        with self._lock:
            arr = self.store[self._node_path(engine, obj_idx)]
            sl = arr[track_index, signal_index,:]
            return np.asarray(sl, dtype=np.float32).reshape(1, -1)

    def attrs(self, engine: str, obj_idx: int) -> Dict[str, Any]:
        with self._lock:
            arr = self.store[self._node_path(engine, obj_idx)]
            return dict(arr.attrs)

    def list_objects(self, engine: str) -> List[int]:
        with self._lock:
            out: List[int] = []
            prefix = f"/{engine}/"
            for key in self.store:
                if key.startswith(prefix):
                    try:
                        out.append(int(key[len(prefix):]))
                    except ValueError:
                        continue
            return sorted(out)
