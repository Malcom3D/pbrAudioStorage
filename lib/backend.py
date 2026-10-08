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
from typing import Any, Dict, List, Optional, Callable

import blosc2
import numpy as np
from numba import njit

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
        with self._lock:
            if self._store is None:
                if not os.path.exists(self.path):
                    os.makedirs(self.path, exist_ok=True)
                    _store = blosc2.TreeStore(self.path, mode='w')
                    _store.close()
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

    def materialize(self, engine: str, obj_indices: List[int], track_names: List[str], total_samples: int, signal_type: str, metadata: Optional[Dict[str, Any]] = None, dtype: np.dtype = np.float32) -> None:
        """
        Pre-allocate one NDArray per object with a single signal slot.
        Idempotent: if the node already exists we leave it alone.
        """
        n_tracks = len(track_names)
        n_signals = 1  # Start with one signal slot
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
                }

                # track-level meta
                tracks_meta = []
                for t_idx, t_name in enumerate(track_names):
                    tracks_meta.append(
                        {
                            "track_idx": t_idx,
                            "track_name": t_name,
                            "signal_names": [None],  # Initially one unnamed signal
                        }
                    )
                    # signal-level meta, keyed by track name
                    arr.attrs[t_name] = [
                        {
                            "signal_idx": 0,
                            "signal_name": None,
                            "signal_type": signal_type,
                            **meta,
                        }
                    ]
                arr.attrs["tracks"] = tracks_meta

                store[node] = arr

    def write_signal(self, data: np.ndarray, engine: str, obj_idx: int, track_name: str, signal_name: Optional[Any] = None, metadata: Optional[Dict[str, Any]] = None, sample_start: Optional[int] = None) -> None:
        """
        Write `data` into a track's signal slot.
        If `signal_name` is None or not found, a new signal slot is added.
        """
        with self._lock:
            store = self.store
            node = self._node_path(engine, obj_idx)
            arr = store[node]

            # resolve track index
            obj_meta = arr.attrs["object"]
            try:
                track_index = list(obj_meta["track_names"]).index(track_name)
            except ValueError as exc:
                raise KeyError(f"track_name {track_name!r} not registered for engine={engine} obj_idx={obj_idx}") from exc

            total_samples = arr.shape[2]

            # normalise input to (C, S)
            if data.ndim == 1:
                data = data.reshape(1, -1)
            data = np.ascontiguousarray(data, dtype=np.float32)
            C_in, S_in = data.shape

            # Get track metadata and find/create signal index
            track_meta_list = arr.attrs['tracks']
            track_meta = track_meta_list[track_index]
            
            if signal_name is None:
                # Append a new signal
                signal_index = len(track_meta['signal_names'])
                track_meta['signal_names'].append(None) # Append unnamed signal
                arr.attrs['tracks'] = track_meta_list # Update attrs
                
                # Resize the NDArray
                new_shape = (arr.shape[0], arr.shape[1] + 1, arr.shape[2])
                arr.resize(new_shape)
                
                # Add signal-specific metadata
                signal_meta_list = arr.attrs[track_name]
                signal_meta_list.append({"signal_idx": signal_index, "signal_name": None})
                arr.attrs[track_name] = signal_meta_list

            else:
                try:
                    signal_index = track_meta['signal_names'].index(signal_name)
                except ValueError:
                    # Not found, append a new signal with the given name
                    signal_index = len(track_meta['signal_names'])
                    track_meta['signal_names'].append(signal_name)
                    arr.attrs['tracks'] = track_meta_list
                    
                    # Resize the NDArray
                    new_shape = (arr.shape[0], arr.shape[1] + 1, arr.shape[2])
                    arr.resize(new_shape)

                    # Add signal-specific metadata
                    signal_meta_list = arr.attrs[track_name]
                    signal_meta_list.append({"signal_idx": signal_index, "signal_name": signal_name})
                    arr.attrs[track_name] = signal_meta_list

            # Now write the data to the resolved/created signal_index
            if sample_start is None:
                if S_in > total_samples:
                    data = data[:, :total_samples]
                row = np.asarray(arr[track_index, signal_index, :], dtype=np.float32).reshape(1, -1)
                if C_in == 1:
                    _copy_into(row, data)
                else:
                    _copy_into(row, data[:1, :]) # Downmix to mono
                arr[track_index, signal_index, :] = row[0]
            else:
                if sample_start < 0:
                    raise ValueError("sample_start must be >= 0")
                if sample_start >= total_samples:
                    return
                buf = np.zeros((1, total_samples), dtype=np.float32) # Always mono for storage
                _pad_into(buf, data[:1, :], int(sample_start))
                arr[track_index, signal_index, :] = buf[0]

            # Update metadata if provided
            if metadata is not None:
                signal_meta_list = arr.attrs[track_name]
                # Update the specific signal's metadata
                signal_meta_list[signal_index].update(metadata)
                arr.attrs[track_name] = signal_meta_list


    def get_ndarray(self, engine: str, obj_idx: int) -> blosc2.NDArray:
        with self._lock:
            return self.store[self._node_path(engine, obj_idx)]

    def read(self, engine: str, obj_idx: int, track_index: int, signal_name: Optional[str] = None, start: int = 0, stop: Optional[int] = None) -> Optional[np.ndarray]:
        """
        Reads a signal signal from a track. If signal_name is None or not found,
        it sums all signals in the track, applying a processing chain if specified
        in the metadata.

        Args:
            engine: The name of the engine (e.g., 'physicsSolver').
            obj_idx: The object index.
            track_index: The index of the track to read from.
            signal_name: The name of the specific signal to read. If None, all
                         signals in the track are combined.
            start: Start sample for partial reads.
            stop: End sample for partial reads.

        Returns:
            A numpy array of the audio data, or None if the track/signal is not found.
        """
        with self._lock:
            node_path = self._node_path(engine, obj_idx)
            if node_path not in self.store:
                print(f"Warning: Node '{node_path}' not found.")
                return None
            arr = self.store[node_path]

            # Get track metadata
            track_meta = arr.attrs['tracks'][track_index]
            signal_names = track_meta.get('signal_names', [])
            track_name = track_meta['track_name']

            # Find signal index by name if provided
            signal_index = -1
            if signal_name is not None:
                try:
                    signal_index = signal_names.index(signal_name)
                except ValueError:
                    # If signal_name is provided but not found, we fall back to summing.
                    print(f"Warning: signal_name '{signal_name}' not found in track '{track_name}'. Summing all signals.")

            # If a specific signal is found, read and return it
            if signal_index != -1:
                sl = arr[track_index, signal_index, start:stop]
                return np.asarray(sl, dtype=np.float32).reshape(1, -1)

            # --- Fallback logic: sum all signals in the track ---
            n_signals = arr.shape[1]
            if n_signals == 0:
                return np.zeros((1, (stop or arr.shape[2]) - start), dtype=np.float32)

            # Fetch all signals in the track
            all_signals = arr[track_index, :, start:stop]  # Shape: (n_signals, n_samples)

            # Check for a processing chain in metadata
            # The metadata is stored per-signal, but a chain would likely be a track-level property.
            # We'll look in the first signal's metadata for a 'chains' definition.
            signal_meta = arr.attrs.get(track_name, [])
            processing_chain = None
            if signal_meta and isinstance(signal_meta, list) and len(signal_meta) > 0:
                processing_chain = signal_meta[0].get('chains')

            if processing_chain:
                # Apply the DAG processing chain
                return self._apply_processing_chain(all_signals, processing_chain)
            else:
                # Default behavior: sum all signals
                return np.sum(all_signals, axis=0, keepdims=True)

    def _apply_processing_chain(self, signals: np.ndarray, chain: Dict[str, Any]) -> np.ndarray:
        """
        Applies a processing chain defined in metadata to the signals.

        Args:
            signals: A numpy array of shape (n_signals, n_samples).
            chain: A dictionary defining the processing steps.
                   Example: {'op': 'mix', 'indices': [0, 1], 'mode': 'sum'}

        Returns:
            A numpy array of shape (1, n_samples) representing the processed signal.
        """
        op = chain.get('op')
        if op == 'mix':
            indices = chain.get('indices', [])
            mode = chain.get('mode', 'sum')
            
            # Ensure indices are valid
            valid_indices = [i for i in indices if 0 <= i < signals.shape[0]]
            if not valid_indices:
                return np.zeros((1, signals.shape[1]), dtype=np.float32)

            signals_to_mix = signals[valid_indices]

            if mode == 'sum':
                return np.sum(signals_to_mix, axis=0, keepdims=True)
            elif mode == 'mean':
                return np.mean(signals_to_mix, axis=0, keepdims=True)
            else:
                print(f"Warning: Unsupported mix mode '{mode}' in processing chain. Defaulting to sum.")
                return np.sum(signals_to_mix, axis=0, keepdims=True)
        
        # Default fallback if op is unknown or not provided
        print(f"Warning: Unsupported processing op '{op}'. Summing all signals.")
        return np.sum(signals, axis=0, keepdims=True)

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
