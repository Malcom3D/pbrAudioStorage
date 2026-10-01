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
from dataclasses import dataclass, field
from typing import Iterable, Dict, List, Tuple, Union
import numpy as np
import blosc2

from pbrAudioCommon import EntityManager, debug_print, set_debug, set_debug_prefix

from ..lib.dag import DAGNode, EdgeTransform
from ..lib.base import ArrayBackend, ArrayHandle
from ..lib.blosc2_backend import Blosc2Backend
from ..lib.zarr_backend import ZarrBackend
from ..lib.graph_builder import GraphBuilder

@dataclass
class MaterializedNode:
    """Holds the DAG node and its corresponding handle in the backend."""
    node: DAGNode
    handle: ArrayHandle

    @property
    def shape(self) -> Tuple[int, ...]:
        """Convenience accessor for the node's shape."""
        return self.node.shape

    @property
    def dtype(self) -> np.dtype:
        """Convenience accessor for the node's dtype."""
        return self.node.dtype


@dataclass
class StorageEngine:
    """
    Materializes a DAG of DAGNodes into a chosen backend and exposes
    non-destructive, chunked processing over the graph.
    """
    entity_manager: EntityManager
    backend: ArrayBackend | None = None
    _mats: Dict[str, MaterializedNode] = field(default_factory=dict, init=False)

    def __post_init__(self):
        config = self.entity_manager.get('config')
        set_debug(config.system.debug)
        set_debug_prefix(self.__class__.__name__)

        storage_config = config.storage
        self.root_path = storage_config.root_path
        self.chunk_size = storage_config.chunk_size_samples

        if self.backend is None:
            if storage_config.backend == "blosc2":
                cparams = {
                    "codec": eval(f"blosc2.Codec.{storage_config.blosc2_codec.upper()}"),
                    "clevel": storage_config.blosc2_clevel,
                    "nthreads": config.storage.blosc2_cparams_threads,
                    "filters": [eval(f"blosc2.Filter.{f.upper()}") for f in storage_config.blosc2_filters],
                }
                self.backend = Blosc2Backend(
                    root_path=self.root_path,
                    cparams=cparams,
                    dparams_nthreads=config.storage.blosc2_dparams_threads,
                    chunk_size_samples=storage_config.chunk_size_samples,
                )
            elif storage_config.backend == "zarr":
                self.backend = ZarrBackend(root_path=self.root_path, **storage_config.zarr_store_kwargs)
            else:
                raise ValueError(f"Unsupported storage backend: {storage_config.backend}")

        debug_print(f"StorageEngine initialized with '{storage_config.backend}' backend at '{self.root_path}'")

    def materialize(self, root: DAGNode, resume: bool = True) -> None:
        """Creates the persistent arrays for the entire DAG."""
        debug_print(f"Materializing DAG from root: {root.name}")
        for node in self._toposort(root):
            if node.shape[-1] <= 0:
                debug_print(f"  Skipping node '{node.name}' with invalid shape {node.shape}")
                continue

            key = self._sanitize(node.name)
            handle = None
            if resume:
                handle = self.backend.try_open(key, node.shape)

            if handle is None:
                debug_print(f"  Creating new array for node: {node.name} (shape: {node.shape})")
                handle = self.backend.create(name=key, shape=node.shape, dtype=node.dtype)

            if handle is None:
                raise RuntimeError(f"Backend {type(self.backend).__name__} returned None for node '{node.name}' (shape={node.shape}, dtype={node.dtype})")

            self._mats[node.name] = MaterializedNode(node, handle)

    def write_node(self, name: str, data: np.ndarray, slices: Union[slice, tuple] = ...) -> None:
        """Writes data to a materialized node."""
        if name not in self._mats:
            raise KeyError(f"Node '{name}' has not been materialized.")
        
        mat = self._mats[name]
        handle = mat.handle
        
        # Ensure data is a numpy array of the correct dtype and contiguous
        data = np.asarray(data)
        if data.dtype != mat.dtype:
            data = data.astype(mat.dtype)
        data = np.ascontiguousarray(data)
        
        # Normalize slices: convert scalar indices to single-element slices
        # because blosc2 may not handle mixed scalar/slice indexing correctly
        normalized_slices = self._normalize_slices(slices, mat.shape)
        
        # Compute the expected shape of the target region
        expected_shape = self._compute_slice_shape(normalized_slices, mat.shape)
        
        # Reshape or broadcast data to match expected shape
        if data.shape != expected_shape:
            if data.size == int(np.prod(expected_shape)):
                data = data.reshape(expected_shape)
            else:
                try:
                    data = np.broadcast_to(data, expected_shape).copy()
                except ValueError:
                    raise ValueError(
                        f"Data shape {data.shape} does not match slice shape {expected_shape} "
                        f"for node '{name}' (node shape: {mat.shape})"
                    )
        
        try:
            handle.write(data, normalized_slices)
        except Exception as e:
            debug_print(
                f"Failed to write to node '{name}' with slices {normalized_slices}: {e}\n"
                f"  Data shape: {data.shape}, dtype: {data.dtype}\n"
                f"  Node shape: {mat.shape}, dtype: {mat.dtype}"
            )
            raise

    def read_node(self, name: str, slices: Union[slice, tuple] = ...) -> np.ndarray:
        """Reads data from a materialized node."""
        if name not in self._mats:
            raise KeyError(f"Node '{name}' has not been materialized.")
        mat = self._mats[name]
        normalized_slices = self._normalize_slices(slices, mat.shape)
        return mat.handle.read(normalized_slices)

    def build_and_process(self, obj_idx: int, duration_s: float):
        """
        High-level method to build the graph for an object and process it.
        """
        builder = GraphBuilder(self.entity_manager)
        root_node = builder.build_graph(obj_idx, duration_s)
        self.materialize(root_node)
        self.process_graph()
        return root_node

    def process_graph(self) -> None:
        """
        Processes all nodes in the DAG in topological order, applying
        transforms from parents to children.
        """
        debug_print("Processing DAG...")
        for name in list(self._mats.keys()):
            self.apply_transform(name)

    def apply_transform(self, name: str) -> None:
        """
        Applies the node's EdgeTransform from its parent(s) in a chunked manner.
        """
        mat = self._mats.get(name)
        if not mat or not mat.node.parent_edges:
            return

        debug_print(f"  Applying transform for node: {name}")
        if len(mat.node.parent_edges) == 1:
            parent_node, tr = mat.node.parent_edges[0]
            parent_mat = self._mats.get(parent_node.name)
            if parent_mat is None:
                debug_print(f"    Parent node '{parent_node.name}' not materialized, skipping.")
                return
            self._apply_single_transform(parent_mat, mat, tr)
        else:
            parents = []
            for p, tr in mat.node.parent_edges:
                parent_mat = self._mats.get(p.name)
                if parent_mat is not None:
                    parents.append((parent_mat, tr))
            if parents:
                self._apply_combine_transform(parents, mat)

    def _apply_single_transform(self, parent_mat: MaterializedNode, child_mat: MaterializedNode, tr: EdgeTransform) -> None:
        """Applies a single transform in chunks along the last (sample) axis."""
        if tr.op is None:
            debug_print(f"    Transform '{tr.name}' has no op, skipping.")
            return

        n_samples = parent_mat.shape[-1]
        if n_samples <= 0:
            return

        parent_ndim = len(parent_mat.shape)
        child_ndim = len(child_mat.shape)

        for start in range(0, n_samples, self.chunk_size):
            stop = min(start + self.chunk_size, n_samples)

            parent_slices = tuple([slice(0, parent_mat.shape[i]) for i in range(parent_ndim - 1)] + [slice(start, stop)])
            child_slices = tuple([slice(0, child_mat.shape[i]) for i in range(child_ndim - 1)] + [slice(start, stop)])

            try:
                src_chunk = parent_mat.handle.read(parent_slices)
            except Exception as e:
                debug_print(f"    Failed to read parent slice {parent_slices}: {e}")
                raise
            src_chunk = np.asarray(src_chunk)

            dst_chunk = tr.op(src_chunk, **tr.params)
            if dst_chunk is None:
                continue
            dst_chunk = np.ascontiguousarray(dst_chunk, dtype=child_mat.dtype)

            expected_shape = child_mat.shape[:-1] + (stop - start,)

            if dst_chunk.shape != expected_shape:
                if dst_chunk.size == int(np.prod(expected_shape)):
                    dst_chunk = dst_chunk.reshape(expected_shape)
                else:
                    new_chunk = np.zeros(expected_shape, dtype=child_mat.dtype)
                    n_common = min(dst_chunk.ndim, len(expected_shape))
                    common_src = tuple(slice(0, min(dst_chunk.shape[i], expected_shape[i])) for i in range(n_common))
                    new_chunk[common_src] = dst_chunk[common_src]
                    dst_chunk = new_chunk

            try:
                child_mat.handle.write(dst_chunk, child_slices)
            except Exception as e:
                debug_print(f"    Failed to write child slice {child_slices}: {e}")
                raise

    def _apply_combine_transform(self, parents: List[Tuple[MaterializedNode, EdgeTransform]], child_mat: MaterializedNode) -> None:
        if not parents:
            return

        n_samples = parents[0][0].shape[-1]
        if n_samples <= 0:
            return

        child_ndim = len(child_mat.shape)

        for start in range(0, n_samples, self.chunk_size):
            stop = min(start + self.chunk_size, n_samples)

            src_chunks = []
            for p, _ in parents:
                p_ndim = len(p.shape)
                p_slices = tuple([slice(0, p.shape[i]) for i in range(p_ndim - 1)] + [slice(start, stop)])
                src_chunks.append(np.asarray(p.handle.read(p_slices)))

            dst_chunk = np.concatenate(src_chunks, axis=0)
            child_slices = tuple([slice(0, child_mat.shape[i]) for i in range(child_ndim - 1)] + [slice(start, stop)])
            child_mat.handle.write(dst_chunk, child_slices)

    def close(self) -> None:
        """Closes all open array handles."""
        for mat in self._mats.values():
            if hasattr(mat.handle, 'close'):
                mat.handle.close()
        debug_print("StorageEngine closed.")

    # --- Internals ---

    @staticmethod
    def _sanitize(name: str) -> str:
        return name.replace("/", "__").replace(" ", "_")

    @staticmethod
    def _normalize_slices(slices: Union[slice, tuple], shape: Tuple[int, ...]) -> tuple:
        """
        Convert scalar indices to single-element slices for blosc2 compatibility.
        
        blosc2's NDArray.__setitem__ can fail with mixed scalar/slice indexing,
        so we normalize all scalar indices to single-element slices.
        """
        if not isinstance(slices, tuple):
            slices = (slices,)
        
        normalized = []
        for s in slices:
            if isinstance(s, (int, np.integer)):
                # Convert scalar to single-element slice
                normalized.append(slice(int(s), int(s) + 1))
            else:
                normalized.append(s)
        
        # Pad with full slices if fewer slices than dimensions
        while len(normalized) < len(shape):
            normalized.append(slice(None))
        
        # Truncate if more slices than dimensions
        normalized = normalized[:len(shape)]
        
        return tuple(normalized)

    @staticmethod
    def _compute_slice_shape(slices: tuple, shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Compute the shape of the region selected by slices."""
        result = []
        for i, s in enumerate(slices):
            if i >= len(shape):
                break
            if isinstance(s, slice):
                start = s.start if s.start is not None else 0
                stop = s.stop if s.stop is not None else shape[i]
                if start < 0:
                    start = shape[i] + start
                if stop < 0:
                    stop = shape[i] + stop
                result.append(stop - start)
            else:
                result.append(1)
        return tuple(result)

    def _toposort(self, root: DAGNode) -> List[DAGNode]:
        seen, out = set(), []

        def visit(n: DAGNode):
            if n.name in seen:
                return
            for p, _ in n.parent_edges:
                visit(p)
            seen.add(n.name)
            out.append(n)

        visit(root)
        return out

