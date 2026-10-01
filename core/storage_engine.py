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
from typing import Iterable, Dict, List, Tuple
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

            self._mats[node.name] = MaterializedNode(node, handle)

    def write_node(self, name: str, data: np.ndarray, slices: slice | tuple = ...) -> None:
        """Writes data to a materialized node."""
        if name not in self._mats:
            raise KeyError(f"Node '{name}' has not been materialized.")
        self._mats[name].handle.write(data, slices)

    def read_node(self, name: str, slices: slice | tuple = ...) -> np.ndarray:
        """Reads data from a materialized node."""
        if name not in self._mats:
            raise KeyError(f"Node '{name}' has not been materialized.")
        return self._mats[name].handle.read(slices)

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

        Since `_mats` is populated in topological order by `materialize()`,
        we can simply iterate over it in insertion order.
        """
        debug_print("Processing DAG...")
        # Iterate over a snapshot of keys to avoid mutation issues.
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
            # BUG FIX: parent_node is a DAGNode, so its name is a string.
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
        """Applies a single transform in chunks."""
        if tr.op is None:
            debug_print(f"    Transform '{tr.name}' has no op, skipping.")
            return

        # BUG FIX: use parent_mat.node.shape instead of parent_mat.shape
        # (we also added a `shape` property on MaterializedNode for convenience).
        n_samples = parent_mat.shape[-1]
        if n_samples <= 0:
            return

        for start in range(0, n_samples, self.chunk_size):
            stop = min(start + self.chunk_size, n_samples)
            src_chunk = parent_mat.handle.read((..., slice(start, stop)))
            dst_chunk = tr.op(src_chunk, **tr.params)
            child_mat.handle.write(dst_chunk, (..., slice(start, stop)))

    def _apply_combine_transform(self, parents: List[Tuple[MaterializedNode, Edge EdgeTransform]], child_mat: MaterializedNode) -> None:
        """Applies a combine transform (e.g., merge) in chunks."""
        # This is a placeholder for more complex merge logic.
        # For now, it assumes a simple concatenation along the first axis.
        if not parents:
            return

        # BUG FIX: use .shape property instead of .shape on MaterializedNode directly.
        n_samples = parents[0][0].shape[-1]
        if n_samples <= 0:
            return

        for start in range(0, n_samples, self.chunk_size):
            stop = min(start + self.chunk_size, n_samples)
            src_chunks = [p.handle.read((..., slice(start, stop))) for p, _ in parents]
            # In a real implementation, the merge logic would be more complex.
            # Here we just concatenate.
            dst_chunk = np.concatenate(src_chunks, axis=0)
            child_mat.handle.write(dst_chunk, (..., slice(start, stop)))

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

