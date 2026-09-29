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
                    "codec": getattr(blosc2, f"Codec.{storage_config.blosc2_codec.upper()}", blosc2.Codec.ZSTD),
                    "clevel": storage_config.blosc2_clevel,
                    "filters": [getattr(blosc2, f"Filter.{f.upper()}") for f in storage_config.blosc2_filters],
                }
                self.backend = Blosc2Backend(root_path=self.root_path, cparams=cparams)
            elif storage_config.backend == "zarr":
                self.backend = ZarrBackend(root_path=self.root_path, **storage_config.zarr_store_kwargs)
            else:
                raise ValueError(f"Unsupported storage backend: {storage_config.backend}")
        
        debug_print(f"StorageEngine initialized with '{storage_config.backend}' backend at '{self.root_path}'")

    def materialize(self, root: DAGNode, resume: bool = True) -> None:
        """Creates the persistent arrays for the entire DAG."""
        debug_print(f"Materializing DAG from root: {root.name}")
        for node in self._toposort(root):
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
        """
        debug_print("Processing DAG...")
        # Get the root node from the materialized nodes (a bit of a hack)
        if not self._mats:
            return
        root_node = next(iter(self._mats.values())).node
        for node_name in [n.name for n in self._toposort(root_node)]:
            if node_name in self._mats:
                self.apply_transform(node_name)

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
            parent_mat = self._mats[parent_node.name.name]
            self._apply_single_transform(parent_mat, mat, tr)
        else:
            parents = [(self._mats[p.name], tr) for p, tr in mat.node.parent_edges]
            self._apply_combine_transform(parents, mat)

    def _apply_single_transform(self, parent_mat: MaterializedNode, child_mat: MaterializedNode, tr: EdgeTransform) -> None:
        """AppApplies a single transform in chunks."""
        n_samples = parent_mat.shape[-1]
        for start in range(0, n_samples, self.chunk_size):
            stop = min(start + self.chunk_size, n_samples)
            src_chunk = parent_mat.handle.read((..., slice(start, stop)))
            dst_chunk = tr.op(src_chunk, **tr.params)
            child_mat.handle.write(dst_chunk, (..., slice(start, stop)))

    def _apply_combine_transform(self, parents: List[Tuple[MaterializedNode, EdgeTransform]], child_mat: MaterializedNode) -> None:
        """Applies a combine transform (e.g., merge) in chunks."""
        # This is a placeholder for more complex merge logic.
        # For now, it assumes a simple concatenation along the first axis.
        n_samples = parents[0][0].shape[-1]
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

