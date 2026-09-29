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
from dataclasses import dataclass, field
from typing import Iterable
import numpy as np

from ..lib.dag import DAGNode, NodeKind, EdgeTransform
from ..lib.base import ArrayBackend, ArrayHandle
from ..lib.blosc2_backend import Blosc2Backend

@dataclass
class MaterializedNode:
    node: DAGNode
    handle: ArrayHandle
    children: list["MaterializedNode"] = field(default_factory=list)


@dataclass
class StorageEngine:
    """
    Materializes a DAG of DAGNodes into a chosen backend and exposes
    non-destructive, JIT-accelerated processing over the graph.
    """
    root_path: str,
    backend: ArrayBackend | None = None,
    default_dtype=np.float32,
    _mats: dict[str, MaterializedNode] = {}

    def materialize(self, root: DAGNode, resume=True):
        for node in self._toposort(root):
            key = self._sanitize(node.name)
            if resume and (h := self.backend.try_open(key, node.shape)):
                self._mats[node.name] = MaterializedNode(node, h)
                continue
            handle = self.backend.create(name=key, shape=node.shape, dtype=node.dtype, cparams=node.meta.get("cparams"))
        self._mats[node.name] = MaterializedNode(node, handle)

    def write_node(self, name: str, data: np.ndarray, slices=...) -> None:
        self._mats[name].handle.write(data, slices)

    def read_node(self, name: str, slices=...) -> np.ndarray:
        return self._mats[name].handle.read(slices)

    def apply_transform(self, name: str) -> None:
        """
        Apply the node's EdgeTransform from its parent(s).
        Non-destructive: writes to child, parent untouched.
        """
        mat = self._mats[name]
        node = mat.node
        if not node.parent_edges:
            return
        if len(node.parent_edges) == 1:
            parent, tr = node.parent_edges[0]
            self._apply_single(self._mats[parent.name], mat, tr)
        else:
            self._apply_combine([(self._mats[p.name], tr) for p, tr in node.parent_edges], mat)

#        if self._is_jit_expressible(tr):
#            mat.handle.apply_jit(tr.name, expr=tr.params["expr"], **tr.params)
#        else:
#            self._chunked_apply(parent_mat, mat, tr)

    def pipeline(self, names: Iterable[str]) -> None:
        """Apply transforms in the order given (topological)."""
        for n in names:
            self.apply_transform(n)

    def close(self) -> None:
        for m in self._mats.values():
            arr = getattr(m.handle, "_arr", None)
            if arr is not None and hasattr(arr, "close"):
                arr.close()

    # ---------- internals ----------

    @staticmethod
    def _sanitize(name: str) -> str:
        return name.replace("/", "__").replace(" ", "_")

    def _toposort(self, root: DAGNode) -> list[DAGNode]:
        seen, out = set(), []

        def visit(n: DAGNode):
            if n.name in seen:
                return
            for p in n.parents:
                visit(p)
            seen.add(n.name)
            out.append(n)

        visit(root)
        return out

    def _is_jit_expressible(self, tr: EdgeTransform) -> bool:
        if tr is not None and tr.params is not None:
            return "expr" in tr.params

    def _chunked_apply(self, parent_mat: MaterializedNode, child_mat: MaterializedNode, tr: EdgeTransform) -> None:
        """Chunk along the time axis (last dim) to keep memory bounded."""
        chunk = 1 << 16  # 65536 samples
        n = parent_mat.handle.shape[-1]
        for start in range(0, n, chunk):
            stop = min(start + chunk, n)
            src = parent_mat.handle.read((..., slice(start, stop)))
            dst = tr.op(src, **tr.params)
            child_mat.handle.write(dst, (..., slice(start, stop)))
