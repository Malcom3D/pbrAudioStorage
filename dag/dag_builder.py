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

from typing import Any, Dict, Iterable, List, Optional

from .dag_node import DagNode
from ..utils.errors import DagError

class DagBuilder:
    """
    Minimal, deterministic DAG builder.

    Nodes are keyed by integer ``node_id``. Callers add sources and
    processors and finally call :py:meth:`compile` to validate reachability
    and topological order.
    """

    def __init__(self) -> None:
        self._nodes: Dict[int, DagNode] = {}
        self._next_id: int = 0
        self._order: Optional[List[int]] = None

    # ------------------------------------------------------------------
    def _new_id(self) -> int:
        nid = self._next_id
        self._next_id += 1
        return nid

    # ------------------------------------------------------------------
    def add_source(self, name: str, audio: Any, metadata: Optional[Dict[str, Any]] = None) -> int:
        nid = self._new_id()
        self._nodes[nid] = DagNode(
            node_id=nid, name=name, is_source=True,
            audio=audio, metadata=dict(metadata or {}),
        )
        return nid

    def add_processor(self, name: str, processor: Any,
                      inputs: Iterable[int],
                      metadata: Optional[Dict[str, Any]] = None) -> int:
        nid = self._new_id()
        self._nodes[nid] = DagNode(
            node_id=nid, name=name, is_source=False,
            processor=processor, inputs=list(inputs),
            metadata=dict(metadata or {}),
        )
        return nid

    # ------------------------------------------------------------------
    def node(self, node_id: int) -> DagNode:
        if node_id not in self._nodes:
            raise DagError(f"unknown node_id {node_id}")
        return self._nodes[node_id]

    def nodes(self):
        return self._nodes

    # ------------------------------------------------------------------
    def compile(self, roots: Optional[Iterable[int]] = None) -> List[int]:
        """
        Compute a topological order (Kahn) restricted to ancestors of
        ``roots`` (or the whole graph when ``roots`` is None).
        Returns the ordered node ids (sources first).
        """
        if roots is None:
            roots = [nid for nid, n in self._nodes.items() if not n.inputs]

        # collect reachable subgraph in reverse
        reachable = set()
        stack = list(roots)
        while stack:
            nid = stack.pop()
            if nid in reachable:
                continue
            reachable.add(nid)
            for child in self._nodes[nid].inputs:
                if child not in self._nodes:
                    raise DagError(f"node {nid} references missing input {child}")
                stack.append(child)

        # in-degree within reachable
        indeg = {nid: 0 for nid in reachable}
        for nid in reachable:
            for child in self._nodes[nid].inputs:
                indeg[nid] += 1

        # Kahn
        ready = [nid for nid, d in indeg.items() if d == 0]
        order: List[int] = []
        while ready:
            nid = ready.pop(0)
            order.append(nid)
            for parent_id, parent in self._nodes.items():
                if parent_id not in reachable:
                    continue
                if nid in parent.inputs:
                    indeg[parent_id] -= 1
                    if indeg[parent_id] == 0:
                        ready.append(parent_id)

        if len(order) != len(reachable):
            raise DagError("cycle detected in DAG")
        self._order = order
        return order

