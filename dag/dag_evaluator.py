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

import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .dag_builder import DagBuilder
from .dag_node import DagNode
from ..utils.errors import DagError

@dataclass
class Props:
    """
    Free-form context handed to processors during evaluation.

    ``custom`` is the injection channel: e.g. ``{"force": np.ndarray}``.
    Processors in pbrAudioSignals already know how to read ``props.custom``.
    """
    custom: Dict[str, Any] = field(default_factory=dict)


class DagEvaluator:
    """
    Walks a compiled DagBuilder, evaluating each processor exactly once
    and sharing intermediate results across branches.
    """

    def __init__(self, builder: DagBuilder, props: Optional[Props] = None) -> None:
        self.builder = builder
        self.props = props if props is not None else Props()

    # ------------------------------------------------------------------
    def evaluate(self, root: int) -> np.ndarray:
        """
        Evaluate the sub-DAG rooted at ``root`` and return the ndarray result.
        Sources are returned as-is; processors are applied in topological order.
        """
        order = self.builder.compile(roots=[root])
        cache: Dict[int, np.ndarray] = {}

        for nid in order:
            node: DagNode = self.builder.node(nid)
            if node.is_source:
                cache[nid] = node.audio
                continue

            # gather inputs
            if not node.inputs:
                raise DagError(f"processor node {nid} ({node.name}) has no inputs")
            parts = [cache[i] for i in node.inputs]

            # single input: 1-to-1
            if len(parts) == 1:
                out = self._apply(node, parts[0])
            else:
                # multi-input: merge on axis 0 first, then process
                arrays = [self._as_2d(p) for p in parts]
                merged = self._stack_or_merge(arrays, node.metadata.get("merge_mode", "concat"))
                out = self._apply(node, merged)

            cache[nid] = out

        return cache[root]

    # ------------------------------------------------------------------
    def evaluate_many(self, roots: Sequence[int]) -> Dict[int, np.ndarray]:
        results: Dict[int, np.ndarray] = {}
        for r in roots:
            results[r] = self.evaluate(r)
        return results

    # ------------------------------------------------------------------
    def _apply(self, node: DagNode, x: np.ndarray) -> np.ndarray:
        proc = node.processor
        if proc is None:
            return x
        # BaseProcessor: call .process(x, props)
        if hasattr(proc, "process"):
            return proc.process(x, self.props)
        return proc(x, self.props)

    # ------------------------------------------------------------------
    @staticmethod
    def _as_2d(x: np.ndarray) -> np.ndarray:
        if x is None:
            return None
        a = np.ascontiguousarray(x, dtype=np.float32)
        if a.ndim == 1:
            a = a.reshape(1, -1)
        return a

    @staticmethod
    def _stack_or_merge(arrays: List[np.ndarray], mode: str) -> np.ndarray:
        if mode == "concat":
            return np.concatenate(arrays, axis=0)
        if mode == "merge":
            # all must have the same shape
            base = arrays[0]
            out = np.array(base, dtype=np.float32, copy=True)
            for a in arrays[1:]:
                if a.shape != base.shape:
                    raise DagError("merge requires equal-shape inputs")
                out += a
            return out
        raise DagError(f"unknown merge_mode {mode!r}")

