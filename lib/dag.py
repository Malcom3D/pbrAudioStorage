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
from typing import Any, Callable, Optional
from enum import Enum
import numpy as np


class NodeKind(str, Enum):
    SOURCE        = "source"          # raw physics / rigidbody emitter
    FORCE         = "force"           # collision, rolling, sliding, ...
    NOISE_ENH     = "noise_enhance"   # procedural noise overlay
    MODAL         = "modal"           # modal displacement
    MODAL_DIFFUSE = "modal_diffuse"   # 3D modal diffusion
    BAND          = "band"            # N-band filterbank split
    AMBISONIC     = "ambisonic"       # acoustic render output
    MIC           = "microphone"


@dataclass
class EdgeTransform:
    """Metadata describing how parent → child is computed."""
    name: str
    op: Optional[Callable] = None # JIT-compilable function signature: (chunk: np.ndarray, **params) -> np.ndarray
    params: dict[str, Any] = field(default_factory=dict) # e.g. {"unit_in": "N", "unit_out": "Pa", "gain": 1e-3}


@dataclass
class DAGNode:
    kind: NodeKind
    name: str
    shape: tuple[int, ...]          # full ND shape at this node
    dtype: np.dtype
    parent_edges: list[tuple["DAGNode", "EdgeTransform"]] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def add_parent(self, parent: "DAGNode", transform: EdgeTransform):
        self.parent_edges.append((parent, transform))
        return self
