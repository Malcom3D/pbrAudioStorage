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
from typing import Any, Callable, Optional, Tuple, List
from enum import Enum
import numpy as np


class NodeKind(str, Enum):
    """
    Defines the fundamental type of a node in the DAG.
    The specific data type (e.g., 'force', 'modal') is stored in the node's meta.
    """
    SOURCE = "source"          # Raw data, e.g., from a physics or rigidbody solver
    PROCESSED = "processed"    # Data that has been transformed, e.g., noise-enhanced
    OUTPUT = "output"          # Final rendered output, e.g., ambisonic


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

    def add_parent(self, parent: "DAGNode", transform: "EdgeTransform"):
        self.parent_edges.append((parent, transform))
        return self


@dataclass
class TrackDescriptor:
    """
    A descriptor for a group of related tracks that can be processed by the StorageEngine.
    This is the primary way to register a new data type with the system.
    """
    # A unique name for this track group, e.g., "physics_forces", "rigidbody_modal"
    name: str
    
    # The kind of node this descriptor produces.
    node_kind: NodeKind
    
    # The names of the individual tracks in this group.
    # The order is important and will be used for indexing.
    track_names: List[str]
    
    # A factory function that creates the EdgeTransform for this track group.
    # It receives the entity_manager and the node being processed as arguments.
    transform_factory: Callable[['EntityManager', DAGNode], EdgeTransform]

    # Optional metadata about the descriptor.
    meta: dict[str, Any] = field(default_factory=dict)
