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

# dag.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Sequence, Optional, Callable
import numpy as np
from audio_model import NodeKind, NodePath, AudioProps, new_id


# ---------- Signal processor protocol ----------

class SignalProcessor:
    """
    Stub. Implementations transform an array of shape
    (n_channels, total_samples) -> same shape.
    """
    def process(self, x: np.ndarray, props: AudioProps) -> np.ndarray:
        raise NotImplementedError


class IdentityProcessor(SignalProcessor):
    def process(self, x, props):
        return x


# ---------- DAG nodes ----------

@dataclass
class DagNode:
    node_id: str
    kind: NodeKind
    path: NodePath
    props: AudioProps
    children: list["DagNode"] = field(default_factory=list)
    processor: SignalProcessor = field(default_factory=IdentityProcessor)
    # for leaf nodes: how to fetch the raw data
    source: Optional["LeafSource"] = None
    # for op nodes: how to combine children
    mixer: Optional["Mixer"] = None


class LeafSource:
    """Abstract fetch of a leaf's audio from disk."""
    def fetch(self) -> np.ndarray:  # (n_channels, total_samples)
        raise NotImplementedError


class Mixer:
    """Abstract combination of children arrays."""
    def mix(self, arrays: Sequence[np.ndarray], props: AudioProps) -> np.ndarray:
        raise NotImplementedError


class SumMixer(Mixer):
    """Trivial sum. Replace with ambisonic-aware / decorrelated mixing."""
    def mix(self, arrays, props):
        if not arrays:
            return np.zeros((props.n_channels, 0), dtype=np.float32)
        n = min(a.shape[-1] for a in arrays)
        out = np.zeros((props.n_channels, n), dtype=np.float32)
        for a in arrays:
            out += a[..., :n]
        return out
