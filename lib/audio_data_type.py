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

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional


class DataTypeKind(str, Enum):
    """High-level kind of the audio data node."""
    MONO = "mono"
    SURROUND = "surround"
    MULTI_TRACK = "multi-track"
    AMBISONIC = "ambisonic"
    CUSTOM = "custom"


class DefaultBehavior(str, Enum):
    """
    Behavior for aggregating children into a parent.

    - NONE   : do not contribute to the parent node
    - SINGLE : take the first child, drop the rest  (shape (1, N))
    - MERGE  : sum children element-wise (shape (1, N))
    - MULTI  : stack children along axis 0 (shape (C, N))
    """
    NONE = "none"
    SINGLE = "single"
    MERGE = "merge"
    MULTI = "multi"


@dataclass
class AudioDataType:
    """
    Metadata (json'ish') describing one node in the graph.

    Every collection, engine, object, track and signal is represented by
    one AudioDataType instance stored in blosc2 attrs as a plain dict.

    No audio data is *kept* in this class; it is metadata only.
    """

    # ---- identity ----
    name: str
    id: int = 0

    # ---- data-shape semantics ----
    data_type: str = DataTypeKind.CUSTOM.value
    default_behavior: str = DefaultBehavior.NONE.value

    # ---- audio metadata (subset of soundfile / surround / ambisonic) ----
    sample_rate: int = 48000
    bit_depth: int = 32
    subtype: str = "FLOAT"
    channels: int = 1
    surround_type: Optional[str] = None       # "5.1", "7.1", "22.2"
    ambisonic_order: Optional[int] = None     # 0, 1, 2, 3
    layout: Optional[List[List[float]]] = None  # [[az_deg, el_deg], ...]

    # ---- graph edges ----
    child_indices: List[int] = field(default_factory=list)
    child_names: List[str] = field(default_factory=list)

    # ---- non-destructive processing ----
    chains: Optional[Dict[str, Any]] = None
    # e.g. {"op":"sequential","processors":[{"type":"dc_blocker","params":{...}}, ...]}

    # ---- free-form ----
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AudioDataType":
        known = {
            "name", "id", "data_type", "default_behavior",
            "sample_rate", "bit_depth", "subtype", "channels",
            "surround_type", "ambisonic_order", "layout",
            "child_indices", "child_names", "chains", "metadata",
        }
        clean = {k: v for k, v in d.items() if k in known}
        # default the enum-like strings
        clean.setdefault("data_type", DataTypeKind.CUSTOM.value)
        clean.setdefault("default_behavior", DefaultBehavior.NONE.value)
        return cls(**clean)

    # ------------------------------------------------------------------
    def add_child(self, idx: int, name: str) -> None:
        self.child_indices.append(int(idx))
        self.child_names.append(str(name))

    def has_chain(self) -> bool:
        return bool(self.chains)

