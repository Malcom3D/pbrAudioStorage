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

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import uuid


class NodeKind(str, Enum):
    SIGNAL = "signal"
    TRACK = "track"
    TRACKS_GROUP = "tracks_group"
    OBJECT = "object"
    COLLECTION = "collection"
    SCENE = "scene"


def new_id() -> str:
    return uuid.uuid4().hex


@dataclass(frozen=True)
class NodePath:
    """Logical path through the hierarchy. Any level may be None."""
    scene_id: Optional[str] = None
    collection_id: Optional[str] = None
    object_id: Optional[str] = None
    tracks_group_id: Optional[str] = None
    track_id: Optional[str] = None
    signal_id: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "scene_id": self.scene_id,
            "collection_id": self.collection_id,
            "object_id": self.object_id,
            "tracks_group_id": self.tracks_group_id,
            "track_id": self.track_id,
            "signal_id": self.signal_id,
        }


@dataclass
class AudioProps:
    """Audio metadata stored in blosc2 attrs."""
    bit_depth: int = 32
    sample_rate: int = 48000
    layout: str = "mono"          # 'mono', 'stereo', 'ambisonic', ...
    n_channels: int = 1
    custom: dict = field(default_factory=dict)

    def to_attrs(self) -> dict:
        return {
            "bit_depth": self.bit_depth,
            "sample_rate": self.sample_rate,
            "layout": self.layout,
            "n_channels": self.n_channels,
            "custom": self.custom,
        }

    @classmethod
    def from_attrs(cls, a: dict) -> "AudioProps":
        return cls(
            bit_depth=a.get("bit_depth", 32),
            sample_rate=a.get("sample_rate", 48000),
            layout=a.get("layout", "mono"),
            n_channels=a.get("n_channels", 1),
            custom=a.get("custom", {}),
        )

