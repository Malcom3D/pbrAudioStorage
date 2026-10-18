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
from typing import Any, Dict, List, Optional

from .audio_data_type import (
    AudioDataType,
    DataTypeKind,
    DefaultBehavior,
)
from ..utils.errors import SchemaError


_VALID_KINDS = {k.value for k in DataTypeKind}
_VALID_BEHAVIORS = {b.value for b in DefaultBehavior}


@dataclass
class AudioSchema:
    """
    A registered audio data schema as produced by an engine.

    The schema is one AudioDataType per (track, signal) template for
    an object tpe. StorageEngine.materialize() expands it per object.
    """
    collection: str
    engine: str
    objs_type: str                          # config attribute name, e.g. "objects"
    track_group: str = "default"
    track_names: List[str] = field(default_factory=list)
    signal_type: str = "float32"            # "float32" | "float64"
    signal_names: List[str] = field(default_factory=list)
    total_samples: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    # per-track behaviour overrides (falls back to per-kind defaults)
    track_behaviors: Dict[str, str] = field(default_factory=dict)
    # per-track data-kind overrides
    track_kinds: Dict[str, str] = field(default_factory=dict)
    # where to load the per-item force side-channel from, if any
    # e.g. {"impact": "coupling_strength", "sliding": "coupling_strength"}
    track_force_source: Dict[str, str] = field(default_factory=dict)

    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "collection": self.collection,
            "engine": self.engine,
            "objs_type": self.objs_type,
            "track_group": self.track_group,
            "track_names": list(self.track_names),
            "signal_type": self.signal_type,
            "signal_names": list(self.signal_names),
            "total_samples": int(self.total_samples),
            "metadata": dict(self.metadata),
            "track_behaviors": dict(self.track_behaviors),
            "track_kinds": dict(self.track_kinds),
            "track_force_source": dict(self.track_force_source),
        }


def validate_schema(schema: AudioSchema) -> None:
    if not schema.collection:
        raise SchemaError("schema.collection must be a non-empty string")
    if not schema.engine:
        raise SchemaError("schema.engine must be a non-empty string")
    if not schema.objs_type:
        raise SchemaError("schema.objs_type must be a non-empty string (e.g. 'objects')")
    if not schema.track_names:
        raise SchemaError("schema.track_names must contain at least one track name")
    if schema.signal_type not in ("float32", "float64"):
        raise SchemaError(f"schema.signal_type must be 'float32' or 'float64', got {schema.signal_type!r}")
    if schema.total_samples < 0:
        raise SchemaError("schema.total_samples must be >= 0")

    for kind in schema.track_kinds.values():
        if kind not in _VALID_KINDS:
            raise SchemaError(f"invalid track kind {kind!r}; valid: {sorted(_VALID_KINDS)}")

    for beh in schema.track_behaviors.values():
        if beh not in _VALID_BEHAVIORS:
            raise SchemaError(f"invalid track behavior {beh!r}; valid: {sorted(_VALID_BEHAVIORS)}")


def make_object_node(name: str, idx: int, signal_type: str, total_samples: int) -> AudioDataType:
    """
    Build the top-level AudioDataType for one object's NDArray.

    The object node has default_behavior='multi' so its tracks can stack
    into an engine-level read.
    """
    return AudioDataType(
        name=name,
        id=int(idx),
        data_type=DataTypeKind.MULTI_TRACK.value,
        default_behavior=DefaultBehavior.MULTI.value,
        sample_rate=48000,      # overridden at materialize() time from config
        bit_depth=32,
        subtype="FLOAT",
        channels=1,
        metadata={
            "signal_type": signal_type,
            "total_samples": int(total_samples),
        },
    )


def make_track_node(track_name: str, track_idx: int) -> AudioDataType:
    return AudioDataType(
        name=track_name,
        id=int(track_idx),
        data_type=DataTypeKind.CUSTOM.value,     # filled by engine later
        default_behavior=DefaultBehavior.MERGE.value,
        channels=1,
    )


def make_signal_node(signal_name: str, signal_idx: int, total_samples: int) -> AudioDataType:
    return AudioDataType(
        name=signal_name,
        id=int(signal_idx),
        data_type=DataTypeKind.MONO.value,
        default_behavior=DefaultBehavior.SINGLE.value,
        channels=1,
        metadata={"total_samples": int(total_samples)},
    )
