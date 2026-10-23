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
from typing import Any, Dict, List, Optional

from .audio_data_type import AudioDataType, DataTypeKind, DefaultBehavior
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
    sample_rate: int = 44100
    bit_depth: int = 32
    subtype: str = "FLOAT"
    metadata: Dict[str, Any] = field(default_factory=dict)
    # default behaviour overrides (falls back to per-kind defaults)
    default_behaviors: Dict[str, DefaultBehavior] = field(default_factory=dict)
    # default data-kind overrides
    default_kinds: Dict[str, DataTypeKind] = field(default_factory=dict)

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
            "sample_rate": int(self.sample_rate),
            "bit_depth": int(self.bit_depth),
            "subtype": self.subtype,
            "metadata": dict(self.metadata),
            "default_behaviors": dict(self.default_behaviors),
            "default_kinds": dict(self.default_kinds),
        }

    def validate(self) -> None:
        if not self.collection:
            raise SchemaError("schema.collection must be a non-empty string")
        if not self.engine:
            raise SchemaError("schema.engine must be a non-empty string")
        if not self.objs_type:
            raise SchemaError("schema.objs_type must be a non-empty string (e.g. 'objects')")
        if not self.track_names:
            raise SchemaError("schema.track_names must contain at least one track name")
        if self.signal_type not in ("float32", "float64"):
            raise SchemaError(f"schema.signal_type must be 'float32' or 'float64', got {self.signal_type!r}")
        if self.total_samples < 0:
            raise SchemaError("self.total_samples must be >= 0")

        for kind in self.default_kinds.values():
            if kind not in _VALID_KINDS:
                raise SchemaError(f"invalid track kind {kind!r}; valid: {sorted(_VALID_KINDS)}")

        for beh in self.default_behaviors.values():
            if beh not in _VALID_BEHAVIORS:
                raise SchemaError(f"invalid track behavior {beh!r}; valid: {sorted(_VALID_BEHAVIORS)}")
