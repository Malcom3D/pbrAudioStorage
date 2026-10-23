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
from typing import Any, Callable, Dict, List, Optional

@dataclass
class DagNode:
    """
    A node in the processing DAG.

    Either:
      * a *source* (``is_source=True``) — has ``audio`` (numpy) and no inputs, or
      * a *processor* — has ``processor`` (a callable ``process(x, props)``)
        and a list of ``inputs`` (child node ids).

    ``props`` is a free-form dict handed to each processor as its ``Props``
    object (see DagEvaluator and Blosc2Backend for how it is built).
    """
    node_id: int
    name: str
    is_source: bool = False

    # source
    audio: Optional[Any] = None  # np.ndarray

    # processor
    processor: Optional[Callable] = None  # (np.ndarray, Props) -> np.ndarray
    inputs: List[int] = field(default_factory=list)

    # metadata carried for the DAG walker
    metadata: Dict[str, Any] = field(default_factory=dict)

    # cached output from the last evaluation
    _cache: Optional[Any] = None
    _cache_valid: bool = False

    # ------------------------------------------------------------------
    def invalidate(self) -> None:
        self._cache = None
        self._cache_valid = False

