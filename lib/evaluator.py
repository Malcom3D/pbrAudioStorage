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
from typing import Optional
import numpy as np
from dag import DagNode, Mixer, SumMixer
from audio_model import AudioProps

class Evaluator:
    """
    Walks the DAG and returns audio for any node.
    Signal processing is delegated to node.processor (stub for now).
    Mixing is delegated to node.mixer (stub for now).
    """

    def __init__(self, cache: bool = True):
        self.cache_enabled = cache
        self._cache: dict[str, np.ndarray] = {}

    def evaluate(self, node: DagNode) -> np.ndarray:
        if self.cache_enabled and node.node_id in self._cache:
            return self._cache[node.node_id]

        if node.source is not None:
            # leaf: fetch then process
            raw = node.source.fetch()
            out = node.processor.process(raw, node.props)
        else:
            # op: evaluate children, mix, then process
            child_arrays = [self.evaluate(c) for c in node.children]
            mixer = node.mixer or SumMixer()
            mixed = mixer.mix(child_arrays, node.props)
            out = node.processor.process(mixed, node.props)

        if self.cache_enabled:
            self._cache[node.node_id] = out
        return out

    def clear_cache(self) -> None:
               self._cache.clear()
