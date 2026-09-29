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

__version__ = "0.0.8"
__author__ = "Malcom3D"
__description__ = "High performance per-object/per-phase Multitrack RAW Audio Data Storage solution"

import os, sys
import numpy as np

decimals = 18
np.set_printoptions(precision=decimals, floatmode='fixed', threshold=np.inf)

from .core.storage_engine import StorageEngine
from .lib.dag import DAGNode, NodeKind, EdgeTransform, TrackDescriptor
from .lib.graph_builder import GraphBuilder
from .lib.track_registry import TrackRegistry
from .lib.base import ArrayBackend, ArrayHandle
from .lib.blosc2_backend import Blosc2Backend
from .lib.zarr_backend import ZarrBackend

__all__ = [
    'StorageEngine',
    'TrackDescriptor',
    'TrackRegistry',
    'DAGNode',
    'NodeKind',
    'EdgeTransform',
    'GraphBuilder',
    'ArrayBackend',
    'ArrayHandle',
    'Blosc2Backend',
    'ZarrBackend',
]

