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

__version__ = "0.2.0"
__author__ = "Malcom3D"
__description__ = "High performance near-infinite multitrack raw audio storage with non-destructive signal processing on a DAG."

import os
import numpy as np

decimals = 18
np.set_printoptions(precision=decimals, floatmode='fixed', threshold=np.inf)

from .core.storage_engine import StorageEngine
from .lib.audio_data_type import AudioDataType, DataTypeKind, DefaultBehavior
from .lib.schema import AudioSchema
from .lib.blosc2_backend import Blosc2Backend
from .lib.zarr_backend import ZarrBackend
from .dag.dag_node import DagNode
from .dag.dag_builder import DagBuilder
from .dag.dag_evaluator import DagEvaluator
from .utils.errors import StorageError, SchemaError, BackendError, DagError, PatchError

__all__ = [
    "AudioDataType",
    "DataTypeKind",
    "DefaultBehavior",
    "AudioSchema",
    "validate_schema",
    "StorageEngine",
    "Blosc2Backend",
    "ZarrBackend",
    "DagNodeNode",
    "DagBuilder",
    "DagEvaluator",
    "StorageError",
    "SchemaError",
    "BackendError",
    "DagError",
    "PatchError",
]

