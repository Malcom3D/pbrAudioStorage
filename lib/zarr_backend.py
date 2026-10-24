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

import os
import weakref
import numpy as np
import zarr
import zarrs
from collections import deque
from typing import List, Tuple, Any, Dict, Optional
from dataclasses import dataclass, field

from ..lib.audio_data_type import AudioDataType
from ..lib.schema import AudioSchema
from ..utils.errors import BackendError

zarr.config.set({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"})

@dataclass
class ZarrBackend:
    """
    Stub for zarr backend with zarrs to speed up i/o.
    """
    urlpath: str

    def __post_init__(self):
        self.urlpath = os.path.abspath(urlpath)
        pass

    def open(self, path: str = None, cparams: Any = None, dparams: Any = None, mode: str = 'a'):
        """
        Default stub method 
        """
        pass

    def register(self, schema: AudioSchema) -> None:
        """
        Default stub method
        """
        pass

    def materialize(self, collection: str, engine: str, obj_idx: int, obj_name: str) -> None:
        """
        Default stub method
        """
        pass

    def _drain(self):
        """
        Default stub method
        """
        pass

    def write(self, collection: str, engine: str, obj_idx: int, track_name: str, audio_data: np.ndarray, sample_start: int = 0, sample_stop: int = None, signal_name: Optional[str] = None, signal_metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Default stub method
        """
        pass

    def _write(self, collection: str, engine: str, obj_idx: int, track_name: str, audio_data: np.ndarray, sample_start: int = 0, sample_stop: int = None, signal_name: Optional[str] = None, signal_metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Default stub method
        """
        pass

    def read(self, collection: str, engine: str, obj_idx: int, track_name: str, signal_idx: int) -> np.ndarray:
        """
        Default stub method
        """
        pass

    def get_child(self, paths: List[str] = ['/']) -> List[str]:
        """
        Default stub method
        """
        pass

    def list(self, path: str = '/') -> List[str]:
        """
        Default stub method
        """
        pass

    def get_meta(self, path: str = '/') -> AudioDataType:
        """
        Default stub method
        """
        pass

    def get_schema(self, collection: str, engine: str) -> AudioSchema:
        """
        Default stub method
        """
        pass
