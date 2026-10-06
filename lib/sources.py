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

import numpy as np
import blosc2
from storage import AudioStore
from dag import LeafSource
from audio_model import AudioProps

class SignalSource(LeafSource):
    """
    Fetches a single signal from a track NDArray stored in the AudioStore.

    NDArray shape: (n_tracks, n_signals, total_samples)
    """

    def __init__(
        self,
        store: AudioStore,
        track_id: str,
        track_index: int,
        signal_index: int,
        *,
        start: int = 0,
        stop: int | None = None,
    ):
        self.store = store
        self.track_id = track_id
        self.track_index = track_index
        self.signal_index = signal_index
        self.start = start
        self.stop = stop

    def fetch(self) -> np.ndarray:
        arr = self.store.get_track(self.track_id)
        # slice out one (1, total_samples) row and squeeze
        sl = arr[self.track_index, self.signal_index, self.start:self.stop]
        # ensure 2D: (1, total_samples)
        return np.asarray(sl, dtype=np.float32).reshape(1, -1)


class TrackSource(LeafSource):
    """
    Fetches a whole track (all signals summed or stacked) from storage.
    Here we return shape (n_signals, total_samples) for the mixer to handle.
    """

    def __init__(self, store: AudioStore, track_id: str, track_index: int):
        self.store = store
        self.track_id = track_id
        self.track_index = track_index

    def fetch(self) -> np.ndarray:
        arr = self.store.get_track(self.track_id)
        return np.asarray(arr[self.track_index], dtype=np.float32)
