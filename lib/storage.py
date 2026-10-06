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

import blosc2
from typing import Iterator, Optional
from audio_model import AudioProps, NodePath

class AudioStore:
    """
    Thin wrapper around blosc2.TreeStore for a single 'tracks group' (B2DIR).

    Layout:
        <root>/                     <- TreeStore root (B2DIR)
            <track_id>/             <- one NDArray per track
                data (n_tracks, n_signals, total_samples)
                attrs: path + audio props
    """

    def __init__(self, root: str, mode: str = "a"):
        self.root = root
        self.store = blosc2.TreeStore(root, mode=mode)

    # -------- write side --------

    def put_track(
        self,
        track_id: str,
        data: "blosc2.NDArray | None" = None,
        *,
        shape: tuple[int, int, int] | None = None,
        dtype="float32",
        path: NodePath | None = None,
        props: AudioProps | None = None,
        urlpath: str | None = None,
    ) -> blosc2.NDArray:
        """Store (or open) one track's NDArray in the TreeStore."""
        if data is None:
            if shape is None:
                raise ValueError("Either `data` or `shape` must be given.")
            data = blosc2.empty(shape, dtype=dtype)

        # external array: keep the actual buffer on disk next to the store
        if urlpath is None:
            urlpath = f"{self.root}/{track_id}.b2nd"

        # NOTE: TreeStore stores NDArray by reference; the caller decides
        # whether it's external (urlpath) or in-memory.
        self.store[track_id] = data
        self.store.attrs[track_id] = {
            "path": (path or NodePath()).as_dict(),
            "props": (props or AudioProps()).to_attrs(),
        }
        return data

    def set_attrs(self, track_id: str, path: NodePath, props: AudioProps) -> None:
        self.store.attrs[track_id] = {
            "path": path.as_dict(),
            "props": props.to_attrs(),
        }

    # -------- read side --------

    def get_track(self, track_id: str) -> blosc2.NDArray:
        return self.store[track_id]

    def get_attrs(self, track_id: str) -> tuple[NodePath, AudioProps]:
        a = self.store.attrs[track_id]
        return NodePath(**a["path"]), AudioProps.from_attrs(a["props"])

    def list_tracks(self) -> Iterator[str]:
        for key in self.store:
            yield key

    def find_tracks(self, **path_filters) -> list[str]:
        """Return track_ids whose stored NodePath matches all given filters."""
        out = []
        for tid in self.list_tracks():
            path, _ = self.get_attrs(tid)
            d = path.as_dict()
            if all(d[k] == v for k, v in path_filters.items()):
                out.append(tid)
        return out
