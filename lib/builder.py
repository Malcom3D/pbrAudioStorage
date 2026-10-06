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

from .collections import defaultdict
from .storage import AudioStore
from .dag import DagNode, SumMixer
from .sources import SignalSource, TrackSource
from .audio_model import NodeKind, NodePath, AudioProps, new_id


class DagBuilder:
    """
    Reads a B2DIR AudioStore and constructs a DAG where:
      signal -> (track) -> (tracks_group) -> (object) -> (collection) -> (scene)
    Each level is a DagNode whose children are the level below.
    """

    def __init__(self, store: AudioStore):
        self.store = store
        self._by_level: dict[NodeKind, dict[str, DagNode]] = defaultdict(dict)

    def build(self) -> DagNode:
        # 1. leaves: one DagNode per (track_id, track_index, signal_index)
        for track_id in self.store.list_tracks():
            path, props = self.store.get_attrs(track_id)
            arr = self.store.get_track(track_id)
            n_tracks, n_signals, _ = arr.shape

            # A "track" DagNode aggregates its signals
            for t_idx in range(n_tracks):
                track_node = DagNode(
                    node_id=f"track:{track_id}:{t_idx}",
                    kind=NodeKind.TRACK,
                    path=path,
                    props=props,
                    children=[
                        DagNode(
                            node_id=f"signal:{track_id}:{t_idx}:{s_idx}",
                            kind=NodeKind.SIGNAL,
                            path=path,
                            props=props,
                            source=SignalSource(self.store, track_id, t_idx, s_idx),
                        )
                        for s_idx in range(n_signals)
                    ],
                    mixer=SumMixer(),
                )
                self._by_level[NodeKind.TRACK][track_node.node_id] = track_node

                # propagate up: group -> object -> collection -> scene
                self._attach_up(track_node, path, props)

        # 2. top: scene node
        scenes = self._by_level[NodeKind.SCENE]
        if not scenes:
            raise RuntimeError("No scene could be constructed from the store.")
        # if multiple scenes exist, you'd return a list; here we return the first
        return next(iter(scenes.values()))

    # -------- helpers --------

    def _attach_up(self, child: DagNode, path: NodePath, props: AudioProps) -> None:
        """Walk up the hierarchy, creating/attaching parent nodes."""
        chain = [
            (NodeKind.TRACKS_GROUP, path.tracks_group_id),
            (NodeKind.OBJECT,       path.object_id),
            (NodeKind.COLLECTION,   path.collection_id),
            (NodeKind.SCENE,        path.scene_id),
        ]
        current = child
        for kind, ident in chain:
            if ident is None:
                continue
            key = f"{kind.value}:{ident}"
            if key not in self._by_level[kind]:
                self._by_level[kind][key] = DagNode(
                    node_id=key, kind=kind, path=path, props=props,
                                       children=[], mixer=SumMixer(),
                )
            parent = self._by_level[kind][key]
            if current not in parent.children:
                parent.children.append(current)
            current = parent
