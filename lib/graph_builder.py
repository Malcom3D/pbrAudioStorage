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
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Any, Tuple

from .dag import DAGNode, NodeKind, EdgeTransform, TrackDescriptor
from .track_registry import TrackRegistry
from pbrAudioCommon import EntityManager

@dataclass
class GraphBuilder:
    """
    Reads EntityManager config and the TrackRegistry to build the DAG for the StorageEngine.
    """
    entity_manager: EntityManager

    def __post_init__(self):
        config = self.entity_manager.get("config")
        self.sample_rate = int(config.system.sample_rate)
        self.dtype = np.dtype(np.float32) # or np.dtype(np.float64)
        frequency_bands = self.entity_manager.get("frequencies")
        self.bands = len(frequency_bands.get_bands()) if frequency_bands is not None else 1

    def build_graph(self, obj_idx: int, duration_s: float) -> DAGNode:
        """
        Builds a complete DAG for a single object by iterating over all
        registered track descriptors.
        """
        n_samples = int(self.sample_rate * duration_s)
        n_samples = 1 if n_samples  < 1 else n_samples
        all_nodes: List[DAGNode] = []

        # Iterate over every registered descriptor to build its part of the graph
        for descriptor in TrackRegistry.get_all().values():
            # Create the root (source) node for this track group
            n_tracks = len(descriptor.track_names)
            
            source_node = DAGNode(
                kind=NodeKind.SOURCE,
                name=f"obj{obj_idx}/{descriptor.name}_source",
                shape=(n_tracks, n_samples),
                dtype=self.dtype,
                meta={
                    "sample_rate": self.sample_rate,
                    "track_names": descriptor.track_names,
                    "descriptor_name": descriptor.name,
                    **descriptor.meta
                },
            )

            # Create the processed node
            processed_node = DAGNode(
                kind=descriptor.node_kind,
                name=f"obj{obj_idx}/{descriptor.name}_processed",
                shape=(n_tracks, self.bands, n_samples),
                dtype=self.dtype,
                meta={
                    "sample_rate": self.sample_rate,
                    "bands": self.bands,
                    "track_names": descriptor.track_names,
                    "descriptor_name": descriptor.name,
                    **descriptor.meta
                },
            )

            # Get the transform from the descriptor's factory and add the edge
            transform = descriptor.transform_factory(self.entity_manager, source_node)
            processed_node.add_parent(source_node, transform)
            
            all_nodes.append(processed_node)

        if not all_nodes:
            # Return an empty node if nothing is registered
            return DAGNode(kind=NodeKind.OUTPUT, name=f"obj{obj_idx}/empty", shape=(0,0), dtype=self.dtype)

        # For now, we assume a single final node.
        # If multiple descriptors are registered, we'd need a merge node.
        # Let's create a merge node if there's more than one.
        if len(all_nodes) == 1:
            return all_nodes[0]
        else:
            # This is a simplified merge. A real implementation might
            # concatenate along the track axis.
            total_tracks = sum(n.shape[0] for n in all_nodes)
            merged_node = DAGNode(
                kind=NodeKind.OUTPUT,
                name=f"obj{obj_idx}/merged_output",
                shape=(total_tracks, self.bands, n_samples),
                dtype=self.dtype,
                meta={"sample_rate": self.sample_rate, "bands": self.bands},
            )
            # The merge logic would be in the StorageEngine's _apply_combine_transform
            # For now, we just add parents.
            for node in all_nodes:
                # The transform here is a placeholder.
                merged_node.add_parent(node, EdgeTransform(name="merge_concat"))
            return merged_node
