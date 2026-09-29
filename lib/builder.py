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
from typing import List

from .dag import DAGNode, NodeKind, EdgeTransform
from pbrAudioCommon import EntityManager, LinkwitzRileyFilter

# Track names remain consistent
PHYSICS_FORCE_TRACKS = ["non_collision", "impact", "rolling", "sliding", "scraping", "coupling_strength"]
PHYSICS_NOISE_TRACKS = ["rolling_noise", "sliding_noise", "scraping_noise"]
RIGIDBODY_DISP_TRACKS = ["rigidbody", "rolling", "sliding", "scraping"]
RIGIDBODY_NOISE_TRACKS = ["rolling_noise", "sliding_noise", "scraping_noise"]


class GraphBuilder:
    """
    Reads EntityManager config and builds the DAG for the StorageEngine.
    """
    def __init__(self, entity_manager: EntityManager):
        self.em = entity_manager
        cfg = entity_manager.get("config")
        self.sr = int(cfg.system.sample_rate)
        self.dtype = np.dtype(np.float32)
        try:
            frequency_bands = self.em.get('frequency_bands')
            self.bands = len(frequency_bands.get_bands())
        except (KeyError, AttributeError):
            # Fallback if frequency_bands not registered yet
            self.bands = 24 # A sensible default

    def build_physics_graph(self, obj_idx: int, duration_s: float) -> DAGNode:
        n = int(self.sr * duration_s)
        n_forces = len(PHYSICS_FORCE_TRACKS)
        n_noise = len(PHYSICS_NOISE_TRACKS)

        root = DAGNode(
            kind=NodeKind.SOURCE,
            name=f"obj{obj_idx}/physics",
            shape=(n_forces, n),
            dtype=self.dtype,
            meta={"sr": self.sr, "force_names": PHYSICS_FORCE_TRACKS},
        )

        force_bands = DAGNode(
            kind=NodeKind.BAND,
            name=f"obj{obj_idx}/force_bands",
            shape=(n_forces, self.bands, n),
            dtype=self.dtype,
            meta={"sr": self.sr, "bands": self.bands},
        )
        force_bands.add_parent(root, EdgeTransform(
            name="filterbank_split",
            op=self._filterbank_split,
            params={"bands": self.bands, "sr": self.sr},
        ))

        noise_bands = DAGNode(
            kind=NodeKind.NOISE_ENH,
            name=f"obj{obj_idx}/noise_bands",
            shape=(n_noise, self.bands, n),
            dtype=self.dtype,
            meta={"sr": self.sr, "noise_names": PHYSICS_NOISE_TRACKS},
        )
        noise_bands.add_parent(force_bands, EdgeTransform(
            name="noise_enhance",
            op=self._noise_enhance,
            params={"noise_names": PHYSICS_NOISE_TRACKS},
        ))

        merged = DAGNode(
            kind=NodeKind.FORCE,
            name=f"obj{obj_idx}/force_block",
            shape=(n_forces + n_noise, self.bands, n),
            dtype=self.dtype,
            parents=[force_bands, noise_bands],
            meta={"sr": self.sr, "layout": PHYSICS_FORCE_TRACKS + PHYSICS_NOISE_TRACKS},
        )
        return merged

    def build_rigidbody_graph(self, obj_idx: int, duration_s: float, modal_count: int = 1024) -> DAGNode:
        n = int(self.sr * duration_s)
        n_disp = len(RIGIDBODYODY_DISP_TRACKS)
        n_noise = len(RIGIDBODY_NOISE_TRACKS)

        root = DAGNode(
            kind=NodeKind.SOURCE,
            name=f"obj{obj_idx}/rigidbody",
            shape=(n_disp, modal_count, self.bands, n),
            dtype=self.dtype,
            meta={"sr": self.sr, "modes": modal_count, "bands": self.bands, "disp_names": RIGIDBODY_DISP_TRACKS},
        )

        noise = DAGNode(
            kind=NodeKind.NOISE_ENH,
            name=f"obj{obj_idx}/rigidbody_noise",
            shape=(n_noise, modal_count, self.bands, n),
            dtype=self.dtype,
            meta={"sr": self.sr, "modes": modal_count, "bands": self.bands, "noise_names": RIGIDBODY_NOISE_TRACKS},
        )
        noise.add_parent(root, EdgeTransform(
            name="modal_noise_enhance",
            op=self._modal_noise_enhance,
            params={"noise_names": RIGIDBODY_NOISE_TRACKS},
        ))

        merged = DAGNode(
            kind=NodeKind.MODAL,
            name=f"obj{obj_idx}/rigidbody_block",
            shape=(n_disp + n_noise, modal_count, self.bands, n),
            dtype=self.dtype,
            parents=[root, noise],
            meta={"sr": self.sr, "modes": modal_count, "bands": self.bands, "layout": RIGIDBODY_DISP_TRACKS + RIGIDBODY_NOISE_TRACKS},
        )
        return merged

    def build_ambisonic_graph(self, obj_idx: int, duration_s: float, order: int = 1, n_mics: int = 16) -> DAGNode:
        n = int(self.sr * duration_s)
        n_channels = (order + 1) ** 2
        root = DAGNode(
            kind=NodeKind.AMBISONIC,
            name=f"obj{obj_idx}/ambisonic",
            shape=(n_mics, n_channels, n),
            dtype=self.dtype,
            meta={"sr": self.sr, "order": order, "n_mics": n_mics},
        )
        return root

    # --- Transform Ops ---
    # These are placeholders. In a real implementation, they would
    # use the actual data from the EntityManager and the LinkwitzRileyFilter.

    def _filterbank_split(self, chunk: np.ndarray, **params) -> np.ndarray:
        """Placeholder: splits a (n_forces, T) chunk into (n_forces, bands, T)."""
        n_forcesces, T = chunk.shape
        bands = params["bands"]
        return np.zeros((n_forces, bands, T), dtype=chunk.dtype)

    def _noise_enhance(self, chunk: np.ndarray, **params) -> np.ndarray:
        """Placeholder: generates noise from (n_forces, bands, T) into (n_noise, bands, T)."""
        n_noise = len(params["noise_names"])
        _, bands, T = chunk.shape
        return np.zeros((n_noise, bands, T), dtype=chunk.dtype)

    def _modal_noise_enhance(self, chunk: np.ndarray, **params) -> np.ndarray:
        """Placeholder: generates noise from (n_disp, modes, bands, T) into (n_noise, modes, bands, T)."""
        n_noise = len(params["noise_names"])
        _, modes, bands, T = chunk.shape
        return np.zeros((n_noise, modes, bands, T), dtype=chunk.dtype)

