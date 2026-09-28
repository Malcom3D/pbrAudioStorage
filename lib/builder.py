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

# pbraudio/storage/builder.py
from __future__ import annotations
import numpy as np

from .dag import DAGNode, NodeKind, EdgeTransform


# Canonical track layouts — keep these in one place so physicsSolver,
# rigidBody, and StorageEngine all agree.
PHYSICS_FORCE_TRACKS = ["non_collision", "impact", "rolling", "sliding", "scraping", "coupling_strength"]
PHYSICS_NOISE_TRACKS = ["rolling_noise", "sliding_noise", "scraping_noise"]
RIGIDBODY_DISP_TRACKS = ["rigidbody", "rollingrolling", "sliding", "scraping"]
RIGIDBODY_NOISE_TRACKS = ["rolling_noise", "sliding_noise", "scraping_noise"]


class GraphBuilder:
    """
    Reads EntityManager config + shader introspection and produces
    the DAG that StorageEngine materializes.
    """

    def __init__(self, entity_manager, shader_engine=None):
        self.em = entity_manager
        self.shaders = shader_engine
        cfg = entity_manager.get("config")
        self.sr = int(cfg.system.sample_rate)
        self.fps = cfg.system.fps
        self.fps_base = cfg.system.fps_base
        self.subframes = cfg.system.subframes
        self.bands = int(cfg.system.bands_per_octave) or 1
        self.dtype = np.dtype(np.float32)  # or read from config

    # ---------- physics graph ----------

    def build_physics_graph(self, obj_idx: int, duration_s: float) -> DAGNode:
        n = int(self.sr * duration_s)
        n_forces = len(PHYSICS_FORCE_TRACKS)
        n_noise = len(PHYSICS_NOISE_TRACKS)
        bands = self.bands

        # root: raw physics force tracks
        root = DAGNode(
            kind=NodeKind.SOURCE,
            name=f"obj{obj_idx}/physics",
            shape=(n_forces, n),
            dtype=self.dtype,
            meta={
                "sr": self.sr,
                "force_names": PHYSICS_FORCE_TRACKS,
            },
        )

        # band-split: (n_forces, bands, N)
        force_bands = DAGNode(
            kind=NodeKind.BAND,
            name=f"obj{obj_idx}/force_bands",
            shape=(n_forces, bands, n),
            dtype=self.dtype,
            meta={"sr": self.sr, "bands": bands},
        )
        force_bands.add_parent(root, EdgeTransform(
            name="filterbank_split",
            op=self._filterbank_split,
            params={"bands": bands, "sr": self.sr},
        ))

        # noise enhancement: (n_noise, bands, N)
        noise_bands = DAGNode(
            kind=NodeKind.NOISE_ENH,
            name=f"obj{obj_idx}/noise_bands",
            shape=(n_noise, bands, n),
            dtype=self.dtype,
            meta={"sr": self.sr, "noise_names": PHYSICS_NOISE_TRACKS},
        )
        noise_bands.add_parent(force_bands, EdgeTransform(
            name="noise_enhance",
            op=self._noise_enhance,
            params={"noise_names": PHYSICS_NOISE_TRACKS},
        ))

        # merge: (n_forces + n_noise, bands, N)
        merged = DAGNode(
            kind=NodeKind.FORCE,
            name=f"obj{obj_idx}/force_block",
            shape=(n_forces + n_noise, bands, n),
            dtype=self.dtype,
            parents=[force_bands, noise_bands],
            meta={
                "sr": self.sr,
                "layout": PHYSICS_FORCE_TRACKS + PHYSICS_NOISE_TRACKS,
            },
        )
        return merged

    # ---------- rigidbody graph ----------

    def build_rigidbody_graph(self, obj_idx: int,
                              duration_s: float,
                              modal_count: int = 1024) -> DAGNode:
        n = int(self.sr * duration_s)
        n_disp = len(RIGIDBODY_DISP_TRACKS)
        n_noise = len(RIGIDBODY_NOISE_TRACKS)
        bands = self.bands
        modes = modal_count

        # root: modal displacement (4, modes, bands, N)
        root = DAGNode(
            kind=NodeKind.SOURCE,
            name=f"obj{obj_idx}/rigidbody",
            shape=(n_disp, modes, bands, n),
            dtype=self.dtype,
            meta={
                "sr": self.sr,
                "modes": modes,
                "bands": bands,
                "disp_names": RIGIDBODY_DISP_TRACKS,
            },
        )

        # noise enhancement on modal displacement
        noise = DAGNode(
            kind=NodeKind.NOISE_ENH,
            name=f"obj{obj_idx}/rigidbody_noise",
            shape=(n_noise, modes, bands, n),
            dtype=self.dtype,
            meta={
                "sr": self.sr,
                "modes": modes,
                "bands": bands,
                "noise_names": RIGIDBODY_NOISE_TRACKS,
            },
        )
        noise.add_parent(root, EdgeTransform(
            name="modal_noise_enhance",
            op=self._modal_noise_enhance,
            params={"noise_names": RIGIDBODY_NOISE_TRACKS},
        ))

        # merged: (n_disp + n_noise, modes, bands, N)
        merged = DAGNode(
            kind=NodeKind.MODAL,
            name=f"obj{obj_idx}/rigidbody_block",
            shape=(n_disp + n_noise, modes, bands, n),
            dtype=self.dtype,
            parents=[root, noise],
            meta={
                "sr": self.sr,
                "modes": modes,
                "bands": bands,
                "layout": RIGIDBODY_DISP_TRACKS + RIGIDBODY_NOISE_TRACKS,
            },
        )
        return merged

    # ---------- ambisonic output graph ----------

    def build_ambisonic_graph(self, obj_idx: int,
                              duration_s: float,
                              order: int = 1,
                              n_mics: int = 16) -> DAGNode:
        """
        Acoustic renderer output. 16 rendered ambisonic tracks per mic.
        """
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

    # ---------- transform ops (chunked numpy fallback) ----------

    def _filterbank_split(self, chunk: np.ndarray, **params) -> np.ndarray:
        """
        Input:  (n_forces, T)
        Output: (n_forces, bands, T)

        Replace with your LinkwitzRileyFilter-based split.
        """
        bands = params["bands"]
        n_for_forces, T = chunk.shape
        out = np.zeros((n_forces, bands, T), dtype=chunk.dtype)
        for b in range(bands):
            out[:, b, :] = chunk  # placeholder
        return out

    def _noise_enhance(self, chunk: np.ndarray, **params) -> np.ndarray:
        """
        Input:  (n_forces, bands, T)
        Output: (n_noise, bands, T)
        """
        n_noise = len(params["noise_names"])
        _, bands, T = chunk.shape
        return np.zeros((n_noise, bands, T), dtype=chunk.dtype)

    def _modal_noise_enhance(self, chunk: np.ndarray, **params) -> np.ndarray:
        """
        Input:  (n_disp, modes, bands, T)
        Output: (n_noise, modes, bands, T)
        """
        n_noise = len(params["noise_names"])
        _, modes, bands, T = chunk.shape
        return np.zeros((n_noise, modes, bands, T), dtype=chunk.dtype)

