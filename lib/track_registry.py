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

from typing import Dict, List
from .dag import TrackDescriptor

class TrackRegistry:
    """
    A central registry for all TrackDescriptors.
    Engines from pbrAudioShaders register their track definitions here.
    """
    _descriptors: Dict[str, TrackDescriptor] = {}

    @classmethod
    def register(cls, descriptor: TrackDescriptor) -> None:
        """Registers a new track descriptor."""
        if descriptor.name in cls._descriptors:
            raise ValueError(f"A descriptor with name '{descriptor.name}' is already registered.")
        cls._descriptors[descriptor.name] = descriptor
        print(f"Registered track descriptor: '{descriptor.name}'")

    @classmethod
    def get(cls, name: str) -> TrackDescriptor | None:
        """Retrieves a descriptor by name."""
        return cls._descriptors.get(name)

    @classmethod
    def get_all(cls) -> Dict[str, TrackDescriptor]:
        """Retrieves all registered descriptors."""
        return cls._descriptors

    @classmethod
    def clear(cls) -> None:
        """Clears the registry. Useful for testing."""
        cls._descriptors.clear()

