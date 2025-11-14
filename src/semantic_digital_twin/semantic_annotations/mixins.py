from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from functools import lru_cache
from typing import ClassVar, Set, List, Optional, Self, Iterable

import numpy as np
import trimesh
from typing_extensions import TYPE_CHECKING

from ..utils import camel_case_split
from ..datastructures.prefixed_name import PrefixedName
from ..spatial_types import Point3
from ..world_description.world_entity import (
    SemanticAnnotation,
    Body,
    Region,
    KinematicStructureEntity,
)

if TYPE_CHECKING:
    from .semantic_annotations import (
        Drawer,
        Door,
    )


@dataclass(eq=False)
class HasBody(SemanticAnnotation, ABC):
    """
    Abstract base class for all household objects. Each semantic annotation refers to a single Body.
    Each subclass automatically derives a MatchRule from its own class name and
    the names of its HouseholdObject ancestors. This makes specialized subclasses
    naturally more specific than their bases.
    """

    body: Body = field(kw_only=True)

    @property
    def bodies(self) -> Iterable[Body]:
        return [self.body]


@dataclass(eq=False)
class HasRegion(SemanticAnnotation, ABC):
    """
    Abstract base class for all household objects. Each semantic annotation refers to a single Body.
    Each subclass automatically derives a MatchRule from its own class name and
    the names of its HouseholdObject ancestors. This makes specialized subclasses
    naturally more specific than their bases.
    """

    region: Region = field(kw_only=True)


@dataclass(eq=False)
class HasDrawers(SemanticAnnotation):
    """
    A mixin class for semantic annotations that have drawers.
    """

    drawers: List[Drawer] = field(default_factory=list, hash=False, kw_only=True)


@dataclass(eq=False)
class HasDoors(SemanticAnnotation):
    """
    A mixin class for semantic annotations that have doors.
    """

    doors: List[Door] = field(default_factory=list, hash=False, kw_only=True)


@dataclass(eq=False)
class HasSupportingSurface(SemanticAnnotation):
    """
    A semantic annotation that represents a supporting surface.
    """

    supporting_surface: Optional[Region] = None
    """
    The area that represents the supporting surface.
    """

    @classmethod
    def create_from_entity(
        cls,
        entity: KinematicStructureEntity,
        upward_threshold: float = 0.95,
        clearance_threshold: float = 0.5,
        min_surface_area: float = 0.0225,  # 15cm x 15cm
    ) -> Self:

        mesh = entity.combined_mesh
        if mesh is None:
            return cls()
        # --- Find upward-facing faces ---
        normals = mesh.face_normals
        upward_mask = normals[:, 2] > upward_threshold

        if not upward_mask.any():
            return cls()

        # --- Find connected upward-facing regions ---
        upward_face_indices = np.nonzero(upward_mask)[0]
        submesh_up = mesh.submesh([upward_face_indices], append=True)
        face_groups = submesh_up.split(only_watertight=False)

        # Compute total area for each group
        large_groups = [g for g in face_groups if g.area >= min_surface_area]

        if not large_groups:
            return cls()

        # --- Merge qualifying upward-facing submeshes ---
        candidates = trimesh.util.concatenate(large_groups)

        # --- Check vertical clearance using ray casting ---
        face_centers = candidates.triangles_center
        ray_origins = face_centers + np.array([0, 0, 0.01])  # small upward offset
        ray_dirs = np.tile([0, 0, 1], (len(ray_origins), 1))

        locations, index_ray, _ = mesh.ray.intersects_location(
            ray_origins=ray_origins, ray_directions=ray_dirs
        )

        # Compute distances to intersections (if any)
        distances = np.full(len(ray_origins), np.inf)
        distances[index_ray] = np.linalg.norm(
            locations - ray_origins[index_ray], axis=1
        )

        # Filter faces with enough space above
        clear_mask = (distances > clearance_threshold) | np.isinf(distances)

        if not clear_mask.any():
            raise ValueError(
                "No upward-facing surfaces with sufficient clearance found."
            )

        candidates_filtered = candidates.submesh([clear_mask], append=True)

        # --- Build the region ---
        points_3d = [
            Point3(
                x,
                y,
                z,
                reference_frame=entity,
            )
            for x, y, z in candidates_filtered.vertices
        ]

        surface_region = Region.from_3d_points(
            name=PrefixedName(f"{entity.name.name}_surface_region"),
            points_3d=points_3d,
            reference_frame=entity,
        )

        return cls(supporting_surface=surface_region)


@dataclass(eq=False)
class Furniture(SemanticAnnotation): ...
