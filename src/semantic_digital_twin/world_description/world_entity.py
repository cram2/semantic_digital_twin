from __future__ import annotations

import importlib
import inspect
import itertools
from abc import ABC
from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from dataclasses import fields
from functools import lru_cache

import numpy as np
import trimesh
import trimesh.boolean
from krrood.entity_query_language.predicate import Symbol
from random_events.utils import SubclassJSONSerializer
from scipy.stats import geom
from trimesh.proximity import closest_point, nearby_faces
from trimesh.sample import sample_surface
from typing_extensions import (
    Deque,
    Type,
    TypeVar,
    Dict,
    Any,
    Self,
)
from typing_extensions import List, Optional, TYPE_CHECKING, Tuple
from typing_extensions import Set

from .geometry import TriangleMesh
from .shape_collection import ShapeCollection, BoundingBoxCollection
from ..datastructures.prefixed_name import PrefixedName
from ..exceptions import ReferenceFrameMismatchError
from ..spatial_types import spatial_types as cas
from ..spatial_types.spatial_types import TransformationMatrix, Expression, Point3
from ..utils import IDGenerator, type_string_to_type

if TYPE_CHECKING:
    from ..world_description.degree_of_freedom import DegreeOfFreedom
    from ..world import World

id_generator = IDGenerator()


@dataclass(unsafe_hash=True)
class WorldEntity(Symbol):
    """
    A class representing an entity in the world.
    """

    _world: Optional[World] = field(default=None, repr=False, kw_only=True, hash=False)
    """
    The backreference to the world this entity belongs to.
    """

    _semantic_annotations: Set[SemanticAnnotation] = field(
        default_factory=set, init=False, repr=False, hash=False
    )
    """
    The semantic annotations this entity is part of.
    """

    name: PrefixedName = field(default=None, kw_only=True)
    """
    The identifier for this world entity.
    """

    def __post_init__(self):
        if self.name is None:
            self.name = PrefixedName(f"{self.__class__.__name__}_{hash(self)}")


@dataclass
class CollisionCheckingConfig:
    buffer_zone_distance: Optional[float] = None
    """
    Distance defining a buffer zone around the entity. The buffer zone represents a soft boundary where
    proximity should be monitored but minor violations are acceptable.
    """

    violated_distance: float = 0.0
    """
    Critical distance threshold that must not be violated. Any proximity below this threshold represents
    a severe collision risk requiring immediate attention.
    """

    disabled: Optional[bool] = None
    """
    Flag to enable/disable collision checking for this entity. When True, all collision checks are ignored.
    """

    max_avoided_bodies: int = 1
    """
    Maximum number of other bodies this body should avoid simultaneously.
    If more bodies than this are in the buffer zone, only the closest ones are avoided.
    """


@dataclass(unsafe_hash=True)
class KinematicStructureEntity(WorldEntity, SubclassJSONSerializer, ABC):
    """
    An entity that is part of the kinematic structure of the world.
    """

    index: Optional[int] = field(default=None, init=False)
    """
    The index of the entity in `_world.kinematic_structure`.
    """

    @property
    def global_pose(self) -> TransformationMatrix:
        """
        Computes the pose of the KinematicStructureEntity in the world frame.
        :return: TransformationMatrix representing the global pose.
        """
        return self._world.compute_forward_kinematics(self._world.root, self)

    @property
    def parent_connection(self) -> Connection:
        """
        Returns the parent connection of this KinematicStructureEntity.
        """
        return self._world.compute_parent_connection(self)

    @property
    def child_kinematic_structure_entities(self) -> List[KinematicStructureEntity]:
        """
        Returns the direct child KinematicStructureEntity of this entity.
        """
        return self._world.compute_child_kinematic_structure_entities(self)

    @property
    def parent_kinematic_structure_entity(self) -> KinematicStructureEntity:
        """
        Returns the parent KinematicStructureEntity of this entity.
        """
        return self._world.compute_parent_kinematic_structure_entity(self)

    def get_first_parent_connection_of_type(
        self, connection_type: Type[GenericConnection]
    ) -> GenericConnection:
        """
        Traverse the chain up until an active connection is found.
        """
        if self == self._world.root:
            raise ValueError(
                f"Cannot get controlled parent connection for root body {self._world.root.name}."
            )
        if isinstance(self.parent_connection, connection_type):
            return self.parent_connection
        return self.parent_connection.parent.get_first_parent_connection_of_type(
            connection_type
        )


@dataclass
class Body(KinematicStructureEntity, SubclassJSONSerializer):
    """
    Represents a body in the world.
    A body is a semantic atom, meaning that it cannot be decomposed into meaningful smaller parts.
    """

    visual: ShapeCollection = field(default_factory=ShapeCollection, repr=False)
    """
    List of shapes that represent the visual appearance of the link.
    The poses of the shapes are relative to the link.
    """

    collision: ShapeCollection = field(default_factory=ShapeCollection, repr=False)
    """
    List of shapes that represent the collision geometry of the link.
    The poses of the shapes are relative to the link.
    """

    collision_config: Optional[CollisionCheckingConfig] = field(
        default_factory=CollisionCheckingConfig
    )
    """
    Configuration for collision checking.
    """

    temp_collision_config: Optional[CollisionCheckingConfig] = None
    """
    Temporary configuration for collision checking, takes priority over `collision_config`.
    """

    index: Optional[int] = field(default=None, init=False)
    """
    The index of the entity in `_world.kinematic_structure`.
    """

    def __post_init__(self):
        if not self.name:
            self.name = PrefixedName(f"body_{id_generator(self)}")

        if self._world is not None:
            self.index = self._world.kinematic_structure.add_node(self)

        self.visual.reference_frame = self
        self.collision.reference_frame = self
        self.collision.transform_all_shapes_to_own_frame()
        self.visual.transform_all_shapes_to_own_frame()

    def get_collision_config(self) -> CollisionCheckingConfig:
        if self.temp_collision_config is not None:
            return self.temp_collision_config
        return self.collision_config

    def set_static_collision_config(self, collision_config: CollisionCheckingConfig):
        merged_config = CollisionCheckingConfig(
            buffer_zone_distance=(
                collision_config.buffer_zone_distance
                if collision_config.buffer_zone_distance is not None
                else self.collision_config.buffer_zone_distance
            ),
            violated_distance=collision_config.violated_distance,
            disabled=(
                collision_config.disabled
                if collision_config.disabled is not None
                else self.collision_config.disabled
            ),
            max_avoided_bodies=collision_config.max_avoided_bodies,
        )
        self.collision_config = merged_config

    def set_static_collision_distances(
        self, buffer_zone_distance: float, violated_distance: float
    ):
        self.collision_config.buffer_zone_distance = buffer_zone_distance
        self.collision_config.violated_distance = violated_distance

    def set_temporary_collision_config(self, collision_config: CollisionCheckingConfig):
        self.temp_collision_config = collision_config

    def reset_temporary_collision_config(self):
        self.temp_collision_config = None

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if other is None:
            return False
        return self.name == other.name and self._world is other._world

    def has_collision(
        self, volume_threshold: float = 1.001e-6, surface_threshold: float = 0.00061
    ) -> bool:
        """
        Check if collision geometry is mesh or simple shape with volume/surface bigger than thresholds.

        :param volume_threshold: Ignore simple geometry shapes with a volume less than this (in m^3)
        :param surface_threshold: Ignore simple geometry shapes with a surface area less than this (in m^2)
        :return: True if collision geometry is mesh or simple shape exceeding thresholds
        """
        return len(self.collision) > 0

    def compute_closest_points_multi(
        self, others: list[Body], sample_size=25
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes the closest points to each given body respectively.

        :param others: The list of bodies to compute the closest points to.
        :param sample_size: The number of samples to take from the surface of the other bodies.
        :return: A tuple containing: The points on the self body, the points on the other bodies, and the distances. All points are in the of this body.
        """

        @lru_cache(maxsize=None)
        def evaluated_geometric_distribution(n: int) -> np.ndarray:
            """
            Evaluates the geometric distribution for a given number of samples.
            :param n: The number of samples to evaluate.
            :return: An array of probabilities for each sample.
            """
            return geom.pmf(np.arange(1, n + 1), 0.5)

        query_points = []
        for other in others:
            # Calculate the closest vertex on this body to the other body
            closest_vert_id = self.collision[0].mesh.kdtree.query(
                (
                    self._world.compute_forward_kinematics_np(self, other)
                    @ other.collision[0].origin.to_np()
                )[:3, 3],
                k=1,
            )[1]
            closest_vert = self.collision[0].mesh.vertices[closest_vert_id]

            # Compute the closest faces on the other body to the closes vertex
            faces = nearby_faces(
                other.collision[0].mesh,
                [
                    (
                        self._world.compute_forward_kinematics_np(other, self)
                        @ self.collision[0].origin.to_np()
                    )[:3, 3]
                    + closest_vert
                ],
            )[0]
            face_weights = np.zeros(len(other.collision[0].mesh.faces))

            # Assign weights to the faces based on a geometric distribution
            face_weights[faces] = evaluated_geometric_distribution(len(faces))

            # Sample points on the surface of the other body
            q = sample_surface(
                other.collision[0].mesh, sample_size, face_weight=face_weights, seed=420
            )[0]
            # Make 4x4 transformation matrix from points
            points = np.tile(np.eye(4, dtype=np.float32), (len(q), 1, 1))
            points[:, :3, 3] = q

            # Transform from the mesh to the other mesh
            transform = (
                np.linalg.inv(self.collision[0].origin.to_np())
                @ self._world.compute_forward_kinematics_np(self, other)
                @ other.collision[0].origin.to_np()
            )
            points = points @ transform

            points = points[
                :, :3, 3
            ]  # Extract the points from the transformation matrix

            query_points.extend(points)

        # Actually compute the closest points
        points, dists = closest_point(self.collision[0].mesh, query_points)[:2]
        # Find the closest points for each body out of all the sampled points
        points = np.array(points).reshape(len(others), sample_size, 3)
        dists = np.array(dists).reshape(len(others), sample_size)
        dist_min = np.min(dists, axis=1)
        points_min_self = points[np.arange(len(others)), np.argmin(dists, axis=1), :]
        points_min_other = np.array(query_points).reshape(len(others), sample_size, 3)[
            np.arange(len(others)), np.argmin(dists, axis=1), :
        ]
        return points_min_self, points_min_other, dist_min

    def to_json(self) -> Dict[str, Any]:
        result = super().to_json()
        result["name"] = self.name.to_json()
        result["collision"] = self.collision.to_json()
        result["visual"] = self.visual.to_json()
        return result

    @classmethod
    def _from_json(cls, data: Dict[str, Any]) -> Self:

        result = cls(name=PrefixedName.from_json(data["name"]))

        collision = ShapeCollection.from_json(data["collision"])
        visual = ShapeCollection.from_json(data["visual"])

        for shape in itertools.chain(collision, visual):
            shape.origin.reference_frame = result

        result.collision = collision
        result.visual = visual

        return result


@dataclass
class Region(KinematicStructureEntity):
    """
    Virtual KinematicStructureEntity representing a semantic region in the world.
    """

    area: ShapeCollection = field(default_factory=ShapeCollection, hash=False)
    """
    The shapes that represent the area of the region.
    """

    def __post_init__(self):
        self.area.reference_frame = self

    def __hash__(self):
        return id(self)

    @classmethod
    def from_3d_points(
        cls,
        name: PrefixedName,
        points_3d: List[Point3],
        reference_frame: Optional[Body] = None,
        minimum_thickness: float = 0.005,
        sv_ratio_tol: float = 1e-7,
    ) -> Self:
        """
        Constructs a Region from a list of 3D points by creating a convex hull around them.
        The points are analyzed to determine if they are approximately planar. If they are,
        a minimum thickness is added to ensure the region has a non-zero volume.

        :param name: Prefixed name for the region.
        :param points_3d: List of 3D points.
        :param reference_frame: Optional reference frame.
        :param minimum_thickness: Minimum thickness to add if points are near-planar.
        :param sv_ratio_tol: Tolerance for determining planarity based on singular value ratio.

        :return: Region object.
        """
        points = np.asarray([point.to_np()[:3] for point in points_3d], dtype=float)
        points = np.unique(points, axis=0)
        assert (
            len(points) >= 3
        ), "At least 4 unique points are required to define a 3D region."

        centered_points = points - points.mean(axis=0, keepdims=True)
        assert np.any(centered_points), "Points must not be all identical."

        # We compute the principal axes of the point cloud using SVD.
        # This allows us to reason about the geometric thickness of our point cloud.
        # The axis with the smallest variance, located at the last index if our `principal_axis` is our `normal`
        # indicating the direction of the region's thickness.
        _, variance, principal_axis = np.linalg.svd(
            centered_points, full_matrices=False
        )
        smallest_variance_axis = principal_axis[-1]  # this is our normal
        unit_vector_normal = smallest_variance_axis / np.linalg.norm(
            smallest_variance_axis
        )

        # We compute the thickness, peak-to-peak (max - min), along the normal direction, to get the thickness of
        # the region.
        thickness_in_normal_direction = np.ptp(centered_points @ unit_vector_normal)
        is_near_planar = variance[0] > 0 and variance[-1] / variance[0] < sv_ratio_tol
        thickness_padding = (
            minimum_thickness / 2
            if thickness_in_normal_direction < minimum_thickness or is_near_planar
            else 0.0
        )

        # We do not provide any 2d shapes, since they would be very weird to handle with raytracing etc.
        # Thus we decided that in near-planar cases we add a minimum thickness to ensure we get a 3d shape.
        if thickness_padding > 0:
            P_aug = np.vstack(
                [
                    points + thickness_padding * unit_vector_normal,
                    points - thickness_padding * unit_vector_normal,
                ]
            )
        else:
            P_aug = points

        hull = trimesh.points.PointCloud(P_aug).convex_hull
        hull.remove_unreferenced_vertices()
        hull.remove_degenerate_faces()
        hull.process()

        area_mesh = TriangleMesh(
            mesh=hull, origin=TransformationMatrix(reference_frame=reference_frame)
        )
        return cls(name=name, area=ShapeCollection([area_mesh]))

    def to_json(self) -> Dict[str, Any]:
        result = super().to_json()
        result["name"] = self.name.to_json()
        result["area"] = self.area.to_json()
        return result

    @classmethod
    def _from_json(cls, data: Dict[str, Any]) -> Self:
        result = cls(name=PrefixedName.from_json(data["name"]))
        area = ShapeCollection.from_json(data["area"])
        for shape in area:
            shape.origin.reference_frame = result
        result.area = area
        return result


GenericKinematicStructureEntity = TypeVar(
    "GenericKinematicStructureEntity", bound=KinematicStructureEntity
)

GenericWorldEntity = TypeVar("GenericWorldEntity", bound=WorldEntity)

@dataclass
class SemanticAnnotation(WorldEntity, SubclassJSONSerializer):
    """
    Represents a semantic annotation on a set of bodies in the world.

    This class can hold references to certain bodies that gain meaning in this context.

    .. warning::

        The hash of a semantic annotation is based on the hash of its type and kinematic structure entities.
        Overwrite this with extreme care and only if you know what you are doing. Hashes are used inside rules to check if
        a new semantic annotation has been created. If you, for instance, just use the object identity, this will fail since python assigns
        new memory pointers always. The same holds for the equality operator.
        If you do not want to change the behavior, make sure to use @dataclass(eq=False) to decorate your class.
    """

    def __post_init__(self):
        if self.name is None:
            self.name = PrefixedName(
                name=f"{self.__class__.__name__}_{id_generator(self)}",
                prefix=self._world.name if self._world is not None else None,
            )

    def __hash__(self):
        return hash(
            tuple(
                [self.__class__]
                + sorted([kse.name for kse in self.kinematic_structure_entities])
            )
        )

    def __eq__(self, other):
        return hash(self) == hash(other)

    def to_json(self) -> Dict[str, Any]:
        result = {
            **super().to_json(),
        }

        for semantic_annotation_field in fields(self):
            value = getattr(self, semantic_annotation_field.name)
            if issubclass(type(value), SubclassJSONSerializer):
                result[semantic_annotation_field.name] = value.to_json()
        return result

    @classmethod
    def _from_json(cls, data: Dict[str, Any]) -> Self:
        semantic_annotation_fields = {f.name: f for f in fields(cls)}

        init_args = {}

        for k, v in semantic_annotation_fields.items():
            if k not in data.keys():
                continue
            field_type = type_string_to_type(data[k]["type"])
            if issubclass(field_type, SubclassJSONSerializer):
                init_args[k] = field_type.from_json(data[k])

        return cls(**init_args)

    def _kinematic_structure_entities(
        self, visited: Set[int], aggregation_type: Type[GenericKinematicStructureEntity]
    ) -> Set[GenericKinematicStructureEntity]:
        """
        Recursively collects all entities that are part of this semantic annotation.
        """
        stack: Deque[object] = deque([self])
        entities: Set[aggregation_type] = set()

        while stack:
            obj = stack.pop()
            oid = id(obj)
            if oid in visited:
                continue
            visited.add(oid)

            match obj:
                case aggregation_type():
                    entities.add(obj)

                case SemanticAnnotation():
                    stack.extend(_attr_values(obj, aggregation_type))

                case Mapping():
                    stack.extend(
                        v
                        for v in obj.values()
                        if _is_entity_semantic_annotation_or_iterable(
                            v, aggregation_type
                        )
                    )

                case Iterable() if not isinstance(obj, (str, bytes, bytearray)):
                    stack.extend(
                        v
                        for v in obj
                        if _is_entity_semantic_annotation_or_iterable(
                            v, aggregation_type
                        )
                    )

        return entities

    @property
    def kinematic_structure_entities(self) -> Iterable[KinematicStructureEntity]:
        """
        Returns a Iterable of all relevant KinematicStructureEntity in this semantic annotation. The default behaviour is to aggregate all KinematicStructureEntity that are accessible
        through the properties and fields of this semantic annotation, recursively.
        If this behaviour is not desired for a specific semantic annotation, it can be overridden by implementing the `KinematicStructureEntity` property.
        """
        return self._kinematic_structure_entities(set(), KinematicStructureEntity)

    @property
    def bodies(self) -> Iterable[Body]:
        """
        Returns an Iterable of all relevant bodies in this semantic annotation. The default behaviour is to aggregate all bodies that are accessible
        through the properties and fields of this semantic annotation, recursively.
        If this behaviour is not desired for a specific semantic annotation, it can be overridden by implementing the `bodies` property.
        """
        return self._kinematic_structure_entities(set(), Body)

    @property
    def regions(self) -> Iterable[Region]:
        """
        Returns an Iterable of all relevant regions in this semantic annotation. The default behaviour is to aggregate all regions that are accessible
        through the properties and fields of this semantic annotation, recursively.
        If this behaviour is not desired for a specific semantic annotation, it can be overridden by implementing the `regions` property.
        """
        return self._kinematic_structure_entities(set(), Region)

    def as_bounding_box_collection_at_origin(
        self, origin: TransformationMatrix
    ) -> BoundingBoxCollection:
        """
        Returns a bounding box collection that contains the bounding boxes of all bodies in this semantic annotation.
        :param reference_frame: The reference frame to express the bounding boxes in.
        :returns: A collection of bounding boxes in world-space coordinates.
        """

        collections = iter(
            entity.collision.as_bounding_box_collection_at_origin(origin)
            for entity in self.kinematic_structure_entities
            if isinstance(entity, Body) and entity.has_collision()
        )
        bbs = BoundingBoxCollection([], origin.reference_frame)

        for bb_collection in collections:
            bbs = bbs.merge(bb_collection)

        return bbs

    def as_bounding_box_collection_in_frame(
        self, reference_frame: KinematicStructureEntity
    ) -> BoundingBoxCollection:
        """
        Provides the bounding box collection for this entity in the given reference frame.
        :param reference_frame: The reference frame to express the bounding boxes in.
        :returns: A collection of bounding boxes in world-space coordinates.
        """
        return self.as_bounding_box_collection_at_origin(
            TransformationMatrix(reference_frame=reference_frame)
        )


@dataclass(eq=False)
class RootedSemanticAnnotation(SemanticAnnotation):
    """
    Represents a semantic annotation that is rooted in a specific KinematicStructureEntity.
    """

    root: Body = field(default=None)

    @property
    def connections(self) -> List[Connection]:
        return self._world.get_connections_of_branch(self.root)

    @property
    def bodies(self) -> List[Body]:
        return self._world.get_bodies_of_branch(self.root)

    @property
    def bodies_with_collisions(self) -> List[Body]:
        return [x for x in self.bodies if x.has_collision()]

    @property
    def bodies_with_enabled_collision(self) -> Set[Body]:
        return set(
            body
            for body in self.bodies
            if body.has_collision() and not body.get_collision_config().disabled
        )


@dataclass(eq=False)
class SemanticEnvironmentAnnotation(RootedSemanticAnnotation):
    """
    Represents a semantic annotation of the environment.
    """

    @property
    def kinematic_structure_entities(self) -> Set[KinematicStructureEntity]:
        """
        Returns a set of all KinematicStructureEntity in the environment semantic annotation.
        """
        return set(
            self._world.compute_descendent_child_kinematic_structure_entities(self.root)
        ) | {self.root}


@dataclass
class Connection(WorldEntity):
    """
    Represents a connection between two entities in the world.
    """

    parent: KinematicStructureEntity
    """
    The parent KinematicStructureEntity of the connection.
    """

    child: KinematicStructureEntity
    """
    The child KinematicStructureEntity of the connection.
    """

    parent_T_connection_expression: TransformationMatrix = field(
        default_factory=TransformationMatrix
    )
    connection_T_child_expression: TransformationMatrix = field(
        default_factory=TransformationMatrix
    )
    """
    The origin expression of a connection is split into 2 transforms:
    1. parent_T_connection describes the pose of the connection and is always constant.
       It typically describes the fixed part of the origin expression, equivalent to the origin tag in urdf. 
       For example, it is the point about which a revolute joint rotates.
    2. connection_T_child describes the pose of the child relative to the connection.
       This typically contains only the expressions that describe how the degrees of freedom move the child.
       For example, it describes how the angle of a revolute joint affects the child pose.

    This split is necessary for copying Connections, because they need parent_T_connection as an input parameter and 
    connection_T_child is generated in the __post_init__ method.
    """

    @property
    def origin_expression(self) -> TransformationMatrix:
        return self.parent_T_connection_expression @ self.connection_T_child_expression

    @property
    def has_hardware_interface(self) -> bool:
        return False

    def add_to_world(self, world: World):
        self._world = world

    def __post_init__(self):
        if self.name is None:
            self.name = PrefixedName(
                f"{self.parent.name.name}_T_{self.child.name.name}",
                prefix=self.child.name.prefix,
            )

        if (
            self.parent_T_connection_expression.reference_frame is not None
            and self.parent_T_connection_expression.reference_frame != self.parent
        ):
            raise ReferenceFrameMismatchError(
                self.parent, self.parent_T_connection_expression.reference_frame
            )

        self.parent_T_connection_expression.reference_frame = self.parent
        self.connection_T_child_expression.child_frame = self.child

    def _post_init_world_part(self):
        """
        Executes post-initialization logic based on the presence of a world attribute.
        """
        if self._world is None:
            self._post_init_without_world()
        else:
            self._post_init_with_world()

    def _post_init_with_world(self):
        """
        Initialize or perform additional setup operations required after the main
        initialization step. Use for world-related configurations or specific setup
        details required post object creation.
        """
        pass

    def _post_init_without_world(self):
        """
        Handle internal initialization processes when _world is None. Perform
        operations post-initialization for internal use only.
        """
        pass

    def __hash__(self):
        return hash((self.parent, self.child))

    def __eq__(self, other):
        return self.name == other.name

    @property
    def origin(self) -> cas.TransformationMatrix:
        """
        :return: The relative transform between the parent and child frame.
        """
        return self._world.compute_forward_kinematics(self.parent, self.child)

    def origin_as_position_quaternion(self) -> Expression:
        position = self.origin_expression.to_position()[:3]
        orientation = self.origin_expression.to_quaternion()
        return cas.Expression.vstack([position, orientation]).T

    @property
    def dofs(self) -> Set[DegreeOfFreedom]:
        """
        Returns the degrees of freedom associated with this connection.
        """
        dofs = set()

        if hasattr(self, "active_dofs"):
            dofs.update(set(self.active_dofs))
        if hasattr(self, "passive_dofs"):
            dofs.update(set(self.passive_dofs))

        return dofs


GenericConnection = TypeVar("GenericConnection", bound=Connection)


def _is_entity_semantic_annotation_or_iterable(
    obj: object, aggregation_type: Type[KinematicStructureEntity]
) -> bool:
    """
    Determines if an object is a KinematicStructureEntity, a semantic annotation, or an Iterable (excluding strings and bytes).
    """
    return isinstance(obj, (aggregation_type, SemanticAnnotation)) or (
        isinstance(obj, Iterable) and not isinstance(obj, (str, bytes, bytearray))
    )


def _attr_values(
    semantic_annotation: SemanticAnnotation,
    aggregation_type: Type[GenericKinematicStructureEntity],
) -> Iterable[object]:
    """
    Yields all dataclass fields and set properties of this semantic annotation.
    Skips private fields (those starting with '_'), as well as the 'bodies' property.

    :param semantic_annotation: The semantic annotation to extract attributes from.
    """
    for f in fields(semantic_annotation):
        if f.name.startswith("_"):
            continue
        v = getattr(semantic_annotation, f.name, None)
        if _is_entity_semantic_annotation_or_iterable(v, aggregation_type):
            yield v

    for name, prop in inspect.getmembers(
        type(semantic_annotation), lambda o: isinstance(o, property)
    ):
        if name in {
            "kinematic_structure_entities",
            "bodies",
            "regions",
        } or name.startswith("_"):
            continue
        try:
            v = getattr(semantic_annotation, name)
        except Exception:
            continue
        if _is_entity_semantic_annotation_or_iterable(v, aggregation_type):
            yield v
