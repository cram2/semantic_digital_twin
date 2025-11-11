import inspect
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from types import NoneType
from typing import (
    Dict,
    List,
    Any,
    ClassVar,
    Type,
    Optional,
    Union,
)
from typing_extensions import Self

import numpy
from mujoco_connector import MultiverseMujocoConnector
import mujoco
from multiverse_simulator import (
    MultiverseSimulator,
    MultiverseSimulatorState,
    MultiverseViewer,
    MultiverseAttribute,
    MultiverseCallbackResult,
)
from krrood.utils import recursive_subclasses
from scipy.spatial.transform import Rotation

from ..callbacks.callback import ModelChangeCallback
from ..datastructures.prefixed_name import PrefixedName
from ..spatial_types.spatial_types import TransformationMatrix, Point3, Quaternion
from ..world import World
from ..world_description.connections import (
    RevoluteConnection,
    PrismaticConnection,
    ActiveConnection1DOF,
    FixedConnection,
)
from ..world_description.geometry import Box, Cylinder, Sphere, Shape
from ..world_description.world_entity import (
    Region,
    Body,
    KinematicStructureEntity,
    Connection,
    WorldEntity,
)
from ..world_description.world_modification import (
    AddKinematicStructureEntityModification,
)


def cas_pose_to_list(pose: TransformationMatrix) -> List[float]:
    """
    Converts a CAS TransformationMatrix to a list of 7 floats (position + quaternion).

    :param pose: The CAS TransformationMatrix to convert.
    :return: A list of 7 floats ([px, py, pz, qw, qx, qy, qz]) representing the position and quaternion.
    """
    pos = pose.to_position()
    quat = pose.to_quaternion()
    px, py, pz, _ = pos.evaluate().tolist()
    qx, qy, qz, qw = quat.evaluate().tolist()
    return [px, py, pz, qw, qx, qy, qz]


@dataclass
class InertialConverter:
    """
    A converter to convert inertia representations to diagonal form and update the inertia quaternion accordingly.
    """

    mass: float
    """
    The mass of the body.
    """

    inertia_pos: Point3
    """
    The position of the inertia frame relative to the body frame [x, y, z].
    """

    inertia_quat: Quaternion
    """
    The orientation of the inertia frame relative to the body frame as a quaternion [qw, qx, qy, qz].
    """

    diagonal_inertia: List[float]
    """
    The diagonal inertia tensor in the form [Ixx, Iyy, Izz].
    """

    def __post_init__(self):
        assert self.mass > 0, "Mass must be positive."
        assert len(self.diagonal_inertia) == 3, "Diagonal inertia must have 3 elements."
        assert all(
            i >= 0 for i in self.diagonal_inertia
        ), "Inertia values must be non-negative."

    @classmethod
    def from_inertia(
        cls,
        mass: float,
        inertia_pos: Point3,
        inertia_quat: Quaternion,
        inertia: List[float],
    ) -> Self:
        """
        Converts inertia given as [Ixx, Iyy, Izz, Ixy, Ixz, Iyz] to diagonal form and updates the inertia quaternion.

        :param mass: The mass of the body.
        :param inertia_pos: The position of the inertia frame relative to the body frame [x, y, z].
        :param inertia_quat: The orientation of the inertia frame relative to the body frame as a quaternion [qw, qx, qy, qz].
        :param inertia: The inertia tensor in the form [Ixx, Iyy, Izz, Ixy, Ixz, Iyz].

        :return: An InertialConverter instance with diagonal inertia and updated quaternion.
        """
        inertia_matrix = [
            inertia[0],
            inertia[3],
            inertia[4],
            inertia[3],
            inertia[1],
            inertia[5],
            inertia[4],
            inertia[5],
            inertia[2],
        ]
        return cls.from_inertia_matrix(
            mass=mass,
            inertia_pos=inertia_pos,
            inertia_quat=inertia_quat,
            inertia_matrix=inertia_matrix,
        )

    @classmethod
    def from_inertia_matrix(
        cls,
        mass: float,
        inertia_pos: Point3,
        inertia_quat: Quaternion,
        inertia_matrix: List[float],
    ) -> Self:
        """
        Converts inertia given as a 3x3 matrix in row-major order to diagonal form and updates the inertia quaternion.

        :param mass: The mass of the body.
        :param inertia_pos: The position of the inertia frame relative to the body frame [x, y, z].
        :param inertia_quat: The orientation of the inertia frame relative to the body frame as a quaternion [qw, qx, qy, qz].
        :param inertia_matrix: The inertia tensor as a 3x3 matrix in row-major order [Ixx, Ixy, Ixz, Iyx, Iyy, Iyz, Izx, Izy, Izz].

        :return: An InertialConverter instance with diagonal inertia and updated quaternion.
        """
        return cls._convert_inertia(
            mass=mass,
            inertia_pos=inertia_pos,
            inertia_quat=inertia_quat,
            inertia_matrix=numpy.array(inertia_matrix).reshape(3, 3),
        )

    @classmethod
    def _convert_inertia(cls, mass, inertia_pos, inertia_quat, inertia_matrix) -> Self:
        """
        Diagonalizes the inertia matrix and updates the inertia quaternion.

        :param mass: The mass of the body.
        :param inertia_pos: The position of the inertia frame relative to the body frame [x, y, z].
        :param inertia_quat: The orientation of the inertia frame relative to the body frame as a quaternion [qw, qx, qy, qz].
        :param inertia_matrix: The inertia tensor as a 3x3 numpy array in row-major order [Ixx, Ixy, Ixz; Iyx, Iyy, Iyz; Izx, Izy, Izz].

        :return: An InertialConverter instance with diagonal inertia and updated quaternion.
        """
        eigenvalues, eigenvectors = numpy.linalg.eigh(inertia_matrix)
        eigenvalues, eigenvectors = cls._sort_and_adjust(eigenvalues, eigenvectors)
        inertia_quat = cls._update_quaternion(inertia_quat, eigenvectors)
        diagonal_inertia = eigenvalues.tolist()
        return cls(
            mass=mass,
            inertia_pos=inertia_pos,
            inertia_quat=inertia_quat,
            diagonal_inertia=diagonal_inertia,
        )

    @staticmethod
    def _sort_and_adjust(eigenvalues: numpy.ndarray, eigenvectors: numpy.ndarray):
        """
        Sorts eigenvalues and eigenvectors, and ensures a right-handed coordinate system.

        :param eigenvalues: The eigenvalues of the inertia matrix.
        :param eigenvectors: The eigenvectors of the inertia matrix.

        :return: Sorted eigenvalues and adjusted eigenvectors.
        """
        idx = numpy.argsort(eigenvalues)
        eigenvalues, eigenvectors = eigenvalues[idx], eigenvectors[:, idx]
        if numpy.linalg.det(eigenvectors) < 0:
            eigenvectors[:, 0] *= -1
        return eigenvalues, eigenvectors

    @staticmethod
    def _update_quaternion(
        quat: numpy.ndarray, eigenvectors: numpy.ndarray
    ) -> Quaternion:
        """
        Updates the inertia quaternion based on the eigenvectors of the inertia matrix.

        :param quat: The original inertia quaternion [qw, qx, qy, qz].
        :param eigenvectors: The eigenvectors of the inertia matrix.

        :return: The updated inertia quaternion [qw, qx, qy, qz].
        """
        R_orig = Rotation.from_quat(quat, scalar_first=True)  # type: ignore
        R_diag = Rotation.from_matrix(eigenvectors)  # type: ignore
        updated_quat = (R_orig * R_diag).as_quat(scalar_first=True)
        return Quaternion(
            x_init=updated_quat[1],
            y_init=updated_quat[2],
            z_init=updated_quat[3],
            w_init=updated_quat[0],
        )


class EntityConverter(ABC):
    """
    A converter to convert an entity object (WorldEntity, Shape, Connection) to a dictionary of properties for Multiverse simulator.
    """

    entity_type: ClassVar[Type[Any]] = Any
    """
    The type of the entity to convert.
    """

    name_str: str = "name"
    """
    The key for the name property in the output dictionary.
    """

    @classmethod
    def convert(cls, entity: entity_type, **kwargs) -> Dict[str, Any]:  # type: ignore
        """
        Converts an entity object to a dictionary of properties for Multiverse simulator.

        :param entity: The object to convert.
        :return: A dictionary of properties.
        """
        for subclass in recursive_subclasses(cls):
            if (
                not inspect.isabstract(subclass)
                and not inspect.isabstract(subclass.entity_type)
                and type(entity) is subclass.entity_type
            ):
                entity_props = subclass()._convert(entity, **kwargs)
                return subclass()._post_convert(entity, entity_props, **kwargs)
        raise NotImplementedError(f"No converter found for entity type {type(entity)}.")

    def _convert(self, entity: entity_type, **kwargs) -> Dict[str, Any]:  # type: ignore
        """
        The actual conversion method to be implemented by subclasses.

        :param entity: The object to convert.
        :return: A dictionary of properties, by default containing the name.
        """
        return {
            self.name_str: (
                entity.name.name
                if hasattr(entity, "name") and isinstance(entity.name, PrefixedName)
                else f"{type(entity).__name__.lower()}_{id(entity)}"
            )
        }

    @abstractmethod
    def _post_convert(
        self, entity: entity_type, entity_props: Dict[str, Any], **kwargs  # type: ignore
    ) -> Dict[str, Any]:
        """
        Post-processes the converted entity properties. This method can be overridden by subclasses to update the properties after conversion.

        :param entity: The object that was converted.
        :param entity_props: The dictionary of properties that was converted.
        :return: The updated dictionary of properties.
        """
        raise NotImplementedError


class KinematicStructureEntityConverter(EntityConverter, ABC):
    """
    Converts a KinematicStructureEntity object to a dictionary of body properties for Multiverse simulator.
    For inheriting classes, the following string attributes must be defined:
    - pos_str: The key for the position property in the output dictionary.
    - quat_str: The key for the quaternion property in the output dictionary.
    """

    entity_type: ClassVar[Type[KinematicStructureEntity]] = KinematicStructureEntity
    pos_str: str
    quat_str: str

    def _convert(self, entity: entity_type, **kwargs) -> Dict[str, Any]:
        """
        Converts a KinematicStructureEntity object to a dictionary of body properties for Multiverse simulator.

        :param entity: The KinematicStructureEntity object to convert.
        :return: A dictionary of body properties, by default containing position and quaternion.
        """

        kinematic_structure_entity_props = EntityConverter._convert(self, entity)
        px, py, pz, qw, qx, qy, qz = cas_pose_to_list(
            entity.parent_connection.origin_expression
        )
        kinematic_structure_entity_pos = [px, py, pz]
        kinematic_structure_entity_quat = [qw, qx, qy, qz]
        kinematic_structure_entity_props.update(
            {
                self.pos_str: kinematic_structure_entity_pos,
                self.quat_str: kinematic_structure_entity_quat,
            }
        )
        return kinematic_structure_entity_props


class BodyConverter(KinematicStructureEntityConverter, ABC):
    """
    Converts a Body object to a dictionary of body properties for Multiverse simulator.
    """

    entity_type: ClassVar[Type[WorldEntity]] = Body
    """
    The type of the entity to convert.
    """

    # Attributes for specifying body properties in the Mujoco simulator.

    mass_str: str
    """
    The key for the mass property in the output dictionary.
    """

    inertia_pos_str: str
    """
    The key for the inertia position property in the output dictionary.
    """

    inertia_quat_str: str
    """
    The key for the inertia quaternion property in the output dictionary.
    """

    diagonal_inertia_str: str
    """
    The key for the diagonal inertia tensor property in the output dictionary.
    """

    def _convert(self, entity: Body, **kwargs) -> Dict[str, Any]:
        """
        Converts a Body object to a dictionary of body properties for Multiverse simulator.

        :param entity: The Body object to convert.
        :return: A dictionary of body properties, including additional mass and inertia properties.
        """
        body_props = KinematicStructureEntityConverter._convert(self, entity)
        mass = 1e-3  # TODO: Take from entity
        inertia_pos = Point3(
            x_init=0.0, y_init=0.0, z_init=0.0
        )  # TODO: Take from entity
        inertia_quat = Quaternion(
            w_init=1.0, x_init=0.0, y_init=0.0, z_init=0.0
        )  # TODO: Take from entity
        diagonal_inertia = [1.5e-8, 1.5e-8, 1.5e-8]  # TODO: Take from entity
        if diagonal_inertia is None:
            inertia = body_props.get("inertia", None)
            inertia_matrix = body_props.get("inertia_matrix", None)
            if isinstance(inertia, list) and len(inertia) == 6:
                inertial_converter = InertialConverter.from_inertia(
                    mass=mass,
                    inertia_pos=inertia_pos,
                    inertia_quat=inertia_quat,
                    inertia=inertia,
                )
            elif isinstance(inertia_matrix, list) and len(inertia_matrix) == 9:
                inertial_converter = InertialConverter.from_inertia_matrix(
                    mass=mass,
                    inertia_pos=inertia_pos,
                    inertia_quat=inertia_quat,
                    inertia_matrix=inertia_matrix,
                )
            else:
                raise ValueError(
                    f"Body {entity.name.name} must have either 'diagonal_inertia' (3 elements), 'inertia' (6 elements) or 'inertia_matrix' (9 elements)."
                )
        else:
            inertial_converter = InertialConverter(
                mass=mass,
                inertia_pos=inertia_pos,
                inertia_quat=inertia_quat,
                diagonal_inertia=diagonal_inertia,
            )
        mass = inertial_converter.mass
        inertia_pos = inertial_converter.inertia_pos.to_np().tolist()[:3]
        inertia_quat = inertial_converter.inertia_quat.to_np().tolist()[:4]
        diagonal_inertia = inertial_converter.diagonal_inertia
        body_props.update(
            {
                self.mass_str: mass,
                self.inertia_pos_str: inertia_pos,
                self.inertia_quat_str: inertia_quat,
                self.diagonal_inertia_str: diagonal_inertia,
            }
        )
        return body_props


class RegionConverter(KinematicStructureEntityConverter, ABC):
    """
    Converts a Region object to a dictionary of region properties for Multiverse simulator.
    """

    entity_type: ClassVar[Type[WorldEntity]] = Region
    """
    The type of the entity to convert.
    """


class ShapeConverter(EntityConverter, ABC):
    """
    Converts a Shape object to a dictionary of shape properties for Multiverse simulator.
    """

    entity_type: ClassVar[Type[Shape]] = Shape
    """
    The type of the entity to convert.
    """

    pos_str: str
    """
    The key for the shape position property in the output dictionary.
    """

    quat_str: str
    """
    The key for the shape quaternion property in the output dictionary.
    """

    rgba_str: str
    """
    The key for the shape RGBA color property in the output dictionary.
    """

    def _convert(self, entity: Shape, **kwargs) -> Dict[str, Any]:
        """
        Converts a Shape object to a dictionary of shape properties for Multiverse simulator.

        :param entity: The Shape object to convert.
        :return: A dictionary of shape properties, by default containing position, quaternion, and RGBA color.
        """
        geom_props = EntityConverter._convert(self, entity)
        px, py, pz, qw, qx, qy, qz = cas_pose_to_list(entity.origin)
        geom_pos = [px, py, pz]
        geom_quat = [qw, qx, qy, qz]
        r, g, b, a = (
            entity.color.R,
            entity.color.G,
            entity.color.B,
            entity.color.A,
        )
        geom_color = [r, g, b, a]
        geom_props.update(
            {
                self.pos_str: geom_pos,
                self.quat_str: geom_quat,
                self.rgba_str: geom_color,
            }
        )
        return geom_props


class BoxConverter(ShapeConverter, ABC):
    """
    Converts a Box object to a dictionary of box properties for Multiverse simulator.
    """

    entity_type: ClassVar[Type[Box]] = Box


class SphereConverter(ShapeConverter, ABC):
    """
    Converts a Sphere object to a dictionary of sphere properties for Multiverse simulator.
    """

    entity_type: ClassVar[Type[Sphere]] = Sphere


class CylinderConverter(ShapeConverter, ABC):
    """
    Converts a Cylinder object to a dictionary of cylinder properties for Multiverse simulator.
    """

    entity_type: ClassVar[Type[Cylinder]] = Cylinder


class ConnectionConverter(EntityConverter, ABC):
    """
    Converts a Connection object to a dictionary of joint properties for Multiverse simulator.
    """

    entity_type: ClassVar[Type[Connection]] = Connection
    """
    The type of the entity to convert.
    """

    pos_str: str
    """
    The key for the joint position property in the output dictionary.
    """

    quat_str: str
    """
    The key for the joint quaternion property in the output dictionary.
    """

    def _convert(self, entity: Connection, **kwargs) -> Dict[str, Any]:
        """
        Converts a Connection object to a dictionary of joint properties for Multiverse simulator.

        :param entity: The Connection object to convert.
        :return: A dictionary of joint properties, by default containing position and quaternion.
        """
        joint_props = EntityConverter._convert(self, entity)
        px, py, pz, qw, qx, qy, qz = cas_pose_to_list(entity.origin)
        joint_pos = [px, py, pz]
        joint_quat = [qw, qx, qy, qz]
        joint_props.update(
            {
                self.pos_str: joint_pos,
                self.quat_str: joint_quat,
            }
        )
        return joint_props


class Connection1DOFConverter(ConnectionConverter, ABC):
    """
    Converts an ActiveConnection1DOF object to a dictionary of joint properties for Multiverse simulator.
    """

    entity_type: ClassVar[Type[ActiveConnection1DOF]] = ActiveConnection1DOF
    """
    The type of the entity to convert.
    """

    axis_str: str
    """
    The key for the joint axis property in the output dictionary.
    """

    range_str: str
    """
    The key for the joint range property in the output dictionary.
    """

    def _convert(self, entity: ActiveConnection1DOF, **kwargs) -> Dict[str, Any]:
        """
        Converts an ActiveConnection1DOF object to a dictionary of joint properties for Multiverse simulator.

        :param entity: The ActiveConnection1DOF object to convert.
        :return: A dictionary of joint properties, including additional axis and range properties.
        """
        joint_props = ConnectionConverter._convert(self, entity)
        assert len(entity.dofs) == 1, "ActiveConnection1DOF must have exactly one DOF."
        dof = list(entity.dofs)[0]
        joint_props.update(
            {
                self.axis_str: entity.axis.to_np().tolist()[:3],
                self.range_str: [dof.lower_limits.position, dof.upper_limits.position],
            }
        )
        return joint_props


class ConnectionRevoluteConverter(Connection1DOFConverter, ABC):
    """
    Converts a RevoluteConnection object to a dictionary of revolute joint properties for Multiverse simulator.
    """

    entity_type: ClassVar[Type[RevoluteConnection]] = RevoluteConnection
    """
    The type of the entity to convert.
    """


class ConnectionPrismaticConverter(Connection1DOFConverter, ABC):
    """
    Converts a PrismaticConnection object to a dictionary of prismatic joint properties for Multiverse simulator.
    """

    entity_type: ClassVar[Type[PrismaticConnection]] = PrismaticConnection
    """
    The type of the entity to convert.
    """


class MujocoConverter(EntityConverter, ABC): ...


class MujocoKinematicStructureEntityConverter(
    MujocoConverter, KinematicStructureEntityConverter, ABC
):
    pos_str: str = "pos"
    quat_str: str = "quat"


class MujocoBodyConverter(MujocoKinematicStructureEntityConverter, BodyConverter):
    mass_str: str = "mass"
    inertia_pos_str: str = "ipos"
    inertia_quat_str: str = "iquat"
    diagonal_inertia_str: str = "inertia"

    def _post_convert(
        self, entity: Body, body_props: Dict[str, Any], **kwargs
    ) -> Dict[str, Any]:
        return body_props


class MujocoRegionConverter(MujocoKinematicStructureEntityConverter, RegionConverter):
    def _post_convert(
        self, entity: Region, region_props: Dict[str, Any], **kwargs
    ) -> Dict[str, Any]:
        return region_props


class MujocoGeomConverter(MujocoConverter, ShapeConverter, ABC):
    pos_str: str = "pos"
    quat_str: str = "quat"
    rgba_str: str = "rgba"
    type: mujoco.mjtGeom

    def _post_convert(
        self, entity: Shape, shape_props: Dict[str, Any], **kwargs
    ) -> Dict[str, Any]:
        shape_props.update(
            {
                "type": self.type,
            }
        )
        if not kwargs.get("visible", True):
            shape_props[self.rgba_str][3] = 0.0
        if not kwargs.get("collidable", True):
            shape_props["contype"] = 0
            shape_props["conaffinity"] = 0
        return shape_props


class MujocoBoxConverter(MujocoGeomConverter, BoxConverter):
    type: mujoco.mjtGeom = mujoco.mjtGeom.mjGEOM_BOX

    def _post_convert(
        self, entity: Box, shape_props: Dict[str, Any], **kwargs
    ) -> Dict[str, Any]:
        shape_props.update(MujocoGeomConverter._post_convert(self, entity, shape_props))
        shape_props.update(
            {"size": [entity.scale.x / 2, entity.scale.y / 2, entity.scale.z / 2]}
        )
        return shape_props


class MujocoSphereConverter(MujocoGeomConverter, SphereConverter):
    type: mujoco.mjtGeom = mujoco.mjtGeom.mjGEOM_SPHERE

    def _post_convert(
        self, entity: Sphere, shape_props: Dict[str, Any], **kwargs
    ) -> Dict[str, Any]:
        shape_props.update(MujocoGeomConverter._post_convert(self, entity, shape_props))
        shape_props.update({"size": [entity.radius, entity.radius, entity.radius]})
        return shape_props


class MujocoCylinderConverter(MujocoGeomConverter, CylinderConverter):
    type: mujoco.mjtGeom = mujoco.mjtGeom.mjGEOM_CYLINDER

    def _post_convert(
        self, entity: Cylinder, shape_props: Dict[str, Any], **kwargs
    ) -> Dict[str, Any]:
        shape_props.update(MujocoGeomConverter._post_convert(self, entity, shape_props))
        shape_props.update({"size": [entity.width / 2, entity.height, 0.0]})
        return shape_props


class MujocoJointConverter(ConnectionConverter, ABC):
    pos_str: str = "pos"
    quat_str: str = "quat"
    type: mujoco.mjtJoint

    def _post_convert(
        self, entity: Connection, joint_props: Dict[str, Any], **kwargs
    ) -> Dict[str, Any]:
        joint_props["type"] = self.type
        return joint_props


class Mujoco1DOFJointConverter(MujocoJointConverter, Connection1DOFConverter):
    axis_str: str = "axis"
    range_str: str = "range"

    def _post_convert(
        self, entity: ActiveConnection1DOF, joint_props: Dict[str, Any], **kwargs
    ) -> Dict[str, Any]:
        joint_props = MujocoJointConverter._post_convert(self, entity, joint_props)
        if not numpy.allclose(joint_props["quat"], [1.0, 0.0, 0.0, 0.0]):
            joint_axis = numpy.array(joint_props["axis"])
            R_joint = Rotation.from_quat(quat=joint_props["quat"], scalar_first=True)  # type: ignore
            joint_props["axis"] = R_joint.apply(joint_axis).tolist()
        del joint_props["quat"]
        return joint_props


class MujocoRevoluteJointConverter(
    Mujoco1DOFJointConverter, ConnectionRevoluteConverter
):
    type: mujoco.mjtJoint = mujoco.mjtJoint.mjJNT_HINGE


class MujocoPrismaticJointConverter(
    Mujoco1DOFJointConverter, ConnectionPrismaticConverter
):
    type: mujoco.mjtJoint = mujoco.mjtJoint.mjJNT_SLIDE


@dataclass
class MultiSimBuilder(ABC):
    """
    A builder to build a world in the Multiverse simulator.
    """

    def build_world(self, world: World, file_path: str):
        """
        Builds the world in the simulator and saves it to a file.

        :param world: The world to build.
        :param file_path: The file path to save the world to.
        """
        if len(world.bodies) == 0:
            with world.modify_world():
                root = Body(name=PrefixedName("world"))
                world.add_body(root)
        elif world.bodies[0].name.name != "world":
            with world.modify_world():
                root_bodies = [
                    body for body in world.bodies if body.parent_connection is None
                ]
                root = Body(name=PrefixedName("world"))
                world.add_body(root)
                for root_body in root_bodies:
                    connection = FixedConnection(parent=root, child=root_body)
                    world.add_connection(connection)

        self._start_build(file_path=file_path)

        for body in world.bodies:
            self.build_body(body=body)

        for region in world.regions:
            self.build_region(region=region)

        for connection in world.connections:
            self._build_connection(connection=connection)

        self._end_build(file_path=file_path)

    def build_body(self, body: Body):
        """
        Builds a body in the simulator including its shapes.

        :param body: The body to build.
        """
        self._build_body(body=body)
        for shape in {
            id(s): s for s in body.visual.shapes + body.collision.shapes
        }.values():
            self._build_shape(
                parent=body,
                shape=shape,
                visible=shape in body.visual or not body.visual,
                collidable=shape in body.collision,
            )

    def build_region(self, region: Region):
        """
        Builds a region in the simulator including its shapes.

        :param region: The region to build.
        """
        self._build_region(region=region)
        for shape in region.area:
            self._build_shape(
                parent=region, shape=shape, visible=True, collidable=False
            )

    @abstractmethod
    def _start_build(self, file_path: str):
        """
        Starts the building process for the simulator.

        :param file_path: The file path to save the world to.
        """
        raise NotImplementedError

    @abstractmethod
    def _end_build(self, file_path: str):
        """
        Ends the building process for the simulator and saves the world to a file.

        :param file_path: The file path to save the world to.
        """
        raise NotImplementedError

    @abstractmethod
    def _build_body(self, body: Body):
        """
        Builds a body in the simulator.

        :param body: The body to build.
        """
        raise NotImplementedError

    @abstractmethod
    def _build_region(self, region: Region):
        """
        Builds a region in the simulator.

        :param region: The region to build.
        """
        raise NotImplementedError

    @abstractmethod
    def _build_shape(
        self, parent: Union[Body, Region], shape: Shape, visible: bool, collidable: bool
    ):
        """
        Builds a shape in the simulator and attaches it to its parent body or region.

        :param parent: The parent body or region to attach the shape to.
        :param shape: The shape to build.
        :param visible: Whether the shape is visible.
        :param collidable: Whether the shape is collidable.
        """
        raise NotImplementedError

    @abstractmethod
    def _build_connection(self, connection: Connection):
        """
        Builds a connection in the simulator.

        :param connection: The connection to build.
        """
        raise NotImplementedError


@dataclass
class MujocoBuilder(MultiSimBuilder):
    """
    A builder to build a world in the Mujoco simulator.
    """

    spec: mujoco.MjSpec = field(default=mujoco.MjSpec())

    def _start_build(self, file_path: str):
        self.spec = mujoco.MjSpec()
        self.spec.modelname = "scene"

    def _end_build(self, file_path: str):
        self.spec.compile()
        self.spec.to_file(file_path)
        try:
            mujoco.MjModel.from_xml_path(file_path)
        except ValueError as e:
            if (
                "Error: mass and inertia of moving bodies must be larger than mjMINVAL"
                in str(e)
            ):  # Fix mujoco error
                import xml.etree.ElementTree as ET

                tree = ET.parse(file_path)
                root = tree.getroot()
                for body_id, body_element in enumerate(root.findall(".//body")):
                    body_spec = self.spec.bodies[body_id + 1]
                    inertial_element = ET.SubElement(body_element, "inertial")
                    inertial_element.set("mass", f"{body_spec.mass}")
                    inertial_element.set(
                        "diaginertia", " ".join(map(str, body_spec.inertia.tolist()))
                    )
                    inertial_element.set(
                        "pos", " ".join(map(str, body_spec.ipos.tolist()))
                    )
                    inertial_element.set(
                        "quat", " ".join(map(str, body_spec.iquat.tolist()))
                    )
                tree.write(file_path)
            else:
                raise e

    def _build_body(self, body: Body):
        self._build_mujoco_body(body=body)

    def _build_region(self, region: Region):
        self._build_mujoco_body(body=region)

    def _build_shape(
        self, parent: Union[Body, Region], shape: Shape, visible: bool, collidable: bool
    ):
        geom_props = MujocoGeomConverter.convert(
            shape, visible=visible, collidable=collidable
        )
        assert geom_props is not None, f"Failed to convert shape {id(shape)}."
        parent_body_name = parent.name.name
        parent_body_spec = self._find_entity(
            entity_type=mujoco.mjtObj.mjOBJ_BODY, entity_name=parent_body_name
        )
        assert (
            parent_body_spec is not None
        ), f"Parent body {parent_body_name} not found."
        geom_spec = parent_body_spec.add_geom(**geom_props)
        assert (
            geom_spec is not None
        ), f"Failed to add geom {id(shape)} to body {parent_body_name}."

    def _build_connection(self, connection: Connection):
        if isinstance(connection, FixedConnection):
            return
        joint_props = MujocoJointConverter.convert(connection)
        assert (
            joint_props is not None
        ), f"Failed to convert connection {connection.name.name}."
        child_body_name = connection.child.name.name
        child_body_spec = self._find_entity(
            entity_type=mujoco.mjtObj.mjOBJ_BODY, entity_name=child_body_name
        )
        assert child_body_spec is not None, f"Child body {child_body_name} not found."
        joint_name = connection.name.name
        joint_spec = child_body_spec.add_joint(**joint_props)
        assert (
            joint_spec is not None
        ), f"Failed to add joint {joint_name} to body {child_body_name}."

    def _build_mujoco_body(self, body: Union[Region, Body]):
        """
        Builds a body in the Mujoco spec. In Mujoco, regions are also represented as bodies.

        :param body: The body or region to build.
        """
        if body.name.name == "world":
            return
        body_props = MujocoKinematicStructureEntityConverter.convert(body)
        assert body_props is not None, f"Failed to convert body {body.name.name}."
        parent_body_name = body.parent_connection.parent.name.name
        parent_body_spec = self._find_entity(
            entity_type=mujoco.mjtObj.mjOBJ_BODY, entity_name=parent_body_name
        )
        assert (
            parent_body_spec is not None
        ), f"Parent body {parent_body_name} not found."
        body_spec = parent_body_spec.add_body(**body_props)
        assert (
            body_spec is not None
        ), f"Failed to add body {body.name.name} to parent {parent_body_name}."

    def _find_entity(
        self,
        entity_type: mujoco.mjtObj,
        entity_name: str,
    ) -> Optional[
        Union[mujoco.MjsBody, mujoco.MjsGeom, mujoco.MjsJoint, mujoco.MjsSite]
    ]:
        """
        Finds an entity in the Mujoco spec by its type and name.

        :param entity_type: The type of the entity
        :param entity_name: The name of the entity.
        :return: The entity if found, None otherwise.
        """
        entity_type_str = entity_type.name.replace("mjOBJ_", "").lower()
        if mujoco.mj_version() >= 330:
            return self.spec.__getattribute__(entity_type_str)(entity_name)
        else:
            return self.spec.__getattribute__(f"find_{entity_type_str}")(entity_name)


class EntitySpawner(ABC):
    """
    A spawner to spawn an entity object (WorldEntity, Shape, Connection) in the Multiverse simulator.
    """

    entity_type: ClassVar[Type[Any]] = Any
    """
    The type of the entity to spawn.
    """

    @classmethod
    def spawn(cls, simulator: MultiverseSimulator, entity: entity_type) -> bool:  # type: ignore
        """
        Spawns a WorldEntity object in the Multiverse simulator.

        :param simulator: The Multiverse simulator to spawn the entity in.
        :param entity: The WorldEntity object to spawn.

        :return: True if the entity was spawned successfully, False otherwise.
        """
        for subclass in recursive_subclasses(cls):
            if (
                not inspect.isabstract(subclass)
                and not inspect.isabstract(subclass.entity_type)
                and type(entity) is subclass.entity_type
            ):
                return subclass()._spawn(simulator, entity)
        raise NotImplementedError(f"No converter found for entity type {type(entity)}.")

    @abstractmethod
    def _spawn(self, simulator: MultiverseSimulator, entity: Any) -> bool:
        """
        The actual spawning method to be implemented by subclasses.

        :param simulator: The Multiverse simulator to spawn the entity in.
        :param entity: The WorldEntity object to spawn.
        :return: True if the entity was spawned successfully, False otherwise.
        """
        raise NotImplementedError


class KinematicStructureEntitySpawner(EntitySpawner):
    entity_type: ClassVar[Type[KinematicStructureEntity]] = KinematicStructureEntity
    """
    The type of the entity to spawn.
    """

    def _spawn(
        self, simulator: MultiverseSimulator, entity: KinematicStructureEntity
    ) -> bool:
        """
        Spawns a KinematicStructureEntity object in the Multiverse simulator including its shapes.

        :param simulator: The Multiverse simulator to spawn the entity in.
        :param entity: The KinematicStructureEntity object to spawn.

        :return: True if the entity and its shapes were spawned successfully, False otherwise.
        """
        return self._spawn_kinematic_structure_entity(
            simulator, entity
        ) and self._spawn_shapes(simulator, entity)

    @abstractmethod
    def _spawn_kinematic_structure_entity(
        self, simulator: MultiverseSimulator, entity: KinematicStructureEntity
    ) -> bool:
        """
        Spawns a KinematicStructureEntity object in the Multiverse simulator.

        :param simulator: The Multiverse simulator to spawn the entity in.
        :param entity: The KinematicStructureEntity object to spawn.

        :return: True if the entity was spawned successfully, False otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    def _spawn_shapes(
        self, simulator: MultiverseSimulator, entity: KinematicStructureEntity
    ) -> bool:
        """
        Spawns the shapes of a KinematicStructureEntity object in the Multiverse simulator.

        :param simulator: The Multiverse simulator to spawn the shapes in.
        :param entity: The KinematicStructureEntity object whose shapes to spawn.

        :return: True if all shapes were spawned successfully, False otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    def _spawn_shape(
        self,
        parent: Union[Body, Region],
        simulator: MultiverseSimulator,
        shape: Shape,
        visible: bool,
        collidable: bool,
    ) -> bool:
        """
        Spawns a shape in the Multiverse simulator and attaches it to its parent body or region.

        :param parent: The parent body or region to attach the shape to.
        :param simulator: The Multiverse simulator to spawn the shape in.
        :param shape: The shape to spawn.
        :param visible: Whether the shape is visible.
        :param collidable: Whether the shape is collidable.

        :return: True if the shape was spawned successfully, False otherwise.
        """
        raise NotImplementedError


class BodySpawner(KinematicStructureEntitySpawner, ABC):
    entity_type: ClassVar[Type[Body]] = Body
    """
    The type of the entity to spawn.
    """

    def _spawn_shapes(self, simulator: MultiverseSimulator, parent: Body) -> bool:
        return all(
            self._spawn_shape(
                parent=parent,
                simulator=simulator,
                shape=shape,
                visible=shape in parent.visual or not parent.visual,
                collidable=shape in parent.collision,
            )
            for shape in {
                id(s): s for s in parent.visual.shapes + parent.collision.shapes
            }.values()
        )


class RegionSpawner(KinematicStructureEntitySpawner, ABC):
    entity_type: ClassVar[Type[Region]] = Region
    """
    The type of the entity to spawn.
    """

    def _spawn_shapes(self, simulator: MultiverseSimulator, parent: Region) -> bool:
        return all(
            self._spawn_shape(
                parent=parent,
                simulator=simulator,
                shape=shape,
                visible=True,
                collidable=False,
            )
            for shape in parent.area
        )


class MujocoEntitySpawner(EntitySpawner, ABC): ...


class MujocoKinematicStructureEntitySpawner(
    MujocoEntitySpawner, KinematicStructureEntitySpawner, ABC
):
    def _spawn_kinematic_structure_entity(
        self, simulator: MultiverseMujocoConnector, entity: KinematicStructureEntity
    ) -> bool:
        kinematic_structure_entity_props = (
            MujocoKinematicStructureEntityConverter.convert(entity)
        )
        assert (
            kinematic_structure_entity_props is not None
        ), f"Failed to convert entity {entity.name.name}."
        entity_name = kinematic_structure_entity_props["name"]
        del kinematic_structure_entity_props["name"]
        result = simulator.add_entity(
            entity_name=entity_name,
            entity_type="body",
            entity_properties=kinematic_structure_entity_props,
            parent_name=entity.parent_connection.parent.name.name,
        )
        return (
            result.type
            == MultiverseCallbackResult.ResultType.SUCCESS_AFTER_EXECUTION_ON_MODEL
        )

    def _spawn_shape(
        self,
        parent: Body,
        simulator: MultiverseMujocoConnector,
        shape: Shape,
        visible: bool,
        collidable: bool,
    ) -> bool:
        shape_props = MujocoGeomConverter.convert(
            shape, visible=visible, collidable=collidable
        )
        assert shape_props is not None, f"Failed to convert shape {id(shape)}."
        shape_name = shape_props["name"]
        del shape_props["name"]
        result = simulator.add_entity(
            entity_name=shape_name,
            entity_type="geom",
            entity_properties=shape_props,
            parent_name=parent.name.name,
        )
        return (
            result.type
            == MultiverseCallbackResult.ResultType.SUCCESS_AFTER_EXECUTION_ON_MODEL
        )


class MujocoBodySpawner(MujocoKinematicStructureEntitySpawner, BodySpawner): ...


class MujocoRegionSpawner(MujocoKinematicStructureEntitySpawner, RegionSpawner): ...


@dataclass
class MultiSimSynchronizer(ModelChangeCallback, ABC):
    """
    A callback to synchronize the world model with the Multiverse simulator.
    This callback will listen to the world model changes and update the Multiverse simulator accordingly.
    """

    world: World
    """
    The world to synchronize with the simulator.
    """

    simulator: MultiverseSimulator
    """
    The Multiverse simulator to synchronize with the world.
    """

    entity_converter: Type[EntityConverter] = NoneType
    """
    The converter to convert WorldEntity, Shape, and Connection objects to dictionaries of properties for the simulator.
    """

    entity_spawner: Type[EntitySpawner] = NoneType
    """
    The spawner to spawn WorldEntity, Shape, and Connection objects in the simulator.
    """

    def _notify(self):
        for modification in self.world._model_manager.model_modification_blocks[-1]:
            if isinstance(modification, AddKinematicStructureEntityModification):
                entity = modification.kinematic_structure_entity
                self.entity_spawner.spawn(simulator=self.simulator, entity=entity)

    def stop(self):
        self.world._model_manager.model_change_callbacks.remove(self)


@dataclass
class MujocoSynchronizer(MultiSimSynchronizer):
    simulator: MultiverseMujocoConnector
    entity_converter: Type[EntityConverter] = field(default=MujocoConverter)
    entity_spawner: Type[EntitySpawner] = field(default=MujocoEntitySpawner)


class MultiSim(ABC):
    """
    Class to handle the simulation of a world using the Multiverse simulator.
    """

    simulator_class: ClassVar[Type[MultiverseSimulator]]
    """
    The class of the Multiverse simulator to use.
    """

    synchronizer_class: ClassVar[Type[MultiSimSynchronizer]]
    """
    The class of the MultiSimSynchronizer to use.
    """

    builder_class: ClassVar[Type[MultiSimBuilder]]
    """
    The class of the MultiSimBuilder to use.
    """

    simulator: MultiverseSimulator
    """
    The Multiverse simulator instance.
    """

    synchronizer: MultiSimSynchronizer
    """
    The MultiSimSynchronizer instance.
    """

    default_file_path: str
    """
    The default file path to save the world to.
    """

    def __init__(
        self,
        world: World,
        viewer: MultiverseViewer,
        headless: bool = False,
        step_size: float = 1e-3,
        real_time_factor: float = 1.0,
    ):
        """
        Initializes the MultiSim class.

        :param world: The world to simulate.
        :param viewer: The MultiverseViewer to read/write objects.
        :param headless: Whether to run the simulation in headless mode.
        :param step_size: The step size for the simulation.
        :param real_time_factor: The real time factor for the simulation (1.0 = real time, 2.0 = twice as fast, -1.0 = as fast as possible).
        """
        self.builder_class().build_world(world=world, file_path=self.default_file_path)
        self.simulator = self.simulator_class(
            file_path=self.default_file_path,
            viewer=viewer,
            headless=headless,
            step_size=step_size,
            real_time_factor=real_time_factor,
        )
        self.synchronizer = self.synchronizer_class(
            world=world,
            simulator=self.simulator,
        )
        self._viewer = viewer

    def start_simulation(self):
        """
        Starts the simulation. This will start one physics simulation thread and render it at 60Hz.
        """
        assert (
            self.simulator.state != MultiverseSimulatorState.RUNNING
        ), "Simulation is already running."
        self.simulator.start()

    def stop_simulation(self):
        """
        Stops the simulation. This will stop the physics simulation and the rendering.
        """
        self.synchronizer.stop()
        self.simulator.stop()

    def pause_simulation(self):
        """
        Pauses the simulation. This will pause the physics simulation but not the rendering.
        """
        if self.simulator.state != MultiverseSimulatorState.PAUSED:
            self.simulator.pause()

    def unpause_simulation(self):
        """
        Unpauses the simulation. This will unpause the physics simulation.
        """
        if self.simulator.state == MultiverseSimulatorState.PAUSED:
            self.simulator.unpause()

    def reset_simulation(self):
        """
        Resets the simulation. This will reset the physics simulation to the initial state.
        """
        self.simulator.reset()

    def set_write_objects(self, write_objects: Dict[str, Dict[str, List[float]]]):
        """
        Sets the objects to be written to the simulator.
        For example, to set the position and quaternion of an object, you can use the following format:
        {
            "object_name": {
                "position": [x, y, z],
                "quaternion": [w, x, y, z]
            }
        }

        :param write_objects: The objects to be written to the simulator.
        """
        self._viewer.write_objects = write_objects
        if self.simulator.state == MultiverseSimulatorState.PAUSED:
            self.simulator.step()

    def set_read_objects(self, read_objects: Dict[str, Dict[str, List[float]]]):
        """
        Sets the objects to be read from the simulator.

        For example, to read the position and quaternion of an object, you can use the following format:
        {
            "object_name": {
                "position": [0.0, 0.0, 0.0], # Default value
                "quaternion": [1.0, 0.0, 0.0], # Default value
            }
        }
        :param read_objects: The objects to be read from the simulator.
        """
        self._viewer.read_objects = read_objects
        if self.simulator.state == MultiverseSimulatorState.PAUSED:
            self.simulator.step()

    def get_read_objects(self) -> Dict[str, Dict[str, MultiverseAttribute]]:
        """
        Gets the objects that are being read from the simulator.
        For example, if you have set the read objects as follows:
        {
            "object_name": {
                "position": [0.0, 0.0, 0.0],
                "quaternion": [1.0, 0.0, 0.0, 0.0],
            }
        }
        You will get the following format:
        {
            "object_name": {
                "position": MultiverseAttribute(...),
                "quaternion": MultiverseAttribute(...),
            }
        }
        where MultiverseAttribute contains the values of the attribute via the .values() method.
        It will return the values that are being read from the simulator in every simulation step.

        :return: The objects that are being read from the simulator.
        """
        if self.simulator.state == MultiverseSimulatorState.PAUSED:
            self.simulator.step()
        return self._viewer.read_objects

    def is_stable(
        self, body_names: List[str], max_simulation_steps: int = 100, atol: float = 1e-2
    ) -> bool:
        """
        Checks if an object is stable in the world. Stable meaning that it's pose will not change after simulating
        physics in the World. This function will pause the simulation, set the read objects to the given body names,
        unpause the simulation, and check if the pose of the objects change after a certain number of simulation steps.
        If the pose of the objects change, the function will return False. If the pose of the objects do not change,
        the function will return True. After checking, the function will restore the read objects and the simulation state.

        :param body_names: The names of the bodies to check for stability
        :param max_simulation_steps: The maximum number of simulation steps to run
        :param atol: The absolute tolerance for comparing the pose
        :return: True if the object is stable, False otherwise
        """

        origin_read_objects = self.get_read_objects()
        origin_state = self.simulator.state

        self.pause_simulation()
        self.set_read_objects(
            read_objects={
                body_name: {
                    "position": [0.0, 0.0, 0.0],
                    "quaternion": [1.0, 0.0, 0.0, 0.0],
                }
                for body_name in body_names
            }
        )
        initial_body_state = numpy.array(self._viewer.read_data)
        current_simulation_step = self.simulator.current_number_of_steps
        self.unpause_simulation()
        stable = True
        while (
            self.simulator.current_number_of_steps
            < current_simulation_step + max_simulation_steps
        ):
            if numpy.abs(initial_body_state - self._viewer.read_data).max() > atol:
                stable = False
                break
            time.sleep(1e-3)
        self._viewer.read_objects = origin_read_objects
        if origin_state == MultiverseSimulatorState.PAUSED:
            self.pause_simulation()
        return stable


class MujocoSim(MultiSim):
    simulator_class: ClassVar[Type[MultiverseSimulator]] = MultiverseMujocoConnector
    synchronizer_class: ClassVar[Type[MultiSimSynchronizer]] = MujocoSynchronizer
    builder_class: ClassVar[Type[MultiSimBuilder]] = MujocoBuilder
    simulator: MultiverseMujocoConnector
    synchronizer: Type[MultiSimSynchronizer] = MujocoSynchronizer
    default_file_path: str = "/tmp/scene.xml"
