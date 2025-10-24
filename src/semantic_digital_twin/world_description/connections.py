from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np
from typing_extensions import List, TYPE_CHECKING, Union, Optional, Dict, Any, Self

from semantic_digital_twin.world_description.geometry import transformation_from_json
from .degree_of_freedom import DegreeOfFreedom
from .world_entity import CollisionCheckingConfig, Connection, KinematicStructureEntity
from .. import spatial_types as cas
from ..datastructures.prefixed_name import PrefixedName
from ..datastructures.types import NpMatrix4x4
from ..spatial_types.derivatives import DerivativeMap

if TYPE_CHECKING:
    from ..world import World


class HasUpdateState(ABC):
    """
    Mixin class for connections that need state updated which are not trivial integrations.
    Typically needed for connections that use active and passive degrees of freedom.
    Look at OmniDrive for an example usage.
    """

    @abstractmethod
    def update_state(self, dt: float) -> None:
        """
        Allows the connection to update the state of its dofs.
        An integration update for active dofs will have happened before this method is called.
        Write directly into self._world.state, but don't touch dofs that don't belong to this connection.
        :param dt: Time passed since last update.
        """
        pass


@dataclass
class FixedConnection(Connection):
    """
    Has 0 degrees of freedom.
    """

    def __hash__(self):
        return hash((self.parent, self.child))


@dataclass
class ActiveConnection(Connection):
    """
    Has one or more degrees of freedom that can be actively controlled, e.g., robot joints.
    """

    frozen_for_collision_avoidance: bool = False
    """
    Should be treated as fixed for collision avoidance.
    Common example are gripper joints, you generally don't want to avoid collisions by closing the fingers, 
    but by moving the whole hand away.
    """

    def to_json(self) -> Dict[str, Any]:
        result = super().to_json()
        result["frozen_for_collision_avoidance"] = self.frozen_for_collision_avoidance
        return result

    @classmethod
    def _from_json(cls, data: Dict[str, Any]) -> Self:
        return cls(
            name=PrefixedName.from_json(data["name"]),
            parent=KinematicStructureEntity.from_json(data["parent"]),
            child=KinematicStructureEntity.from_json(data["child"]),
            parent_T_connection_expression=transformation_from_json(
                data["parent_T_connection_expression"]
            ),
            frozen_for_collision_avoidance=data["frozen_for_collision_avoidance"],
        )

    @property
    def has_hardware_interface(self) -> bool:
        """
        Whether this connection is linked to a controller and can therefore respond to control commands.

        E.g. the caster wheels of a PR2 are active, because they have a DOF, but they are not directly controlled.
        Instead a the omni drive connection is directly controlled and a low level controller translates these commands
        to commands for the caster wheels.

        A door hinge is also active but cannot be controlled.
        """
        return any(dof.has_hardware_interface for dof in self.dofs)

    @has_hardware_interface.setter
    def has_hardware_interface(self, value: bool) -> None:
        for dof in self.dofs:
            dof.has_hardware_interface = value

    @property
    def active_dofs(self) -> List[DegreeOfFreedom]:
        return []

    def set_static_collision_config_for_direct_child_bodies(
        self, collision_config: CollisionCheckingConfig
    ):
        for child_body in self._world.get_direct_child_bodies_with_collision(self):
            if not child_body.get_collision_config().disabled:
                child_body.set_static_collision_config(collision_config)


@dataclass
class PassiveConnection(Connection):
    """
    Has one or more degrees of freedom that cannot be actively controlled.
    Useful if a transformation is only tracked, e.g., the robot's localization.
    """

    @property
    def passive_dofs(self) -> List[DegreeOfFreedom]:
        return []


@dataclass
class ActiveConnection1DOF(ActiveConnection, ABC):
    """
    Superclass for active connections with 1 degree of freedom.
    """

    axis: cas.Vector3 = field(kw_only=True)
    """
    Connection moves along this axis, should be a unit vector.
    The axis is defined relative to the local reference frame of the parent KinematicStructureEntity.
    """

    multiplier: float = 1.0
    """
    Movement along the axis is multiplied by this value. Useful if Connections share DoFs.
    """

    offset: float = 0.0
    """
    Movement along the axis is offset by this value. Useful if Connections share DoFs.
    """

    dof_name: PrefixedName = field(kw_only=True)
    """
    Name of a Degree of freedom to control movement along the axis.
    """

    def to_json(self) -> Dict[str, Any]:
        result = super().to_json()
        result["axis"] = self.axis.to_np().tolist()
        result["multiplier"] = self.multiplier
        result["offset"] = self.offset
        result["dof_name"] = self.dof_name.to_json()
        return result

    @classmethod
    def _from_json(cls, data: Dict[str, Any]) -> Self:
        return cls(
            name=PrefixedName.from_json(data["name"]),
            parent=KinematicStructureEntity.from_json(data["parent"]),
            child=KinematicStructureEntity.from_json(data["child"]),
            parent_T_connection_expression=transformation_from_json(
                data["parent_T_connection_expression"]
            ),
            frozen_for_collision_avoidance=data["frozen_for_collision_avoidance"],
            axis=cas.Vector3.from_iterable(data["axis"]),
            multiplier=data["multiplier"],
            offset=data["offset"],
            dof_name=PrefixedName.from_json(data["dof_name"]),
        )

    def add_to_world(self, world: World):
        super().add_to_world(world)
        if self.multiplier is None:
            self.multiplier = 1
        else:
            self.multiplier = self.multiplier
        if self.offset is None:
            self.offset = 0
        else:
            self.offset = self.offset
        self.axis = self.axis

    @property
    def dof(self) -> DegreeOfFreedom:
        """
        A reference to the Degree of Freedom associated with this connection.
        .. warning:: WITH multiplier and offset applied.
        """
        result = deepcopy(self.raw_dof)
        result.symbols = result.symbols * self.multiplier
        if self.multiplier < 0:
            # if multiplier is negative, we need to swap the limits
            result.lower_limits, result.upper_limits = (
                result.upper_limits,
                result.lower_limits,
            )
        result.lower_limits = result.lower_limits * self.multiplier
        result.upper_limits = result.upper_limits * self.multiplier

        result.symbols.position += self.offset
        if result.lower_limits.position is not None:
            result.lower_limits.position = result.lower_limits.position + self.offset
        if result.upper_limits.position is not None:
            result.upper_limits.position = result.upper_limits.position + self.offset
        return result

    @property
    def raw_dof(self) -> DegreeOfFreedom:
        """
        A reference to the Degree of Freedom associated with this connection.
        .. warning:: WITHOUT multiplier and offset applied.
        """
        return self._world.get_degree_of_freedom_by_name(self.dof_name)

    @property
    def active_dofs(self) -> List[DegreeOfFreedom]:
        return [self.raw_dof]

    def __hash__(self):
        return hash((self.parent, self.child))

    @property
    def position(self) -> float:
        return self._world.state[self.dof.name].position * self.multiplier + self.offset

    @position.setter
    def position(self, value: float) -> None:
        self._world.state[self.dof.name].position = (
            value - self.offset
        ) / self.multiplier
        self._world.notify_state_change()

    @property
    def velocity(self) -> float:
        return self._world.state[self.dof.name].velocity * self.multiplier

    @velocity.setter
    def velocity(self, value: float) -> None:
        self._world.state[self.dof.name].velocity = value / self.multiplier
        self._world.notify_state_change()

    @property
    def acceleration(self) -> float:
        return self._world.state[self.dof.name].acceleration * self.multiplier

    @acceleration.setter
    def acceleration(self, value: float) -> None:
        self._world.state[self.dof.name].acceleration = value / self.multiplier
        self._world.notify_state_change()

    @property
    def jerk(self) -> float:
        return self._world.state[self.dof.name].jerk * self.multiplier

    @jerk.setter
    def jerk(self, value: float) -> None:
        self._world.state[self.dof.name].jerk = value / self.multiplier
        self._world.notify_state_change()


@dataclass
class PrismaticConnection(ActiveConnection1DOF):
    """
    Allows translation along an axis.
    """

    def add_to_world(self, world: World):
        super().add_to_world(world)

        translation_axis = self.axis * self.dof.symbols.position
        self.connection_T_child_expression = cas.TransformationMatrix.from_xyz_rpy(
            x=translation_axis[0],
            y=translation_axis[1],
            z=translation_axis[2],
            child_frame=self.child,
        )

    def __hash__(self):
        return hash((self.parent, self.child))


@dataclass
class RevoluteConnection(ActiveConnection1DOF):
    """
    Allows rotation about an axis.
    """

    def add_to_world(self, world: World):
        super().add_to_world(world)

        self.connection_T_child_expression = (
            cas.TransformationMatrix.from_xyz_axis_angle(
                axis=self.axis,
                angle=self.dof.symbols.position,
                child_frame=self.child,
            )
        )

    def __hash__(self):
        return hash((self.parent, self.child))


@dataclass
class Connection6DoF(PassiveConnection):
    """
    Has full 6 degrees of freedom, that cannot be actively controlled.
    Useful for synchronizing with transformations from external providers.
    """

    x_name: PrefixedName = field(kw_only=True)
    """
    Displacement of child KinematicStructureEntity with respect to parent KinematicStructureEntity along the x-axis.
    """
    y_name: PrefixedName = field(kw_only=True)
    """
    Displacement of child KinematicStructureEntity with respect to parent KinematicStructureEntity along the y-axis.
    """
    z_name: PrefixedName = field(kw_only=True)
    """
    Displacement of child KinematicStructureEntity with respect to parent KinematicStructureEntity along the z-axis.
    """

    qx_name: PrefixedName = field(kw_only=True)
    qy_name: PrefixedName = field(kw_only=True)
    qz_name: PrefixedName = field(kw_only=True)
    qw_name: PrefixedName = field(kw_only=True)
    """
    Rotation of child KinematicStructureEntity with respect to parent KinematicStructureEntity represented as a quaternion.
    """

    def to_json(self) -> Dict[str, Any]:
        result = super().to_json()
        result["x_name"] = self.x_name.to_json()
        result["y_name"] = self.y_name.to_json()
        result["z_name"] = self.z_name.to_json()
        result["qx_name"] = self.qx_name.to_json()
        result["qy_name"] = self.qy_name.to_json()
        result["qz_name"] = self.qz_name.to_json()
        result["qw_name"] = self.qw_name.to_json()
        return result

    @classmethod
    def _from_json(cls, data: Dict[str, Any]) -> Self:
        return cls(
            name=PrefixedName.from_json(data["name"]),
            parent=KinematicStructureEntity.from_json(data["parent"]),
            child=KinematicStructureEntity.from_json(data["child"]),
            parent_T_connection_expression=transformation_from_json(
                data["parent_T_connection_expression"]
            ),
            x_name=PrefixedName.from_json(data["x_name"]),
            y_name=PrefixedName.from_json(data["y_name"]),
            z_name=PrefixedName.from_json(data["z_name"]),
            qx_name=PrefixedName.from_json(data["qx_name"]),
            qy_name=PrefixedName.from_json(data["qy_name"]),
            qz_name=PrefixedName.from_json(data["qz_name"]),
            qw_name=PrefixedName.from_json(data["qw_name"]),
        )

    @property
    def x(self) -> DegreeOfFreedom:
        return self._world.get_degree_of_freedom_by_name(self.x_name)

    @property
    def y(self) -> DegreeOfFreedom:
        return self._world.get_degree_of_freedom_by_name(self.y_name)

    @property
    def z(self) -> DegreeOfFreedom:
        return self._world.get_degree_of_freedom_by_name(self.z_name)

    @property
    def qx(self) -> DegreeOfFreedom:
        return self._world.get_degree_of_freedom_by_name(self.qx_name)

    @property
    def qy(self) -> DegreeOfFreedom:
        return self._world.get_degree_of_freedom_by_name(self.qy_name)

    @property
    def qz(self) -> DegreeOfFreedom:
        return self._world.get_degree_of_freedom_by_name(self.qz_name)

    @property
    def qw(self) -> DegreeOfFreedom:
        return self._world.get_degree_of_freedom_by_name(self.qw_name)

    def __hash__(self):
        return hash(self.name)

    def add_to_world(self, world: World):
        super().add_to_world(world)
        parent_P_child = cas.Point3(
            x_init=self.x.symbols.position,
            y_init=self.y.symbols.position,
            z_init=self.z.symbols.position,
        )
        parent_R_child = cas.Quaternion(
            x_init=self.qx.symbols.position,
            y_init=self.qy.symbols.position,
            z_init=self.qz.symbols.position,
            w_init=self.qw.symbols.position,
        ).to_rotation_matrix()
        self.connection_T_child_expression = (
            cas.TransformationMatrix.from_point_rotation_matrix(
                point=parent_P_child,
                rotation_matrix=parent_R_child,
                child_frame=self.child,
            )
        )

    @classmethod
    def with_auto_generated_dofs(
        cls,
        world: World,
        parent: KinematicStructureEntity,
        child: KinematicStructureEntity,
        name: Optional[PrefixedName] = None,
        parent_T_connection_expression: Optional[cas.TransformationMatrix] = None,
    ) -> Self:
        """
        Creates an instance of the class with automatically generated degrees of freedom (DoFs)
        for the provided parent and child kinematic entities within the specified world.

        This method initializes and adds the required degrees of freedom to the world,
        and sets their properties accordingly. It generates a name for the connection if
        none is provided, and ensures valid initial state for relevant degrees of freedom.

        :param world: The World object where the degrees of freedom are added and modified.
        :param parent: The KinematicStructureEntity serving as the parent.
        :param child: The KinematicStructureEntity serving as the child.
        :param name: An optional PrefixedName for the connection. If None, it will be
                     auto-generated based on the parent and child names.
        :param parent_T_connection_expression: Optional transformation matrix specifying
                                               the connection relationship between parent
                                               and child entities.
        :return: A new instance of the class representing the parent-child connection with
                 automatically defined degrees of freedom.
        """
        if name is None:
            name = PrefixedName(f"{parent.name.name}_T_{child.name.name}")

        with world.modify_world():
            x = DegreeOfFreedom(name=PrefixedName("x", str(name)))
            world.add_degree_of_freedom(x)
            y = DegreeOfFreedom(name=PrefixedName("y", str(name)))
            world.add_degree_of_freedom(y)
            z = DegreeOfFreedom(name=PrefixedName("z", str(name)))
            world.add_degree_of_freedom(z)
            qx = DegreeOfFreedom(name=PrefixedName("qx", str(name)))
            world.add_degree_of_freedom(qx)
            qy = DegreeOfFreedom(name=PrefixedName("qy", str(name)))
            world.add_degree_of_freedom(qy)
            qz = DegreeOfFreedom(name=PrefixedName("qz", str(name)))
            world.add_degree_of_freedom(qz)
            qw = DegreeOfFreedom(name=PrefixedName("qw", str(name)))
            world.add_degree_of_freedom(qw)
            world.state[qw.name].position = 1.0

        return cls(
            parent=parent,
            child=child,
            parent_T_connection_expression=parent_T_connection_expression,
            name=name,
            x_name=x.name,
            y_name=y.name,
            z_name=z.name,
            qx_name=qx.name,
            qy_name=qy.name,
            qz_name=qz.name,
            qw_name=qw.name,
        )

    @property
    def passive_dofs(self) -> List[DegreeOfFreedom]:
        return [self.x, self.y, self.z, self.qx, self.qy, self.qz, self.qw]

    @property
    def origin(self) -> cas.TransformationMatrix:
        return super().origin

    @origin.setter
    def origin(
        self, transformation: Union[NpMatrix4x4, cas.TransformationMatrix]
    ) -> None:
        if not isinstance(transformation, cas.TransformationMatrix):
            transformation = cas.TransformationMatrix(data=transformation)
        position = transformation.to_position().to_np()
        orientation = transformation.to_rotation_matrix().to_quaternion().to_np()
        self._world.state[self.x.name].position = position[0]
        self._world.state[self.y.name].position = position[1]
        self._world.state[self.z.name].position = position[2]
        self._world.state[self.qx.name].position = orientation[0]
        self._world.state[self.qy.name].position = orientation[1]
        self._world.state[self.qz.name].position = orientation[2]
        self._world.state[self.qw.name].position = orientation[3]
        self._world.notify_state_change()


@dataclass
class OmniDrive(ActiveConnection, PassiveConnection, HasUpdateState):
    """
    A connection describing an omnidirectional drive.
    It can rotate about its z-axis and drive on the x-y plane simultaneously.
    - x/y: Passive dofs describing the measured odometry with respect to parent frame.
        We assume that the robot can't fly, and we can't measure its z-axis position, so z=0.
        The odometry sensors typically provide velocity measurements with respect to the child frame,
        therefore the velocity values of x/y must stay 0.
    - x_vel/y_vel: The measured and commanded velocity is represented with respect to the child frame with these
        active dofs. It must be ensured that their position values stay 0.
    - roll/pitch: Some robots, like the PR2, have sensors to measure pitch and roll using an IMU,
        we therefore have passive dofs for them.
    - yaw: Since the robot can only rotate about its z-axis, we don't need different dofs for position and velocity of yaw.
        They are combined into one active dof.
    """

    # passive dofs
    x_name: PrefixedName = field(kw_only=True)
    y_name: PrefixedName = field(kw_only=True)
    roll_name: PrefixedName = field(kw_only=True)
    pitch_name: PrefixedName = field(kw_only=True)

    # active dofs
    yaw_name: PrefixedName = field(kw_only=True)
    x_velocity_name: PrefixedName = field(kw_only=True)
    y_velocity_name: PrefixedName = field(kw_only=True)

    def to_json(self) -> Dict[str, Any]:
        result = super().to_json()
        result["x_name"] = self.x_name.to_json()
        result["y_name"] = self.y_name.to_json()
        result["roll_name"] = self.roll_name.to_json()
        result["pitch_name"] = self.pitch_name.to_json()
        result["yaw_name"] = self.yaw_name.to_json()
        result["x_velocity_name"] = self.x_velocity_name.to_json()
        result["y_velocity_name"] = self.y_velocity_name.to_json()
        return result

    @classmethod
    def _from_json(cls, data: Dict[str, Any]) -> Self:
        return cls(
            name=PrefixedName.from_json(data["name"]),
            parent=KinematicStructureEntity.from_json(data["parent"]),
            child=KinematicStructureEntity.from_json(data["child"]),
            parent_T_connection_expression=transformation_from_json(
                data["parent_T_connection_expression"]
            ),
            x_name=PrefixedName.from_json(data["x_name"]),
            y_name=PrefixedName.from_json(data["y_name"]),
            roll_name=PrefixedName.from_json(data["roll_name"]),
            pitch_name=PrefixedName.from_json(data["pitch_name"]),
            yaw_name=PrefixedName.from_json(data["yaw_name"]),
            x_velocity_name=PrefixedName.from_json(data["x_velocity_name"]),
            y_velocity_name=PrefixedName.from_json(data["y_velocity_name"]),
        )

    @property
    def x(self) -> DegreeOfFreedom:
        return self._world.get_degree_of_freedom_by_name(self.x_name)

    @property
    def y(self) -> DegreeOfFreedom:
        return self._world.get_degree_of_freedom_by_name(self.y_name)

    @property
    def roll(self) -> DegreeOfFreedom:
        return self._world.get_degree_of_freedom_by_name(self.roll_name)

    @property
    def pitch(self) -> DegreeOfFreedom:
        return self._world.get_degree_of_freedom_by_name(self.pitch_name)

    @property
    def yaw(self) -> DegreeOfFreedom:
        return self._world.get_degree_of_freedom_by_name(self.yaw_name)

    @property
    def x_velocity(self) -> DegreeOfFreedom:
        return self._world.get_degree_of_freedom_by_name(self.x_velocity_name)

    @property
    def y_velocity(self) -> DegreeOfFreedom:
        return self._world.get_degree_of_freedom_by_name(self.y_velocity_name)

    def add_to_world(self, world: World):
        super().add_to_world(world)
        odom_T_bf = cas.TransformationMatrix.from_xyz_rpy(
            x=self.x.symbols.position,
            y=self.y.symbols.position,
            yaw=self.yaw.symbols.position,
        )
        bf_T_bf_vel = cas.TransformationMatrix.from_xyz_rpy(
            x=self.x_velocity.symbols.position, y=self.y_velocity.symbols.position
        )
        bf_vel_T_bf = cas.TransformationMatrix.from_xyz_rpy(
            x=0,
            y=0,
            z=0,
            roll=self.roll.symbols.position,
            pitch=self.pitch.symbols.position,
            yaw=0,
        )
        self.connection_T_child_expression = odom_T_bf @ bf_T_bf_vel @ bf_vel_T_bf
        self.connection_T_child_expression.child_frame = self.child

    @classmethod
    def with_auto_generated_dofs(
        cls,
        world: World,
        parent: KinematicStructureEntity,
        child: KinematicStructureEntity,
        name: Optional[PrefixedName] = None,
        parent_T_connection_expression: Optional[cas.TransformationMatrix] = None,
        translation_velocity_limits: float = 0.6,
        rotation_velocity_limits: float = 0.5,
    ) -> Self:
        """
        Creates an instance of the class with automatically generated degrees of freedom
        (DOFs) for translation on the x and y axes, rotation along roll, pitch, and yaw
        axes, and velocity limits for translation and rotation.

        This method modifies the provided world to add all required degrees of freedom
        and their limits, based on the provided settings. Names for the degrees of
        freedom are auto-generated using the stringified version of the provided name
        or its default setting.

        :param world: The world where the configuration is being applied, and degrees of freedom are added.
        :param parent: The parent kinematic structure entity.
        :param child: The child kinematic structure entity.
        :param name: Name of the connection. If None, it will be auto-generated.
        :param parent_T_connection_expression: Transformation matrix representing the
            relative position/orientation of the child to the parent. Default is Identity.
        :param translation_velocity_limits: The velocity limit applied to the
            translation degrees of freedom (default is 0.6).
        :param rotation_velocity_limits: The velocity limit applied to the rotation
            degrees of freedom (default is 0.5).
        :return: An instance of the class with the auto-generated DOFs incorporated.
        """
        with world.modify_world():
            stringified_name = str(name)
            lower_translation_limits = DerivativeMap()
            lower_translation_limits.velocity = -translation_velocity_limits
            upper_translation_limits = DerivativeMap()
            upper_translation_limits.velocity = translation_velocity_limits
            lower_rotation_limits = DerivativeMap()
            lower_rotation_limits.velocity = -rotation_velocity_limits
            upper_rotation_limits = DerivativeMap()
            upper_rotation_limits.velocity = rotation_velocity_limits

            with world.modify_world():
                x = DegreeOfFreedom(name=PrefixedName("x", stringified_name))
                world.add_degree_of_freedom(x)
                y = DegreeOfFreedom(name=PrefixedName("y", stringified_name))
                world.add_degree_of_freedom(y)
                roll = DegreeOfFreedom(name=PrefixedName("roll", stringified_name))
                world.add_degree_of_freedom(roll)
                pitch = DegreeOfFreedom(name=PrefixedName("pitch", stringified_name))
                world.add_degree_of_freedom(pitch)
                yaw = DegreeOfFreedom(
                    name=PrefixedName("yaw", stringified_name),
                    lower_limits=lower_rotation_limits,
                    upper_limits=upper_rotation_limits,
                )
                world.add_degree_of_freedom(yaw)

                x_vel = DegreeOfFreedom(
                    name=PrefixedName("x_vel", stringified_name),
                    lower_limits=lower_translation_limits,
                    upper_limits=upper_translation_limits,
                )
                world.add_degree_of_freedom(x_vel)
                y_vel = DegreeOfFreedom(
                    name=PrefixedName("y_vel", stringified_name),
                    lower_limits=lower_translation_limits,
                    upper_limits=upper_translation_limits,
                )
                world.add_degree_of_freedom(y_vel)

        return cls(
            parent=parent,
            child=child,
            parent_T_connection_expression=parent_T_connection_expression,
            name=name,
            x_name=x.name,
            y_name=y.name,
            roll_name=roll.name,
            pitch_name=pitch.name,
            yaw_name=yaw.name,
            x_velocity_name=x_vel.name,
            y_velocity_name=y_vel.name,
        )

    @property
    def active_dofs(self) -> List[DegreeOfFreedom]:
        return [self.x_velocity, self.y_velocity, self.yaw]

    @property
    def passive_dofs(self) -> List[DegreeOfFreedom]:
        return [self.x, self.y, self.roll, self.pitch]

    @property
    def dofs(self) -> List[DegreeOfFreedom]:
        return self.active_dofs + self.passive_dofs

    def update_state(self, dt: float) -> None:
        state = self._world.state
        state[self.x_velocity.name].position = 0
        state[self.y_velocity.name].position = 0

        x_vel = state[self.x_velocity.name].velocity
        y_vel = state[self.y_velocity.name].velocity
        delta = state[self.yaw.name].position
        x_velocity = np.cos(delta) * x_vel - np.sin(delta) * y_vel
        state[self.x.name].position += x_velocity * dt
        y_velocity = np.sin(delta) * x_vel + np.cos(delta) * y_vel
        state[self.y.name].position += y_velocity * dt

    @property
    def origin(self) -> cas.TransformationMatrix:
        return super().origin

    @origin.setter
    def origin(
        self, transformation: Union[NpMatrix4x4, cas.TransformationMatrix]
    ) -> None:
        if isinstance(transformation, np.ndarray):
            transformation = cas.TransformationMatrix(data=transformation)
        position = transformation.to_position()
        roll, pitch, yaw = transformation.to_rotation_matrix().to_rpy()
        assert (
            position.z.to_np() == 0.0
        ), "OmniDrive only supports planar movement in the XY plane, z must be 0"
        assert (
            roll.to_np() == 0.0
        ), "OmniDrive only supports planar movement in the XY plane, roll must be 0"
        assert (
            pitch.to_np() == 0.0
        ), "OmniDrive only supports planar movement in the XY plane, pitch must be 0"
        self._world.state[self.x.name].position = position.x.to_np()
        self._world.state[self.y.name].position = position.y.to_np()
        self._world.state[self.yaw.name].position = yaw.to_np()
        self._world.notify_state_change()

    def get_free_variable_names(self) -> List[PrefixedName]:
        return [self.x.name, self.y.name, self.yaw.name]

    def __hash__(self):
        return hash(self.name)

    @property
    def has_hardware_interface(self) -> bool:
        return self.x_velocity.has_hardware_interface

    @has_hardware_interface.setter
    def has_hardware_interface(self, value: bool) -> None:
        self.x_velocity.has_hardware_interface = value
        self.y_velocity.has_hardware_interface = value
        self.yaw.has_hardware_interface = value
