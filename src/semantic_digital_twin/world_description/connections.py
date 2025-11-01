from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np
from typing_extensions import List, TYPE_CHECKING, Union

from .degree_of_freedom import DegreeOfFreedom
from .world_entity import CollisionCheckingConfig, Connection
from .. import spatial_types as cas
from ..datastructures.prefixed_name import PrefixedName
from ..datastructures.types import NpMatrix4x4
from ..exceptions import WorldEntityNotFoundError
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

    dof_name: PrefixedName = field(default=None)
    """
    Name of a Degree of freedom to control movement along the axis.
    """

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
        self._post_init_world_part()

    def _post_init_with_world(self):
        try:
            dof = self._world.get_degree_of_freedom_by_name(self.dof_name)
        except WorldEntityNotFoundError:
            # catch the case where the dof_name is set, but a dof of that name doesn't exist in the world (anymore)
            # can happen if you remove and re-add a connection
            self.dof_name = None
        if self.dof_name is None:
            dof = DegreeOfFreedom(
                name=self.name,
            )
            self._world.add_degree_of_freedom(dof)
            self.dof_name = dof.name
            return

        if dof._world is None:
            self._world.add_degree_of_freedom(dof)

    def _post_init_without_world(self):
        if self.dof_name is None:
            raise ValueError(
                f"{self.__class__.__name__} cannot be created without a world "
                "if the dof is not provided."
            )

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

    x: DegreeOfFreedom = field(default=None)
    """
    Displacement of child KinematicStructureEntity with respect to parent KinematicStructureEntity along the x-axis.
    """
    y: DegreeOfFreedom = field(default=None)
    """
    Displacement of child KinematicStructureEntity with respect to parent KinematicStructureEntity along the y-axis.
    """
    z: DegreeOfFreedom = field(default=None)
    """
    Displacement of child KinematicStructureEntity with respect to parent KinematicStructureEntity along the z-axis.
    """

    qx: DegreeOfFreedom = field(default=None)
    qy: DegreeOfFreedom = field(default=None)
    qz: DegreeOfFreedom = field(default=None)
    qw: DegreeOfFreedom = field(default=None)
    """
    Rotation of child KinematicStructureEntity with respect to parent KinematicStructureEntity represented as a quaternion.
    """

    def __hash__(self):
        return hash(self.name)

    def add_to_world(self, world: World):
        super().add_to_world(world)
        self._post_init_world_part()
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

    def _post_init_with_world(self):
        if all(dof is None for dof in self.passive_dofs):
            with self._world.modify_world():
                self.x = DegreeOfFreedom(name=PrefixedName("x", str(self.name)))
                self._world.add_degree_of_freedom(self.x)
                self.y = DegreeOfFreedom(name=PrefixedName("y", str(self.name)))
                self._world.add_degree_of_freedom(self.y)
                self.z = DegreeOfFreedom(name=PrefixedName("z", str(self.name)))
                self._world.add_degree_of_freedom(self.z)
                self.qx = DegreeOfFreedom(name=PrefixedName("qx", str(self.name)))
                self._world.add_degree_of_freedom(self.qx)
                self.qy = DegreeOfFreedom(name=PrefixedName("qy", str(self.name)))
                self._world.add_degree_of_freedom(self.qy)
                self.qz = DegreeOfFreedom(name=PrefixedName("qz", str(self.name)))
                self._world.add_degree_of_freedom(self.qz)
                self.qw = DegreeOfFreedom(name=PrefixedName("qw", str(self.name)))
                self._world.add_degree_of_freedom(self.qw)
                self._world.state[self.qw.name].position = 1.0
        elif any(dof is None for dof in self.passive_dofs):
            raise ValueError(
                "Connection6DoF can only be created "
                "if you provide all or none of the passive degrees of freedom"
            )

    def _post_init_without_world(self):
        if any(dof is None for dof in self.passive_dofs):
            raise ValueError(
                "Connection6DoF cannot be created without a world "
                "if some passive degrees of freedom are not provided."
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
    x: DegreeOfFreedom = field(default=None)
    y: DegreeOfFreedom = field(default=None)
    z: DegreeOfFreedom = field(default=None)
    roll: DegreeOfFreedom = field(default=None)
    pitch: DegreeOfFreedom = field(default=None)
    yaw: DegreeOfFreedom = field(default=None)
    x_vel: DegreeOfFreedom = field(default=None)
    y_vel: DegreeOfFreedom = field(default=None)

    translation_velocity_limits: float = field(default=0.6)
    rotation_velocity_limits: float = field(default=0.5)

    def add_to_world(self, world: World):
        super().add_to_world(world)
        self._post_init_world_part()
        odom_T_bf = cas.TransformationMatrix.from_xyz_rpy(
            x=self.x.symbols.position,
            y=self.y.symbols.position,
            yaw=self.yaw.symbols.position,
        )
        bf_T_bf_vel = cas.TransformationMatrix.from_xyz_rpy(
            x=self.x_vel.symbols.position, y=self.y_vel.symbols.position
        )
        bf_vel_T_bf = cas.TransformationMatrix.from_xyz_rpy(
            x=0,
            y=0,
            z=self.z.symbols.position,
            roll=self.roll.symbols.position,
            pitch=self.pitch.symbols.position,
            yaw=0,
        )
        self.connection_T_child_expression = odom_T_bf @ bf_T_bf_vel @ bf_vel_T_bf
        self.connection_T_child_expression.child_frame = self.child

    def _post_init_with_world(self):
        if all(dof is None for dof in self.dofs):
            stringified_name = str(self.name)
            lower_translation_limits = DerivativeMap()
            lower_translation_limits.velocity = -self.translation_velocity_limits
            upper_translation_limits = DerivativeMap()
            upper_translation_limits.velocity = self.translation_velocity_limits
            lower_rotation_limits = DerivativeMap()
            lower_rotation_limits.velocity = -self.rotation_velocity_limits
            upper_rotation_limits = DerivativeMap()
            upper_rotation_limits.velocity = self.rotation_velocity_limits

            with self._world.modify_world():
                self.x = DegreeOfFreedom(name=PrefixedName("x", stringified_name))
                self._world.add_degree_of_freedom(self.x)
                self.y = DegreeOfFreedom(name=PrefixedName("y", stringified_name))
                self._world.add_degree_of_freedom(self.y)
                self.z = DegreeOfFreedom(name=PrefixedName("z", stringified_name))
                self._world.add_degree_of_freedom(self.z)
                self.roll = DegreeOfFreedom(name=PrefixedName("roll", stringified_name))
                self._world.add_degree_of_freedom(self.roll)
                self.pitch = DegreeOfFreedom(
                    name=PrefixedName("pitch", stringified_name)
                )
                self._world.add_degree_of_freedom(self.pitch)
                self.yaw = DegreeOfFreedom(
                    name=PrefixedName("yaw", stringified_name),
                    lower_limits=lower_rotation_limits,
                    upper_limits=upper_rotation_limits,
                )
                self._world.add_degree_of_freedom(self.yaw)

                self.x_vel = DegreeOfFreedom(
                    name=PrefixedName("x_vel", stringified_name),
                    lower_limits=lower_translation_limits,
                    upper_limits=upper_translation_limits,
                )
                self._world.add_degree_of_freedom(self.x_vel)
                self.y_vel = DegreeOfFreedom(
                    name=PrefixedName("y_vel", stringified_name),
                    lower_limits=lower_translation_limits,
                    upper_limits=upper_translation_limits,
                )
                self._world.add_degree_of_freedom(self.y_vel)
        elif any(dof is None for dof in self.passive_dofs):
            raise ValueError(
                "OmniDrive can only be created "
                "if you provide all or none of the passive degrees of freedom"
            )

    def _post_init_without_world(self):
        if any(dof is None for dof in self.dofs):
            raise ValueError(
                "OmniDrive cannot be created without a world "
                "if some passive degrees of freedom are not provided."
            )

    @property
    def active_dofs(self) -> List[DegreeOfFreedom]:
        return [self.x_vel, self.y_vel, self.yaw]

    @property
    def passive_dofs(self) -> List[DegreeOfFreedom]:
        return [self.x, self.y, self.z, self.roll, self.pitch]

    @property
    def dofs(self) -> List[DegreeOfFreedom]:
        return self.active_dofs + self.passive_dofs

    def update_state(self, dt: float) -> None:
        state = self._world.state
        state[self.x_vel.name].position = 0
        state[self.y_vel.name].position = 0

        x_vel = state[self.x_vel.name].velocity
        y_vel = state[self.y_vel.name].velocity
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
        return self.x_vel.has_hardware_interface

    @has_hardware_interface.setter
    def has_hardware_interface(self, value: bool) -> None:
        self.x_vel.has_hardware_interface = value
        self.y_vel.has_hardware_interface = value
        self.yaw.has_hardware_interface = value
