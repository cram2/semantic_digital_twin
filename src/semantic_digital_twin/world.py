from __future__ import absolute_import
from __future__ import annotations

import inspect
import logging
import os
from copy import deepcopy
from dataclasses import dataclass, field
from enum import IntEnum
from functools import wraps, lru_cache, cached_property
from itertools import combinations_with_replacement

import matplotlib.pyplot as plt
import numpy as np
import rustworkx as rx
import rustworkx.visit
import rustworkx.visualization
from lxml import etree
from rustworkx import NoEdgeBetweenNodes
from typing_extensions import (
    Dict,
    Tuple,
    Optional,
    TypeVar,
    Union,
    Callable,
    Any,
    Iterable,
)
from typing_extensions import List
from typing_extensions import Type, Set

from .callbacks.callback import ModelChangeCallback
from .collision_checking.collision_detector import CollisionDetector
from .collision_checking.trimesh_collision_detector import TrimeshCollisionDetector
from .datastructures.prefixed_name import PrefixedName
from .datastructures.types import NpMatrix4x4
from .exceptions import (
    DuplicateWorldEntityError,
    AddingAnExistingSemanticAnnotationError,
    WorldEntityNotFoundError,
    AlreadyBelongsToAWorldError,
    DuplicateKinematicStructureEntityError,
    MissingWorldModificationContextError,
)
from .robots.abstract_robot import AbstractRobot
from .spatial_computations.forward_kinematics import ForwardKinematicsVisitor
from .spatial_computations.ik_solver import InverseKinematicsSolver
from .spatial_computations.raytracer import RayTracer
from .spatial_types import spatial_types as cas
from .spatial_types.derivatives import Derivatives
from .utils import IDGenerator, copy_lru_cache
from .world_description.connection_factories import ConnectionFactory
from .world_description.connections import (
    ActiveConnection,
    PassiveConnection,
    FixedConnection,
    Connection6DoF,
    ActiveConnection1DOF,
)
from .world_description.connections import HasUpdateState
from .world_description.degree_of_freedom import DegreeOfFreedom
from .world_description.world_entity import (
    Connection,
    SemanticAnnotation,
    KinematicStructureEntity,
    Region,
    GenericKinematicStructureEntity,
    GenericConnection,
    CollisionCheckingConfig,
    Body,
    WorldEntity,
    GenericWorldEntity,
)
from .world_description.world_modification import (
    RemoveSemanticAnnotationModification,
)
from .world_description.world_modification import (
    WorldModelModification,
    AddDegreeOfFreedomModification,
    RemoveDegreeOfFreedomModification,
    AddKinematicStructureEntityModification,
    AddConnectionModification,
    RemoveBodyModification,
    RemoveConnectionModification,
    WorldModelModificationBlock,
    SetDofHasHardwareInterface,
    AddSemanticAnnotationModification,
)
from .world_description.world_state import WorldState

logger = logging.getLogger(__name__)

id_generator = IDGenerator()

GenericSemanticAnnotation = TypeVar(
    "GenericSemanticAnnotation", bound=SemanticAnnotation
)

FunctionStack = List[Tuple[Callable, Dict[str, Any]]]


class PlotAlignment(IntEnum):
    HORIZONTAL = 0
    VERTICAL = 1


class ResetStateContextManager:
    """
    A context manager for resetting the state of a given `World` instance.

    This class is designed to allow operations to be performed on a `World`
    object, ensuring that its state can be safely returned to its previous
    condition upon leaving the context. If no exceptions occur within the
    context, the original state of the `World` instance is restored, and the
    state change is notified.
    """

    def __init__(self, world: World):
        self.world = world

    def __enter__(self) -> None:
        self.state = deepcopy(self.world.state)

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[Exception],
        exc_tb: Optional[type],
    ) -> None:
        if exc_type is None:
            self.world.state = self.state
            self.world.notify_state_change()


class WorldModelUpdateContextManager:
    """
    Context manager for updating the state of a given `World` instance.
    This class manages that updates to the world within the context of this class only trigger recomputations after all
    desired updates have been performed.
    """

    first: bool = True
    """
    First time flag.
    """

    modification: Callable = None
    """
    Modification function.
    """

    arguments: Dict[str, Any] = None
    """
    Arguments of the modification function.
    """

    def __init__(self, world: World):
        self.world = world

    def __enter__(self):
        if self.world.world_is_being_modified:
            self.first = False
        self.world.world_is_being_modified = True

        if self.first:
            self.world.get_world_model_manager().current_model_modification_block = WorldModelModificationBlock()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.first:
            self.world.get_world_model_manager().model_modification_blocks.append(
                self.world.get_world_model_manager().current_model_modification_block
            )
            self.world.get_world_model_manager().current_model_modification_block = None
            if exc_type is None:
                self.world._notify_model_change()
            self.world.world_is_being_modified = False


class AtomicWorldModificationNotAtomic(Exception):
    """
    Exception raised when atomic world modifications are overlapping.
    If this exception is raised, it means that somewhere in the code a function decorated with @atomic_world_modification
    triggered another function decorated with it. This must not happen ever!
    """


def atomic_world_modification(
    func=None, modification: Type[WorldModelModification] = None
):
    """
    Decorator for ensuring atomicity in world modification operations.

    This decorator ensures that no other atomic world modification is in progress when the decorated function is executed.
    It records the function call along with its arguments for potential replay or tracking purposes.
    If an operation is attempted when the world is locked, it raises an appropriate exception.

    Raises:
        AtomicWorldModificationNotAtomic: If the world is already locked during the execution of another atomic operation.
    """

    def _decorate(func):

        sig = inspect.signature(func)

        @wraps(func)
        def wrapper(self: World, *args, **kwargs):
            if self._atomic_modification_is_being_executed:
                raise AtomicWorldModificationNotAtomic(f"World {self} is locked.")
            self._atomic_modification_is_being_executed = True

            # bind args and kwargs
            bound = sig.bind_partial(
                self, *args, **kwargs
            )  # use bind() if you require all args
            bound.apply_defaults()  # fill in default values

            # record function call
            # Build a dict with all arguments (including positional), excluding 'self'
            bound_args = dict(bound.arguments)
            bound_args.pop("self", None)
            if self.get_world_model_manager().current_model_modification_block is None:
                raise MissingWorldModificationContextError(func)
            self.get_world_model_manager().current_model_modification_block.append(
                modification.from_kwargs(bound_args)
            )

            result = func(self, *args, **kwargs)

            self._atomic_modification_is_being_executed = False
            return result

        return wrapper

    if func is None:
        return _decorate

    return _decorate(func)


@dataclass
class CollisionPairManager:
    """
    Manages disabled collision pairs in the world.
    """

    world: World
    """
    The world to manage collision pairs for.
    """

    _disabled_collision_pairs: Set[Tuple[Body, Body]] = field(
        default_factory=set, repr=False
    )
    """
    Collisions for these Body pairs is disabled.f
    """

    _temp_disabled_collision_pairs: Set[Tuple[Body, Body]] = field(
        default_factory=set, repr=False
    )
    """
    A set of Body pairs for which collisions are temporarily disabled.
    """

    def reset_temporary_collision_config(self):
        self._temp_disabled_collision_pairs = set()
        for body in self.world.bodies_with_enabled_collision:
            body.reset_temporary_collision_config()

    @property
    def disabled_collision_pairs(
        self,
    ) -> Set[Tuple[Body, Body]]:
        return self._disabled_collision_pairs | self._temp_disabled_collision_pairs

    @property
    def enabled_collision_pairs(self) -> Set[Tuple[Body, Body]]:
        """
        The complement of disabled_collision_pairs with respect to all possible body combinations with enabled collision.
        """
        all_combinations = set(
            combinations_with_replacement(self.world.bodies_with_enabled_collision, 2)
        )
        return all_combinations - self.disabled_collision_pairs

    def add_temp_disabled_collision_pair(
        self, body_a: KinematicStructureEntity, body_b: KinematicStructureEntity
    ):
        """
        Disable collision checking between two bodies
        """
        pair = tuple(sorted([body_a, body_b], key=lambda b: b.name))
        self._temp_disabled_collision_pairs.add(pair)

    def load_collision_srdf(self, file_path: str):
        """
        Creates a CollisionConfig instance from an SRDF file.

        Parse an SRDF file to configure disabled collision pairs or bodies for a given world.
        Process SRDF elements like `disable_collisions`, `disable_self_collision`,
        or `disable_all_collisions` to update collision configuration
        by referencing bodies in the provided `world`.

        :param file_path: The path to the SRDF file used for collision configuration.
        """
        SRDF_DISABLE_ALL_COLLISIONS: str = "disable_all_collisions"
        SRDF_DISABLE_SELF_COLLISION: str = "disable_self_collision"
        SRDF_MOVEIT_DISABLE_COLLISIONS: str = "disable_collisions"

        srdf = etree.parse(file_path)
        srdf_root = srdf.getroot()

        children_with_tag = [child for child in srdf_root if hasattr(child, "tag")]

        child_disable_collisions = [
            c for c in children_with_tag if c.tag == SRDF_DISABLE_ALL_COLLISIONS
        ]

        for c in child_disable_collisions:
            body = self.world.get_body_by_name(c.attrib["link"])
            body.set_static_collision_config(CollisionCheckingConfig(disabled=True))

        child_disable_moveit_and_self_collision = [
            c
            for c in children_with_tag
            if c.tag in {SRDF_MOVEIT_DISABLE_COLLISIONS, SRDF_DISABLE_SELF_COLLISION}
        ]

        disabled_collision_pairs = [
            (body_a, body_b)
            for child in child_disable_moveit_and_self_collision
            if (
                body_a := self.world.get_body_by_name(child.attrib["link1"])
            ).has_collision()
            and (
                body_b := self.world.get_body_by_name(child.attrib["link2"])
            ).has_collision()
        ]

        for body_a, body_b in disabled_collision_pairs:
            self.add_disabled_collision_pair(body_a, body_b)

    def disable_collisions_for_adjacent_bodies(self):
        """
        Computes pairs of bodies that should not be collision checked because they have no controlled connections
        between them.

        When all connections between two bodies are not controlled, these bodies cannot move relative to each
        other, so collision checking between them is unnecessary.

        :return: Set of body pairs that should have collisions disabled
        """

        body_combinations = combinations_with_replacement(
            self.world.bodies_with_enabled_collision, 2
        )

        for body_a, body_b in (
            (a, b)
            for a, b in body_combinations
            if not self.world.is_controlled_connection_in_chain(a, b)
        ):
            self.add_disabled_collision_pair(body_a, body_b)

    def disable_non_robot_collisions(self) -> None:
        """
        Disable collision checks between bodies that do not belong to any robot.
        """
        # Bodies that are part of any robot and participate in collisions
        robot_bodies: Set[Body] = {
            body
            for robot in self.world.get_semantic_annotations_by_type(AbstractRobot)
            for body in robot.bodies_with_collisions
        }

        # Bodies with collisions that are NOT part of a robot
        non_robot_bodies: Set[Body] = (
            set(self.world.bodies_with_enabled_collision) - robot_bodies
        )
        if not non_robot_bodies:
            return

        # Disable every unordered pair (including self-collisions) exactly once
        for a, b in combinations_with_replacement(non_robot_bodies, 2):
            self.add_disabled_collision_pair(a, b)

    def add_disabled_collision_pair(self, body_a: Body, body_b: Body):
        """
        Disable collision checking between two bodies
        """
        pair = tuple(sorted([body_a, body_b], key=lambda body: body.name))
        self._disabled_collision_pairs.add(pair)

@dataclass
class WorldModelManager:
    """
    Manages the world model version and modification blocks.
    """

    model_version: int = 0
    """
    The version of the model. This increases whenever a change to the kinematic model is made. Mostly triggered
    by adding/removing bodies and connections.
    """

    model_modification_blocks: List[WorldModelModificationBlock] = field(
        default_factory=list, repr=False, init=False
    )
    """
    All atomic modifications applied to the world. Tracked by @atomic_world_modification.
    The field itself is a list of lists. The outer lists indicates when to trigger the model/state change callbacks.
    The inner list is a block of modifications where change callbacks must not be called in between.
    """

    current_model_modification_block: Optional[WorldModelModificationBlock] = field(
        default=None, repr=False, init=False
    )
    """
    The current modification block called within one context of @atomic_world_modification.
    """

    model_change_callbacks: List[ModelChangeCallback] = field(
        default_factory=list, repr=False
    )
    """
    Callbacks to be called when the model of the world changes.
    """

    def update_model_version_and_notify_callbacks(self) -> None:
        """
        Notifies the system of a model change and updates necessary states, caches,
        and forward kinematics expressions while also triggering registered callbacks
        for model changes.
        """
        self.model_version += 1
        for callback in self.model_change_callbacks:
            callback.notify()


@dataclass
class World:
    """
    A class representing the world.
    The world manages a set of kinematic structure entities and connections represented as a tree-like graph.
    The nodes represent kinematic structure entities in the world, and the edges represent joins between them.
    """

    kinematic_structure: rx.PyDAG[KinematicStructureEntity] = field(
        default_factory=lambda: rx.PyDAG(multigraph=False), kw_only=True, repr=False
    )
    """
    The kinematic structure of the world.
    The kinematic structure is a tree shaped directed graph where the nodes represent kinematic structure entities
     in the world, and the edges represent connections between them.
    """

    semantic_annotations: List[SemanticAnnotation] = field(
        default_factory=list, repr=False
    )
    """
    All semantic annotations the world is aware of.
    """

    degrees_of_freedom: List[DegreeOfFreedom] = field(default_factory=list)
    """
    All degrees of freedom in the world.
    """

    state: WorldState = field(default_factory=WorldState)
    """
    2d array where rows are derivatives and columns are dof values for that derivative.
    """

    world_is_being_modified: bool = False
    """
    Is set to True, when a world.modify_world context is used.
    """

    name: Optional[str] = None
    """
    Name of the world. May act as default namespace for all bodies and semantic annotations in the world which do not have a prefix.
    """

    _atomic_modification_is_being_executed: bool = field(init=False, default=False)
    """
    Flag that indicates if an atomic world operation is currently being executed.
    See `atomic_world_modification` for more information.
    """

    _collision_pair_manager: CollisionPairManager = field(init=False, repr=False)
    """
    Manages disabled collision pairs in the world.
    """

    _world_model_manager: WorldModelManager = field(default_factory=WorldModelManager, repr=False)
    """
    Manages the world model version and modification blocks.
    """

    def __post_init__(self):
        self._collision_pair_manager = CollisionPairManager(self)

    def __hash__(self):
        return hash(id(self))

    def __str__(self):
        return f"{self.__class__.__name__} with {len(self.kinematic_structure_entities)} bodies."

    def validate(self) -> bool:
        """
        Validate the world.

        The world must be a tree.
        :return: True if the world is valid, raises an AssertionError otherwise.
        """
        if self.empty:
            return True
        assert len(self.kinematic_structure_entities) == (len(self.connections) + 1)
        assert rx.is_weakly_connected(self.kinematic_structure)
        actual_dofs = {
            dof for connection in self.connections for dof in connection.dofs
        }
        assert actual_dofs == set(
            self.degrees_of_freedom
        ), "self.degrees_of_freedom does not match the actual dofs used in connections. Did you forget to call deleted_orphaned_dof()?"
        return True

    # %% Properties
    @property
    def root(self) -> Optional[KinematicStructureEntity]:
        """
        The root of the world is the unique node with in-degree 0.

        :return: The root of the world.
        """

        if self.empty:
            return None

        possible_roots = [
            node
            for node in self.kinematic_structure_entities
            if self.kinematic_structure.in_degree(node.index) == 0
        ]
        if len(possible_roots) == 1:
            return possible_roots[0]

        raise ValueError(
            f"A World must have exactly one root. Found {len(possible_roots)} possible roots: {possible_roots}."
        )

    @property
    def active_degrees_of_freedom(self) -> Set[DegreeOfFreedom]:
        active_connections = self.get_connections_by_type(ActiveConnection)
        dofs = {
            dof
            for connection in active_connections
            for dof in connection.active_dofs
        }
        return dofs

    @property
    def passive_degrees_of_freedom(self) -> Set[DegreeOfFreedom]:
        passive_connections = self.get_connections_by_type(PassiveConnection)
        dofs = {
            dof
            for connection in passive_connections
            for dof in connection.passive_dofs
        }
        return dofs

    @property
    def kinematic_structure_entities(self) -> List[KinematicStructureEntity]:
        """
        :return: A list of all bodies in the world.
        """
        return list(self.kinematic_structure.nodes())

    @property
    def kinematic_structure_entities_topologically_sorted(
        self,
    ) -> List[KinematicStructureEntity]:
        """
        Return a list of all kinematic_structure_entities in the world, sorted topologically.
        """
        indices = rx.topological_sort(self.kinematic_structure)
        return [self.kinematic_structure[index] for index in indices]

    @property
    def regions(self) -> List[Region]:
        """
        :return: A list of all regions in the world.
        """
        return self.get_kinematic_structure_entity_by_type(Region)

    @property
    def bodies(self) -> List[Body]:
        """
        :return: A list of all bodies in the world.
        """
        return self.get_kinematic_structure_entity_by_type(Body)

    @property
    def bodies_with_enabled_collision(self) -> List[Body]:
        return [
            b
            for b in self.bodies
            if b.has_collision()
            and b.get_collision_config
            and not b.get_collision_config().disabled
        ]

    @property
    def bodies_topologically_sorted(self) -> List[Body]:
        return [
            body
            for body in self.kinematic_structure_entities_topologically_sorted
            if isinstance(body, Body)
        ]

    @property
    def connections(self) -> List[Connection]:
        """
        :return: A list of all connections in the world.
        """
        return list(self.kinematic_structure.edges())

    @property
    def controlled_connections(self) -> Set[ActiveConnection]:
        """
        A subset of the robot's connections that are controlled by a controller.
        """
        return set(
            connection
            for connection in self.connections
            if isinstance(connection, ActiveConnection)
            and connection.has_hardware_interface
        )

    # %% Add Stuff to World
    def add_connection(
        self, connection: Connection, handle_duplicates: bool = False
    ) -> None:
        """
        Add a connection and the entities it connects to the world.

        :param connection: The connection to add.
        """
        connection.add_to_world(self)
        dofs_without_world = [dof for dof in connection.dofs if dof._world is None]
        for dof in dofs_without_world:
            self.add_degree_of_freedom(dof)

        self._add_kinematic_structure_entity_if_not_in_world(connection.parent)
        self._add_kinematic_structure_entity_if_not_in_world(connection.child)

        self._add_connection(connection)

    def _add_kinematic_structure_entity_if_not_in_world(self, entity: KinematicStructureEntity):
        try:
            self.get_kinematic_structure_entity_by_name(entity.name)
        except WorldEntityNotFoundError:
            self.add_kinematic_structure_entity(entity)

    @atomic_world_modification(modification=AddConnectionModification)
    def _add_connection(self, connection: Connection):
        """
        Adds a connection to the kinematic structure.

        The method updates the connection instance to associate it with the current
        world instance and reflects the connection in the kinematic structure.
        Do not call this function directly, use add_connection instead.

        :param connection: The connection to be added to the kinematic structure.
        """
        connection._world = self
        self.kinematic_structure.add_edge(
            connection.parent.index, connection.child.index, connection
        )

    def add_body(
        self, body: KinematicStructureEntity, handle_duplicates: bool = False
    ) -> Optional[int]:
        return self.add_kinematic_structure_entity(body, handle_duplicates)

    def add_kinematic_structure_entity(
        self,
        kinematic_structure_entity: KinematicStructureEntity,
        handle_duplicates: bool = False,
    ) -> int:
        """
        Add a kinematic_structure_entity to the world if it does not exist already.

        :param kinematic_structure_entity: The kinematic_structure_entity to add.
        :param handle_duplicates: If True, the kinematic_structure_entity will be added under a different name, if
        the name already exists. If False, an error will be raised. Default is False.
        :return: The index of the added kinematic_structure_entity.
        """
        logger.info(
            f"Trying to add kinematic_structure_entity with name {kinematic_structure_entity.name}"
        )

        if (
            kinematic_structure_entity._world is not None
            and kinematic_structure_entity._world is not self
        ):
            raise AlreadyBelongsToAWorldError(
                world=kinematic_structure_entity._world,
                type_trying_to_add=KinematicStructureEntity,
            )

        try:
            self.get_kinematic_structure_entity_by_name(kinematic_structure_entity.name)
            if not handle_duplicates:
                raise DuplicateKinematicStructureEntityError(
                    [kinematic_structure_entity.name]
                )
            kinematic_structure_entity.name.name = (
                kinematic_structure_entity.name.name
                + f"_{id_generator(kinematic_structure_entity)}"
            )
        except WorldEntityNotFoundError:
            pass
        return self._add_kinematic_structure_entity(kinematic_structure_entity)

    @atomic_world_modification(modification=AddKinematicStructureEntityModification)
    def _add_kinematic_structure_entity(
        self, kinematic_structure_entity: KinematicStructureEntity
    ) -> int:
        """
        Add a kinematic_structure_entity to the world.
        Do not call this function directly, use add_kinematic_structure_entity instead.

        :param kinematic_structure_entity: The kinematic_structure_entity to add.
        :return: The index of the added kinematic_structure_entity.
        """
        index = kinematic_structure_entity.index = self.kinematic_structure.add_node(
            kinematic_structure_entity
        )
        kinematic_structure_entity._world = self
        return index

    def add_degree_of_freedom(self, dof: DegreeOfFreedom) -> None:
        """
        Adds degree of freedom in the world.
        This is used to register DoFs that are not created by the world, but are part of the world model.
        :param dof: The degree of freedom to register.
        """
        if dof._world is self and dof in self.degrees_of_freedom:
            return
        if dof._world is not None:
            raise AlreadyBelongsToAWorldError(
                world=dof._world, type_trying_to_add=DegreeOfFreedom
            )
        self._add_degree_of_freedom(dof)

    @atomic_world_modification(modification=AddDegreeOfFreedomModification)
    def _add_degree_of_freedom(self, dof: DegreeOfFreedom) -> None:
        """
        Adds a degree of freedom to the current system and initializes its state.

        This method modifies the internal state of the system by adding a new
        degree of freedom (DOF). It sets the initial position of the DOF based
        on its configured lower and upper position limits, ensuring it respects
        both constraints. The DOF is then added to the list of degrees of freedom
        in the system.

        :param dof: The degree of freedom to be added to the system.
        :return: None
        """
        dof._world = self
        self.state.add_degree_of_freedom_position(dof)
        self.degrees_of_freedom.append(dof)

    def add_semantic_annotation(
        self, semantic_annotation: SemanticAnnotation, exists_ok: bool = False
    ) -> None:
        """
        Adds a semantic annotation to the current list of semantic annotations if it doesn't already exist. Ensures
        that the `semantic_annotation` is associated with the current instance and maintains the
        integrity of unique semantic annotation names.

        :param semantic_annotation: The semantic annotation instance to be added. Its name must be unique within
            the current context.
        :param exists_ok: Whether to raise an error or not when a semantic annotation already exists.

        :raises AddingAnExistingSemanticAnnotationError: If exists_ok is False and a semantic annotation with the same name and type already exists.
        """
        try:
            self.get_semantic_annotation_by_name(semantic_annotation.name)
            if not exists_ok:
                raise AddingAnExistingSemanticAnnotationError(semantic_annotation)
        except WorldEntityNotFoundError:
            self._add_semantic_annotation(semantic_annotation)

    @atomic_world_modification(modification=AddSemanticAnnotationModification)
    def _add_semantic_annotation(self, semantic_annotation: SemanticAnnotation):
        """
        The atomic method that adds a semantic annotation to the current list of semantic annotations.
        """
        semantic_annotation._world = self
        self.semantic_annotations.append(semantic_annotation)

    # %% Remove Stuff from World
    def remove_connection(self, connection: Connection) -> None:
        """
        Removes a connection and deletes the corresponding degree of freedom, if it was only used by this connection.
        Might create disconnected entities, so make sure to add a new connection or delete the child kinematic_structure_entity.

        :param connection: The connection to be removed
        """
        remaining_dofs = {
            dof
            for remaining_connection in self.connections
            if remaining_connection != connection
            for dof in remaining_connection.dofs
        }

        removed_dofs = set(connection.dofs) - remaining_dofs

        with self.modify_world():
            for dof in removed_dofs:
                self.remove_degree_of_freedom(dof)
            self._remove_connection(connection)

    @atomic_world_modification(modification=RemoveConnectionModification)
    def _remove_connection(self, connection: Connection) -> None:
        try:
            self.kinematic_structure.remove_edge(
                connection.parent.index, connection.child.index
            )
        except NoEdgeBetweenNodes:
            pass
        connection._world = None
        connection.index = None

    def remove_kinematic_structure_entity(
        self, kinematic_structure_entity: KinematicStructureEntity
    ) -> None:
        """
        Removes a kinematic_structure_entity from the world.

        :param kinematic_structure_entity: The kinematic_structure_entity to remove.
        """
        if (
            kinematic_structure_entity._world is not self
            or kinematic_structure_entity.index is None
        ):
            logger.debug(
                "Trying to remove an kinematic_structure_entity that is not part of this world."
            )
            return

        self._remove_kinematic_structure_entity(kinematic_structure_entity)

    @atomic_world_modification(modification=RemoveBodyModification)
    def _remove_kinematic_structure_entity(
        self, kinematic_structure_entity: KinematicStructureEntity
    ) -> None:
        """
        Removes a kinematic_structure_entity from the world.

        Do not call this function directly, use `remove_kinematic_structure_entity` instead.

        :param kinematic_structure_entity: The kinematic_structure_entity to remove.
        """
        self.kinematic_structure.remove_node(kinematic_structure_entity.index)
        kinematic_structure_entity._world = None
        kinematic_structure_entity.index = None

    def remove_degree_of_freedom(self, dof: DegreeOfFreedom) -> None:
        if dof._world is not self:
            logger.debug("Trying to remove an dof that is not part of this world.")
            return
        self._remove_degree_of_freedom(dof)

    @atomic_world_modification(modification=RemoveDegreeOfFreedomModification)
    def _remove_degree_of_freedom(self, dof: DegreeOfFreedom) -> None:
        dof._world = None
        self.degrees_of_freedom.remove(dof)
        del self.state[dof.name]

    def remove_semantic_annotation(
        self, semantic_annotation: SemanticAnnotation
    ) -> None:
        """
        Removes a semantic annotation from the current list of semantic annotations if it exists.

        :param semantic_annotation: The semantic annotation instance to be removed.
        """
        try:
            existing_semantic_annotation = self.get_semantic_annotation_by_name(
                semantic_annotation.name
            )
            if existing_semantic_annotation == semantic_annotation:
                self._remove_semantic_annotation(semantic_annotation)
            else:
                raise ValueError(
                    "The provided semantic annotation instance does not match the existing semantic annotation with the same name."
                )
        except WorldEntityNotFoundError:
            logger.debug(
                f"semantic annotation {semantic_annotation.name} not found in the world. No action taken."
            )

    @atomic_world_modification(modification=RemoveSemanticAnnotationModification)
    def _remove_semantic_annotation(self, semantic_annotation: SemanticAnnotation):
        """
        The atomic method that removes a semantic annotation from the current list of semantic annotations.
        """
        self.semantic_annotations.remove(semantic_annotation)
        semantic_annotation._world = None

    # %% Other Atomic World Modifications
    @atomic_world_modification(modification=SetDofHasHardwareInterface)
    def set_dofs_has_hardware_interface(
        self, dofs: Iterable[DegreeOfFreedom], value: bool
    ):
        """
        Sets whether the specified degrees of freedom (DOFs) have a hardware interface or not.

        This method allows controlling the presence of a hardware interface for multiple
        DOFs at once. The modification is atomic, ensuring that all DOFs are updated as
        a single operation and the state remains consistent. The method iterates through
        the given DOFs and updates their `has_hardware_interface` attribute to the provided
        value.

        :param dofs: An iterable collection of DegreeOfFreedom instances whose
                     `has_hardware_interface` attribute is to be updated.
        :param value: A boolean value indicating whether the DOFs should have a hardware
                      interface (True) or not (False).
        """
        for dof in dofs:
            dof.has_hardware_interface = value

    # %% Getter
    def get_connection(
        self, parent: KinematicStructureEntity, child: KinematicStructureEntity
    ) -> Connection:
        """
        Retrieves the connection between a parent and child kinematic_structure_entity in the kinematic structure.
        """
        return self.kinematic_structure.get_edge_data(parent.index, child.index)

    def get_connections_by_type(
        self, connection_type: Type[GenericConnection]
    ) -> List[GenericConnection]:
        """
        Retrieves the connections of a given type.

        :param connection_type: The type of connection to retrieve.
        :return: A list of connections of the given type.
        """
        return self._get_world_entity_by_type_from_iterable(
            connection_type, self.connections
        )

    def get_semantic_annotations_by_type(
        self, semantic_annotation_type: Type[GenericSemanticAnnotation]
    ) -> List[GenericSemanticAnnotation]:
        """
        Retrieves all semantic annotations of a specific type from the world.

        :param semantic_annotation_type: The class (type) of the semantic annotations to search for.
        :return: A list of `SemanticAnnotation` objects that match the given type.
        """
        return self._get_world_entity_by_type_from_iterable(
            semantic_annotation_type, self.semantic_annotations
        )

    def get_kinematic_structure_entity_by_type(
        self, entity_type: Type[GenericKinematicStructureEntity]
    ) -> List[GenericKinematicStructureEntity]:
        """
        Retrieves all kinematic structure entities of a specific type from the world.

        :param entity_type: The class (type) of the kinematic structure entities to search for.
        :return: A list of `KinematicStructureEntity` objects that match the given type.
        """
        return self._get_world_entity_by_type_from_iterable(
            entity_type, self.kinematic_structure_entities
        )

    @staticmethod
    def _get_world_entity_by_type_from_iterable(
        world_entity_type: Type[GenericWorldEntity], iterable: Iterable[WorldEntity]
    ) -> List[GenericWorldEntity]:
        """
        Helper function to retrieve all world entities of a specific type from an iterable.
        :param world_entity_type: The type of the world entity.
        :param iterable: The iterable to search for the world entity, for example self.connections or self.kinematic_structure_entities.
        :return: A list of `WorldEntity` objects that match the given type.
        """
        return [entity for entity in iterable if isinstance(entity, world_entity_type)]

    def get_semantic_annotation_by_name(
        self, name: Union[str, PrefixedName]
    ) -> SemanticAnnotation:
        semantic_annotation: SemanticAnnotation = (
            self._get_world_entity_by_name_from_iterable(
                name, self.semantic_annotations
            )
        )
        return semantic_annotation

    def get_kinematic_structure_entity_by_name(
        self, name: Union[str, PrefixedName]
    ) -> KinematicStructureEntity:
        kse: KinematicStructureEntity = self._get_world_entity_by_name_from_iterable(
            name, self.kinematic_structure_entities
        )
        return kse

    def get_body_by_name(self, name: Union[str, PrefixedName]) -> Body:
        body: Body = self._get_world_entity_by_name_from_iterable(name, self.bodies)
        return body

    def get_degree_of_freedom_by_name(
        self, name: Union[str, PrefixedName]
    ) -> DegreeOfFreedom:
        dof: DegreeOfFreedom = self._get_world_entity_by_name_from_iterable(
            name, self.degrees_of_freedom
        )
        return dof

    def get_connection_by_name(self, name: Union[str, PrefixedName]) -> Connection:
        connection: Connection = self._get_world_entity_by_name_from_iterable(
            name, self.connections
        )
        return connection

    @staticmethod
    def _get_world_entity_by_name_from_iterable(
        name: Union[str, PrefixedName], world_entity_iterable: Iterable[WorldEntity]
    ) -> WorldEntity:
        """
        Retrieve a world entity by its name from an iterable of world entities.
        This iterable would, for example, be self.connections or self.kinematic_structure_entities.
        This method accepts either a string or a `PrefixedName` instance.
        It searches through the provided iterable and returns the one
        that matches the given name. If the `PrefixedName` contains a prefix,
        the method ensures the name, including the prefix, matches an existing
        world entity. Otherwise, it only considers the unprefixed name. If more than
        one connection matches the specified name, or if no connection is found,
        an exception is raised.
        :param name: The name of the connection to retrieve. Can be a string or
            a `PrefixedName` instance. If a prefix is included in `PrefixedName`,
            it will be used for matching.
        :param world_entity_iterable:
        :return: The `WorldEntity` object that matches the given name.
        :raises WorldEntityNotFoundError: If no world entity with the given name exists.
        :raises DuplicateWorldEntityError: If multiple world entities with the given name exist.
        """
        original_name = name
        prefix = None
        if isinstance(name, PrefixedName):
            prefix = name.prefix if name.prefix is not None else None
            name = name.name

        matches = [
            world_entity
            for world_entity in world_entity_iterable
            if world_entity.name.name == name
        ]

        if prefix is not None:
            matches = [
                world_entity
                for world_entity in matches
                if world_entity.name.prefix == prefix
            ]

        if len(matches) == 0:
            raise WorldEntityNotFoundError(original_name)
        if len(matches) > 1:
            raise DuplicateWorldEntityError(matches)

        return matches[0]

    # %% World Merging
    def merge_world_at_pose(self, other: World, pose: cas.TransformationMatrix) -> None:
        """
        Merge another world into the existing one, creates a 6DoF connection between the root of this world and the root
        of the other world.
        :param other: The world to be added.
        :param pose: world_root_T_other_root, the pose of the other world's root with respect to the current world's root
        """
        with self.modify_world():
            root_connection = Connection6DoF(
                parent=self.root, child=other.root, _world=self
            )
            self.merge_world(other, root_connection)
            root_connection.origin = pose

    def merge_world(
        self,
        other: World,
        root_connection: Connection = None,
        handle_duplicates: bool = False,
    ) -> None:
        """
        Merge a world into the existing one by merging degrees of freedom, states, connections, and bodies.
        This removes all bodies and connections from `other`.

        :param other: The world to be added.
        :param root_connection: If provided, this connection will be used to connect the two worlds. Otherwise, a new Connection6DoF will be created
        :param handle_duplicates: If True, bodies and semantic annotations with duplicate names will be renamed. If False, an error will be raised if duplicates are found.
        :return: None
        """
        assert other is not self, "Cannot merge a world with itself."

        with self.modify_world():
            old_state = deepcopy(other.state)
            self_root = self.root
            other_root = other.root
            with other.modify_world():
                self._merge_dofs_of_world(other)
                self._merge_connections_of_world(other, handle_duplicates)
                self._remove_kinematic_structure_entities_of_world(other)
                self._merge_semantic_annotations_of_world(other, handle_duplicates)

            if not root_connection and self_root:
                root_connection = Connection6DoF(
                    parent=self_root, child=other_root, _world=self
                )

            if root_connection:
                self.add_connection(
                    root_connection, handle_duplicates=handle_duplicates
                )

            for dof_name in old_state.keys():
                self.state[dof_name] = old_state[dof_name]

    def _merge_dofs_of_world(self, other: World):
        for dof in other.degrees_of_freedom.copy():
            other.remove_degree_of_freedom(dof)
            self.add_degree_of_freedom(dof)

    def _merge_connections_of_world(self, other: World, handle_duplicates: bool):
        other_root = other.root
        for connection in other.connections:
            other.remove_kinematic_structure_entity(connection.parent)
            other.remove_kinematic_structure_entity(connection.child)
            self.add_connection(connection, handle_duplicates=handle_duplicates)
        else:
            other.remove_kinematic_structure_entity(other_root)
            self._add_kinematic_structure_entity_if_not_in_world(other_root)

    @staticmethod
    def _remove_kinematic_structure_entities_of_world(other: World):
        other_kse_with_world = [
            kse for kse in other.kinematic_structure_entities if kse._world is not None
        ]
        for kinematic_structure_entity in other_kse_with_world:
            other.remove_kinematic_structure_entity(kinematic_structure_entity)

    def _merge_semantic_annotations_of_world(
        self, other: World, handle_duplicates: bool
    ):
        other_semantic_annotations = [
            semantic_annotation for semantic_annotation in other.semantic_annotations
        ]
        for semantic_annotation in other_semantic_annotations:
            other.remove_semantic_annotation(semantic_annotation)
            self.add_semantic_annotation(
                semantic_annotation, exists_ok=handle_duplicates
            )

    # %% Subgraph Targeting
    def get_connections_of_branch(
        self, root: KinematicStructureEntity
    ) -> List[Connection]:
        """
        Collect all connections that are below root in the tree.

        :param root: The root body of the branch
        :return: List of all connections in the subtree rooted at the given body
        """

        # Create a custom visitor to collect connections
        class ConnectionCollector(rustworkx.visit.DFSVisitor):
            def __init__(self, world: "World"):
                self.world = world
                self.connections = []

            def tree_edge(self, edge: Tuple[int, int, Connection]):
                """Called for each tree edge during DFS traversal"""
                self.connections.append(edge[2])  # edge[2] is the connection

        visitor = ConnectionCollector(self)
        self._travel_branch(root, visitor)
        return visitor.connections

    def get_bodies_of_branch(
        self, root: KinematicStructureEntity
    ) -> List[KinematicStructureEntity]:
        """
        Collect all bodies that are below root in the tree.

        :param root: The root body of the branch
        :return: List of all bodies in the subtree rooted at the given body (including the root)
        """

        # Create a custom visitor to collect bodies
        class BodyCollector(rustworkx.visit.DFSVisitor):
            def __init__(self, world: World):
                self.world = world
                self.bodies = []

            def discover_vertex(self, node_index: int, time: int) -> None:
                """Called when a vertex is first discovered during DFS traversal"""
                body = self.world.kinematic_structure[node_index]
                self.bodies.append(body)

        visitor = BodyCollector(self)
        self._travel_branch(root, visitor)

        return visitor.bodies

    def get_direct_child_bodies_with_collision(
        self, connection: Connection
    ) -> Set[Body]:
        """
        Collect all child Bodies until a movable connection is found.

        :param connection: The connection from the kinematic structure whose child bodies will be traversed.
        :return: A set of Bodies that are moved directly by only this connection.
        """

        class BodyCollector(rx.visit.DFSVisitor):
            def __init__(self, world: World):
                self.world = world
                self.bodies = set()

            def discover_vertex(self, node_index: int, time: int) -> None:
                body = self.world.kinematic_structure[node_index]
                if isinstance(body, Body) and body.has_collision():
                    self.bodies.add(body)

            def tree_edge(self, args: Tuple[int, int, Connection]) -> None:
                parent_index, child_index, e = args
                if (
                    isinstance(e, ActiveConnection)
                    and e.has_hardware_interface
                    and not e.frozen_for_collision_avoidance
                ):
                    raise rx.visit.PruneSearch()

        visitor = BodyCollector(self)
        self._travel_branch(connection.child, visitor)

        return visitor.bodies

    def _travel_branch(
        self,
        root_kinematic_structure_entity: KinematicStructureEntity,
        visitor: rustworkx.visit.DFSVisitor,
    ) -> None:
        """
        Apply a DFS Visitor to a subtree of the kinematic structure.

        :param root_kinematic_structure_entity: Starting point of the search
        :param visitor: This visitor to apply.
        """
        rx.dfs_search(
            self.kinematic_structure, [root_kinematic_structure_entity.index], visitor
        )

    def move_branch(
        self,
        branch_root: KinematicStructureEntity,
        new_parent: KinematicStructureEntity,
    ) -> None:
        """
        Destroys the connection between branch_root and its parent, and moves it to a new parent using a new connection
        of the same type. The pose of body with respect to root stays the same.

        :param branch_root: The root of the branch to be moved.
        :param new_parent: The new parent of the branch.
        """
        new_connection = None
        new_parent_T_root = self.compute_forward_kinematics(new_parent, branch_root)
        old_connection = branch_root.parent_connection

        match old_connection:
            case FixedConnection():
                new_connection = FixedConnection(
                    parent=new_parent,
                    child=branch_root,
                    _world=self,
                    parent_T_connection_expression=new_parent_T_root,
                )

            case Connection6DoF():
                new_connection = Connection6DoF(
                    parent=new_parent,
                    child=branch_root,
                    _world=self,
                )

            case _:
                raise ValueError(
                    "The branch root must be connected to a Connection6DoF or FixedConnection."
                )

        with self.modify_world():
            self.add_connection(new_connection)
            self.remove_connection(old_connection)

        if isinstance(new_connection, Connection6DoF):
            new_connection.origin = new_parent_T_root

    def copy_subgraph_to_new_world(self, new_root: KinematicStructureEntity) -> World:
        """
        Copies the subgraph of the kinematic structure from the root body to a new world and removes it from the old world.

        :param new_root: The root body of the subgraph to be copied.
        :return: A new `World` instance containing the copied subgraph.
        """
        with self.modify_world():
            new_world = World(name=self.name)
            child_bodies = self.compute_descendent_child_kinematic_structure_entities(
                new_root
            )
            child_body_parent_connections = [
                body.parent_connection for body in child_bodies
            ]

            with new_world.modify_world():
                self.remove_kinematic_structure_entity(new_root)
                new_world.add_kinematic_structure_entity(new_root)

                for body in child_bodies:
                    self.remove_kinematic_structure_entity(body)
                    new_world.add_kinematic_structure_entity(body)

                for connection in child_body_parent_connections:
                    self.remove_connection(connection)
                    new_world.add_connection(connection)

        return new_world

    # %% Change Notifications
    def notify_state_change(self) -> None:
        """
        If you have changed the state of the world, call this function to trigger necessary events and increase
        the state version.
        """
        if not self.empty:
            self._fk_computer.recompute()
        self.state._notify_state_change()

    def _notify_model_change(self) -> None:
        """
        Notifies the system of a model change and updates necessary states, caches,
        and forward kinematics expressions while also triggering registered callbacks
        for model changes.
        """
        # if not self.world_is_being_modified:
        self._compile_forward_kinematics_expressions()
        self.clear_all_lru_caches()
        self.notify_state_change()

        self._world_model_manager.update_model_version_and_notify_callbacks()

        for callback in self.state.state_change_callbacks:
            callback.update_previous_world_state()

        self.validate()
        self._collision_pair_manager.disable_non_robot_collisions()
        self._collision_pair_manager.disable_collisions_for_adjacent_bodies()

    def delete_orphaned_dofs(self):
        actual_dofs = {
            dof for connection in self.connections for dof in connection.dofs
        }
        self.degrees_of_freedom = list(actual_dofs)

    # %% Kinematic Structure Computations
    @lru_cache(maxsize=None)
    def compute_descendent_child_kinematic_structure_entities(
        self, kinematic_structure_entity: KinematicStructureEntity
    ) -> List[KinematicStructureEntity]:
        """
        Computes all child entities of a given KinematicStructureEntity in the world recursively.
        :param kinematic_structure_entity: The KinematicStructureEntity for which to compute children.
        :return: A list of all child KinematicStructureEntities.
        """
        children = self.compute_child_kinematic_structure_entities(
            kinematic_structure_entity
        )
        for child in children:
            children.extend(
                self.compute_descendent_child_kinematic_structure_entities(child)
            )
        return children

    @lru_cache(maxsize=None)
    def compute_child_kinematic_structure_entities(
        self, kinematic_structure_entity: KinematicStructureEntity
    ) -> List[KinematicStructureEntity]:
        """
        Computes the child entities of a given KinematicStructureEntity in the world.
        :param kinematic_structure_entity: The KinematicStructureEntity for which to compute children.
        :return: A list of child KinematicStructureEntities.
        """
        return list(
            self.kinematic_structure.successors(kinematic_structure_entity.index)
        )

    @lru_cache(maxsize=None)
    def compute_parent_connection(
        self, kinematic_structure_entity: KinematicStructureEntity
    ) -> Optional[Connection]:
        """
        Computes the parent connection of a given KinematicStructureEntity in the world.
        :param kinematic_structure_entity: The entityKinematicStructureEntity for which to compute the parent connection.
        :return: The parent connection of the given KinematicStructureEntity.
        """
        parent = self.compute_parent_kinematic_structure_entity(
            kinematic_structure_entity
        )
        if parent is None:
            return None

        return self.kinematic_structure.get_edge_data(
            parent.index,
            kinematic_structure_entity.index,
        )

    @lru_cache(maxsize=None)
    def compute_parent_kinematic_structure_entity(
        self, kinematic_structure_entity: KinematicStructureEntity
    ) -> Optional[KinematicStructureEntity]:
        """
        Computes the parent KinematicStructureEntity of a given KinematicStructureEntity in the world.
        :param kinematic_structure_entity: The KinematicStructureEntity for which to compute the parent KinematicStructureEntity.
        :return: The parent KinematicStructureEntity of the given KinematicStructureEntity.
         If the given KinematicStructureEntity is the root, None is returned.
        """
        parent = self.kinematic_structure.predecessors(kinematic_structure_entity.index)
        if len(parent) == 0:
            return None
        return parent[0]

    @lru_cache(maxsize=None)
    def compute_chain_of_connections(
        self, root: KinematicStructureEntity, tip: KinematicStructureEntity
    ) -> List[Connection]:
        """
        Computes the chain of connections between root and tip. Can handle chains that start and end anywhere in the tree.
        """
        entity_chain = self.compute_chain_of_kinematic_structure_entities(root, tip)
        return [
            self.get_connection(entity_chain[i], entity_chain[i + 1])
            for i in range(len(entity_chain) - 1)
        ]

    @lru_cache(maxsize=None)
    def compute_chain_of_kinematic_structure_entities(
        self, root: KinematicStructureEntity, tip: KinematicStructureEntity
    ) -> List[KinematicStructureEntity]:
        """
        Computes the chain between root and tip. Can handle chains that start and end anywhere in the tree.
        """
        if root == tip:
            return [root]
        shortest_paths = rx.all_shortest_paths(
            self.kinematic_structure, root.index, tip.index, as_undirected=False
        )

        if len(shortest_paths) == 0:
            raise rx.NoPathFound(f"No path found from {root} to {tip}")

        return [self.kinematic_structure[index] for index in shortest_paths[0]]

    def is_controlled_connection_in_chain(
        self, root: KinematicStructureEntity, tip: KinematicStructureEntity
    ) -> bool:
        root_part, tip_part = self.compute_split_chain_of_connections(root, tip)
        connections = root_part + tip_part
        return any(
            isinstance(connection, ActiveConnection)
            and connection.has_hardware_interface
            and not connection.frozen_for_collision_avoidance
            for connection in connections
        )

    def is_body_controlled(self, body: KinematicStructureEntity) -> bool:
        root_part, tip_part = self.compute_split_chain_of_connections(self.root, body)
        connections = root_part + tip_part
        return any(
            isinstance(c, ActiveConnection)
            and c.has_hardware_interface
            and not c.frozen_for_collision_avoidance
            for c in connections
        )

    def compute_chain_reduced_to_controlled_joints(
        self, root: KinematicStructureEntity, tip: KinematicStructureEntity
    ) -> Tuple[KinematicStructureEntity, KinematicStructureEntity]:
        """
        Removes root and tip links until they are both connected with a controlled connection.
        Useful for implementing collision avoidance.

        1. Compute the kinematic chain of bodies between root and tip.
        2. Remove all entries from link_a downward until one is connected with a connection from this semantic annotation.
        2. Remove all entries from link_b upward until one is connected with a connection from this semantic annotation.

        :param root: start of the chain
        :param tip: end of the chain
        :return: start and end link of the reduced chain
        """
        downward_chain, upward_chain = self.compute_split_chain_of_connections(
            root=root, tip=tip
        )
        chain = downward_chain + upward_chain

        condition_for_controlled_connection = (
            lambda conn: isinstance(conn, ActiveConnection)
            and conn.has_hardware_interface
            and not conn.frozen_for_collision_avoidance
        )

        new_root = next(
            (conn for conn in chain if condition_for_controlled_connection(conn)), None
        )
        if new_root is None:
            raise KeyError(
                f"no controlled connection in chain between {root} and {tip}"
            )

        new_tip = next(
            (
                conn
                for conn in reversed(chain)
                if condition_for_controlled_connection(conn)
            ),
            None,
        )
        if new_tip is None:
            raise KeyError(
                f"no controlled connection in chain between {root} and {tip}"
            )

        # if new_root is in the downward chain, we need to "flip" it by returning its child
        new_root_body = new_root.parent if new_root in upward_chain else new_root.child

        # if new_tip is in the downward chain, we need to "flip" it by returning its parent
        new_tip_body = new_tip.parent if new_tip in downward_chain else new_tip.child
        return new_root_body, new_tip_body

    @lru_cache(maxsize=None)
    def compute_split_chain_of_connections(
        self, root: KinematicStructureEntity, tip: KinematicStructureEntity
    ) -> Tuple[List[Connection], List[Connection]]:
        """
        Computes split chains of connections between 'root' and 'tip' bodies. Returns tuple of two Connection lists:
        (root->common ancestor, tip->common ancestor). Returns empty lists if root==tip.

        :param root: The starting `KinematicStructureEntity` object for the chain of connections.
        :param tip: The ending `KinematicStructureEntity` object for the chain of connections.
        :return: A tuple of two lists: the first list contains `Connection` objects from the `root` to
            the common ancestor, and the second list contains `Connection` objects from the `tip` to the
            common ancestor.
        """
        if root == tip:
            return [], []
        root_chain, common_ancestor, tip_chain = (
            self.compute_split_chain_of_kinematic_structure_entities(root, tip)
        )
        root_chain.append(common_ancestor[0])
        tip_chain.insert(0, common_ancestor[0])

        root_connections = [
            self.get_connection(root_chain[i + 1], root_chain[i])
            for i in range(len(root_chain) - 1)
        ]

        tip_connections = [
            self.get_connection(tip_chain[i], tip_chain[i + 1])
            for i in range(len(tip_chain) - 1)
        ]
        return root_connections, tip_connections

    @lru_cache(maxsize=None)
    def compute_split_chain_of_kinematic_structure_entities(
        self, root: KinematicStructureEntity, tip: KinematicStructureEntity
    ) -> Tuple[
        List[KinematicStructureEntity],
        List[KinematicStructureEntity],
        List[KinematicStructureEntity],
    ]:
        """
        Computes the chain between root and tip. Can handle chains that start and end anywhere in the tree.
        :param root: The root KinematicStructureEntity to start the chain from
        :param tip: The tip KinematicStructureEntity to end the chain at
        :return: tuple containing
                    1. chain from root to the common ancestor (excluding common ancestor)
                    2. list containing just the common ancestor
                    3. chain from common ancestor to tip (excluding common ancestor)
        """
        if root == tip:
            return [], [root], []

        # Paths from the tree's true root to each endpoint (inclusive).
        root_path = self.compute_chain_of_kinematic_structure_entities(self.root, root)
        tip_path = self.compute_chain_of_kinematic_structure_entities(self.root, tip)

        # Find the first index where the paths diverge.
        max_common_index = min(len(root_path), len(tip_path))
        i = next(
            (
                index
                for index, (root_path_entity, tip_path_entity) in enumerate(
                    zip(root_path, tip_path)
                )
                if root_path_entity != tip_path_entity
            ),
            max_common_index,
        )

        # The last common ancestor is the last common node.
        common_ancestor = root_path[i - 1]

        # 1) From `root` up to (but not including) last common ancestor, in order starting at `root`.
        #    `root_path[i:]` is the common_ancestor->root tail; reverse it to start at `root`.
        up_from_root = list(reversed(root_path[i:]))

        # 3) From just below last common ancestor down to `tip` (already in CA->tip order, excluding CA).
        down_to_tip = tip_path[i:]

        return up_from_root, [common_ancestor], down_to_tip

    # %% Forward Kinematics
    def _compile_forward_kinematics_expressions(self) -> None:
        """
        Traverse the kinematic structure and compile forward kinematics expressions for fast evaluation.
        """

        if self.empty:
            return

        new_fks = ForwardKinematicsVisitor(self)
        self._travel_branch(self.root, new_fks)
        new_fks.compile_forward_kinematics()
        self._fk_computer = new_fks

    def compute_forward_kinematics(
        self, root: KinematicStructureEntity, tip: KinematicStructureEntity
    ) -> cas.TransformationMatrix:
        """
        Compute the forward kinematics from the root KinematicStructureEntity to the tip KinematicStructureEntity.

        Calculate the transformation matrix representing the pose of the
        tip KinematicStructureEntity relative to the root KinematicStructureEntity.

        :param root: Root KinematicStructureEntity, for which the kinematics are computed.
        :param tip: Tip KinematicStructureEntity, to which the kinematics are computed.
        :return: Transformation matrix representing the relative pose of the tip KinematicStructureEntity with respect to the root KinematicStructureEntity.
        """
        return self._fk_computer.compute_forward_kinematics(root, tip)

    def compute_forward_kinematics_np(
        self, root: KinematicStructureEntity, tip: KinematicStructureEntity
    ) -> NpMatrix4x4:
        """
        Compute the forward kinematics from the root KinematicStructureEntity to the tip KinematicStructureEntity, root_T_tip and return it as a 4x4 numpy ndarray.

        Calculate the transformation matrix representing the pose of the
        tip KinematicStructureEntity relative to the root KinematicStructureEntity, expressed as a numpy ndarray.

        :param root: Root KinematicStructureEntity, for which the kinematics are computed.
        :param tip: Tip KinematicStructureEntity, to which the kinematics are computed.
        :return: Transformation matrix representing the relative pose of the tip KinematicStructureEntity with respect to the root KinematicStructureEntity.
        """
        return self._fk_computer.compute_forward_kinematics_np(root, tip).copy()

    # May I delete this? its not used anywhere @simon
    # def compute_forward_kinematics_of_all_collision_bodies(self) -> np.ndarray:
    #     """
    #     Computes a 4 by X matrix, with the forward kinematics of all collision bodies stacked on top each other.
    #     The entries are sorted by name of body.
    #     """
    #     return self._fk_computer.collision_fks

    # %% Inverse Kinematics
    def compute_inverse_kinematics(
        self,
        root: KinematicStructureEntity,
        tip: KinematicStructureEntity,
        target: cas.TransformationMatrix,
        dt: float = 0.05,
        max_iterations: int = 200,
        translation_velocity: float = 0.2,
        rotation_velocity: float = 0.2,
    ) -> Dict[DegreeOfFreedom, float]:
        """
        Compute inverse kinematics using quadratic programming.

        :param root: Root KinematicStructureEntity of the kinematic chain
        :param tip: Tip KinematicStructureEntity of the kinematic chain
        :param target: Desired tip pose relative to the root KinematicStructureEntity
        :param dt: Time step for integration
        :param max_iterations: Maximum number of iterations
        :param translation_velocity: Maximum translation velocity
        :param rotation_velocity: Maximum rotation velocity
        :return: Dictionary mapping DOF names to their computed positions
        """
        ik_solver = InverseKinematicsSolver(self)
        return ik_solver.solve(
            root,
            tip,
            target,
            dt,
            max_iterations,
            translation_velocity,
            rotation_velocity,
        )

    # %% World Utils
    def clear(self):
        """
        Clears all stored data and resets the state of the instance.
        """
        with self.modify_world():
            kse = self.kinematic_structure_entities
            for body in kse:
                self.remove_kinematic_structure_entity(body)

            self.semantic_annotations.clear()
            self.degrees_of_freedom.clear()
            self.state = WorldState()

    @property
    def empty(self):
        """
        :return: Returns True if the world contains no kinematic_structure_entities, else False.
        """
        return len(self.kinematic_structure_entities) == 0

    def clear_all_lru_caches(self):
        for method_name in dir(self):
            method = getattr(self, method_name, None)
            cache_clear = getattr(method, "cache_clear", None)
            if callable(cache_clear):
                cache_clear()

    def transform(
        self,
        spatial_object: cas.GenericSpatialType,
        target_frame: KinematicStructureEntity,
    ) -> cas.GenericSpatialType:
        """
        Transform a given spatial object from its reference frame to a target frame.

        Calculate the transformation from the reference frame of the provided
        spatial object to the specified target frame. Apply the transformation
        differently depending on the type of the spatial object:

        - If the object is a Quaternion, compute its rotation matrix, transform it, and
          convert back to a Quaternion.
        - For other types, apply the transformation matrix directly.

        :param spatial_object: The spatial object to be transformed.
        :param target_frame: The target KinematicStructureEntity frame to which the spatial object should
            be transformed.
        :return: The spatial object transformed to the target frame. If the input object
            is a Quaternion, the returned object is a Quaternion. Otherwise, it is the
            transformed spatial object.
        """
        target_frame_T_reference_frame = self.compute_forward_kinematics(
            root=target_frame, tip=spatial_object.reference_frame
        )
        if isinstance(spatial_object, cas.Quaternion):
            reference_frame_R = spatial_object.to_rotation_matrix()
            target_frame_R = target_frame_T_reference_frame @ reference_frame_R
            return target_frame_R.to_quaternion()

        return target_frame_T_reference_frame @ spatial_object

    def __deepcopy__(self, memo):
        # TODO surely this can be optimized, but I need to fully understand this first because fucking it up will make jonas big mad
        new_world = World(name=self.name)
        body_mapping = {}
        dof_mapping = {}
        with new_world.modify_world():
            for body in self.bodies:
                new_body = Body(
                    visual=body.visual,
                    collision=body.collision,
                    name=body.name,
                )
                new_world.add_kinematic_structure_entity(new_body)
                body_mapping[body] = new_body
            for dof in self.degrees_of_freedom:
                new_dof = DegreeOfFreedom(
                    name=dof.name,
                    lower_limits=dof.lower_limits,
                    upper_limits=dof.upper_limits,
                )
                new_world.add_degree_of_freedom(new_dof)
                dof_mapping[dof] = new_dof
            for connection in self.connections:
                con_factory = ConnectionFactory.from_connection(connection)
                con_factory.create(new_world)
            for dof in self.degrees_of_freedom:
                new_world.state[dof.name] = self.state[dof.name].data
        return new_world

    # %% Associations
    def load_collision_srdf(self, file_path: str):
        self._collision_pair_manager.load_collision_srdf(file_path)

    def modify_world(self) -> WorldModelUpdateContextManager:
        return WorldModelUpdateContextManager(self)

    def reset_state_context(self) -> ResetStateContextManager:
        return ResetStateContextManager(self)

    def get_world_model_manager(self) -> WorldModelManager:
        return self._world_model_manager

    @cached_property
    def collision_detector(self) -> CollisionDetector:
        """
        A collision detector for the world.
        :return: A collision detector for the world.
        """
        return TrimeshCollisionDetector(self)

    @cached_property
    def ray_tracer(self) -> RayTracer:
        """
        A ray tracer for the world.
        :return: A ray tracer for the world.
        """
        return RayTracer(self)

    @property
    def forward_kinematic_manager(self):
        return self._get_forward_kinematics_manager()

    @lru_cache(maxsize=None)
    def _get_forward_kinematics_manager(self):
        return ForwardKinematicsVisitor(self)

    def apply_control_commands(
        self, commands: np.ndarray, dt: float, derivative: Derivatives
    ) -> None:
        """
        Updates the state of a system by applying control commands at a specified derivative level,
        followed by backward integration to update lower derivatives.

        :param commands: Control commands to be applied at the specified derivative
            level. The array length must match the number of free variables
            in the system.
        :param dt: Time step used for the integration of lower derivatives.
        :param derivative: The derivative level to which the control commands are
            applied.
        :return: None
        """
        assert len(commands) == len(
            self.degrees_of_freedom
        ), f"Commands length {len(commands)} does not match number of free variables {len(self.degrees_of_freedom)}"

        self.state.set_derivative(derivative, commands)

        for i in range(derivative - 1, -1, -1):
            self.state.set_derivative(
                i,
                self.state.get_derivative(i) + self.state.get_derivative(i + 1) * dt,
            )
        for connection in self.connections:
            if isinstance(connection, HasUpdateState):
                connection.update_state(dt)
        self.notify_state_change()

    def set_positions_1DOF_connection(
        self, new_state: Dict[ActiveConnection1DOF, float]
    ) -> None:
        """
        Set the positions of 1DOF connections and notify the world of the state change.
        """
        for connection, value in new_state.items():
            connection.position = value
        self.notify_state_change()

    def get_world_state_symbols(self) -> List[cas.Symbol]:
        """
        Constructs and returns a list of symbols representing the state of the system. The state
        is defined in terms of positions, velocities, accelerations, and jerks for each degree
        of freedom specified in the current state.

        :raises KeyError: If a degree of freedom defined in the state does not exist in
            the `degrees_of_freedom`.
        :returns: A combined list of symbols corresponding to the positions, velocities,
            accelerations, and jerks for each degree of freedom in the state.
        """
        positions = [
            self.get_degree_of_freedom_by_name(v_name).symbols.position
            for v_name in self.state
        ]
        velocities = [
            self.get_degree_of_freedom_by_name(v_name).symbols.velocity
            for v_name in self.state
        ]
        accelerations = [
            self.get_degree_of_freedom_by_name(v_name).symbols.acceleration
            for v_name in self.state
        ]
        jerks = [
            self.get_degree_of_freedom_by_name(v_name).symbols.jerk
            for v_name in self.state
        ]
        return positions + velocities + accelerations + jerks
