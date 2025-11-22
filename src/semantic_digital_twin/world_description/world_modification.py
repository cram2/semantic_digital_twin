from __future__ import annotations

from abc import abstractmethod, ABC
from dataclasses import dataclass, field

from krrood.adapters.json_serializer import SubclassJSONSerializer
from typing_extensions import (
    List,
    Dict,
    Any,
    Self,
    Callable,
    TYPE_CHECKING,
)

from .actuators import Actuator
from .degree_of_freedom import DegreeOfFreedom
from .world_entity import (
    KinematicStructureEntity,
    SemanticAnnotation,
    Connection,
    WorldEntity,
)
from ..adapters.world_entity_kwargs_tracker import (
    KinematicStructureEntityKwargsTracker,
)
from ..datastructures.prefixed_name import PrefixedName

if TYPE_CHECKING:
    from ..world import World


@dataclass
class UnknownWorldModification(Exception):
    """
    Raised when an unknown world modification is attempted.
    """

    call: Callable
    kwargs: Dict[str, Any]

    def __post_init__(self):
        super().__init__(
            " Make sure that world modifications are atomic and that every atomic modification is "
            "represented by exactly one subclass of WorldModelModification."
            "This module might be incomplete, you can help by expanding it."
        )


@dataclass
class WorldModelModification(SubclassJSONSerializer, ABC):
    """
    A record of a modification to the model (structure) of the world.
    This includes add/remove body and add/remove connection.

    All modifications are compared via the names of the objects they reference.

    This class is referenced by the `atomic_world_modification` decorator and should be used for a method that
    applies such a modification to the world.
    """

    @abstractmethod
    def apply(self, world: World):
        """
        Apply this change to the given world.

        :param world: The world to modify.
        """

    @classmethod
    @abstractmethod
    def from_kwargs(cls, kwargs: Dict[str, Any]) -> Self:
        """
        Factory to construct this change from the kwargs of its corresponding method in World decorated with
        `atomic_world_modification(modification=cls)`.

        :param kwargs: The kwargs of the function call.
        :return: A new instance.
        """
        raise NotImplementedError


@dataclass
class AddKinematicStructureEntityModification(WorldModelModification):
    """
    Addition of a body to the world.
    """

    kinematic_structure_entity: KinematicStructureEntity
    """
    The body that was added.
    """

    @classmethod
    def from_kwargs(cls, kwargs: Dict[str, Any]):
        return cls(kwargs["kinematic_structure_entity"])

    def apply(self, world: World):
        world.add_kinematic_structure_entity(self.kinematic_structure_entity)

    def to_json(self):
        return {**super().to_json(), "body": self.kinematic_structure_entity.to_json()}

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        return cls(
            kinematic_structure_entity=KinematicStructureEntity.from_json(
                data["body"], **kwargs
            )
        )

    def __eq__(self, other: Any) -> bool:
        return (
            self.kinematic_structure_entity.name
            == other.kinematic_structure_entity.name
        )


@dataclass
class RemoveBodyModification(WorldModelModification):
    """
    Removal of a body from the world.
    """

    body_name: PrefixedName
    """
    The name of the body that was removed.
    """

    @classmethod
    def from_kwargs(cls, kwargs: Dict[str, Any]):
        return cls(kwargs["kinematic_structure_entity"].name)

    def apply(self, world: World):
        world.remove_kinematic_structure_entity(
            world.get_kinematic_structure_entity_by_name(self.body_name)
        )

    def to_json(self) -> Dict[str, Any]:
        return {**super().to_json(), "body_name": self.body_name.to_json()}

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        return cls(body_name=PrefixedName.from_json(data["body_name"], **kwargs))


@dataclass
class AddConnectionModification(WorldModelModification):
    """
    Addition of a connection to the world.
    """

    connection: Connection
    """
    The connection that was added.
    """

    @classmethod
    def from_kwargs(cls, kwargs: Dict[str, Any]):
        return cls(kwargs["connection"])

    def apply(self, world: World):
        world.add_connection(self.connection)

    def to_json(self):
        return {
            **super().to_json(),
            "connection": self.connection.to_json(),
        }

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        return cls(connection=Connection.from_json(data["connection"], **kwargs))

    def __eq__(self, other):
        return (
            isinstance(other, AddConnectionModification)
            and self.connection.name == other.connection.name
        )


@dataclass
class RemoveConnectionModification(WorldModelModification):
    """
    Removal of a connection from the world.
    """

    connection_name: PrefixedName
    """
    The name of the connection that was removed.
    """

    @classmethod
    def from_kwargs(cls, kwargs: Dict[str, Any]):
        return cls(kwargs["connection"].name)

    def apply(self, world: World):
        world._remove_connection(world.get_connection_by_name(self.connection_name))

    def to_json(self):
        return {
            **super().to_json(),
            "connection_name": self.connection_name.to_json(),
        }

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        return cls(
            connection_name=PrefixedName.from_json(data["connection_name"], **kwargs)
        )


@dataclass
class AddDegreeOfFreedomModification(WorldModelModification):
    """
    Addition of a degree of freedom to the world.
    """

    dof: DegreeOfFreedom
    """
    The degree of freedom that was added.
    """

    @classmethod
    def from_kwargs(cls, kwargs: Dict[str, Any]):
        return cls(dof=kwargs["dof"])

    def apply(self, world: World):
        world.add_degree_of_freedom(self.dof)

    def to_json(self):
        return {
            **super().to_json(),
            "dof": self.dof.to_json(),
        }

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        return cls(dof=DegreeOfFreedom.from_json(data["dof"], **kwargs))

    def __eq__(self, other):
        return self.dof.name == other.dof.name


@dataclass
class RemoveDegreeOfFreedomModification(WorldModelModification):
    dof_name: PrefixedName

    @classmethod
    def from_kwargs(cls, kwargs: Dict[str, Any]):
        return cls(dof_name=kwargs["dof"].name)

    def apply(self, world: World):
        world.remove_degree_of_freedom(
            world.get_degree_of_freedom_by_name(self.dof_name)
        )

    def to_json(self):
        return {
            **super().to_json(),
            "dof": self.dof_name.to_json(),
        }

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        return cls(dof_name=PrefixedName.from_json(data["dof"], **kwargs))


@dataclass
class AddSemanticAnnotationModification(WorldModelModification):
    semantic_annotation: SemanticAnnotation

    @classmethod
    def from_kwargs(cls, kwargs: Dict[str, Any]):
        return cls(semantic_annotation=kwargs["semantic_annotation"])

    def apply(self, world: World):
        world.add_semantic_annotation(self.semantic_annotation)

    def to_json(self):
        return {
            **super().to_json(),
            "semantic_annotation": self.semantic_annotation.to_json(),
        }

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        return cls(
            semantic_annotation=SemanticAnnotation.from_json(
                data["semantic_annotation"], **kwargs
            )
        )


@dataclass
class RemoveSemanticAnnotationModification(WorldModelModification):
    semantic_annotation: SemanticAnnotation

    @classmethod
    def from_kwargs(cls, kwargs: Dict[str, Any]):
        return cls(semantic_annotation=kwargs["semantic_annotation"])

    def apply(self, world: World):
        world.remove_semantic_annotation(self.semantic_annotation)

    def to_json(self):
        return {
            **super().to_json(),
            "semantic_annotation": self.semantic_annotation.to_json(),
        }

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        return cls(
            semantic_annotation=SemanticAnnotation.from_json(
                data["semantic_annotation"], **kwargs
            )
        )


@dataclass
class AddActuatorModification(WorldModelModification):
    actuator: Actuator

    @classmethod
    def from_kwargs(cls, kwargs: Dict[str, Any]):
        return cls(actuator=kwargs["actuator"])

    def apply(self, world: World):
        world.add_actuator(self.actuator)

    def to_json(self):
        return {
            **super().to_json(),
            "actuator": self.actuator.to_json(),
        }

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        return cls(actuator=Actuator.from_json(data["actuator"], **kwargs))


@dataclass
class WorldModelModificationBlock(SubclassJSONSerializer):
    """
    A sequence of WorldModelModifications that were applied to the world within one `with world.modify_world()` context.
    """

    modifications: List[WorldModelModification] = field(default_factory=list)
    """
    The list of modifications to apply to the world.
    """

    def apply(self, world: World):
        with world.modify_world():
            for modification in self.modifications:
                modification.apply(world)

    def to_json(self):
        return {
            **super().to_json(),
            "modifications": [m.to_json() for m in self.modifications],
        }

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        return cls(
            modifications=[
                WorldModelModification.from_json(d, **kwargs)
                for d in data["modifications"]
            ],
        )

    def __iter__(self):
        return iter(self.modifications)

    def __getitem__(self, item):
        return self.modifications[item]

    def __len__(self):
        return len(self.modifications)

    def append(self, modification: WorldModelModification):
        self.modifications.append(modification)


@dataclass
class SetDofHasHardwareInterface(WorldModelModification):
    degree_of_freedom_names: List[PrefixedName]
    value: bool

    def apply(self, world: World):
        for dof_name in self.degree_of_freedom_names:
            world.get_degree_of_freedom_by_name(dof_name).has_hardware_interface = (
                self.value
            )

    @classmethod
    def from_kwargs(cls, kwargs: Dict[str, Any]) -> Self:
        dofs = kwargs["dofs"]
        degree_of_freedom_names = [dof.name for dof in dofs]
        return cls(
            degree_of_freedom_names=degree_of_freedom_names, value=kwargs["value"]
        )

    def to_json(self) -> Dict[str, Any]:
        return {
            **super().to_json(),
            "degree_of_freedom_names": [
                dof.to_json() for dof in self.degree_of_freedom_names
            ],
            "value": self.value,
        }

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        return cls(
            degree_of_freedom_names=[
                PrefixedName.from_json(dof, **kwargs)
                for dof in data["degree_of_freedom_names"]
            ],
            value=data["value"],
        )
