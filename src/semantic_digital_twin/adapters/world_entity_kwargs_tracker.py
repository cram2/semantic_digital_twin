from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import Dict, Optional, TYPE_CHECKING, Self, ClassVar, Any

from ..exceptions import KinematicStructureEntityNotInKwargs, WorldEntityNotFoundError

if TYPE_CHECKING:
    from ..world import World
    from ..datastructures.prefixed_name import PrefixedName
    from ..world_description.world_entity import KinematicStructureEntity


@dataclass
class KinematicStructureEntityKwargsTracker:
    """
    Keeps track of the kinematic structure entities that have been parsed in a from_json call from SubclassJSONSerializer.
    Usage:
        Top-level object must create a new tracker, optionally using a world instance if present, and pass it along:
            tracker = KinematicStructureEntityKwargsTracker.from_world(world)
            SubclassJSONSerializer.from_json(json_data, **tracker.create_from_json_kwargs())

        Objects that create kinematic structure entities:
            def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
                new_instance = cls(...)
                tracker = KinematicStructureEntityKwargsTracker.from_kwargs(kwargs)
                tracker.add_kinematic_structure_entity(new_instance)
                ...

        Objects that need kinematic structure entities for parsing:
            def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
                tracker = KinematicStructureEntityKwargsTracker.from_kwargs(kwargs)
                entity = tracker.get_kinematic_structure_entity(name_of_entity)
                ...
    """

    _kinematic_structure_entities: Dict[PrefixedName, KinematicStructureEntity] = field(
        default_factory=dict
    )
    _world: Optional[World] = field(init=False, default=None)
    __world_entity_tracker: ClassVar[str] = "__world_entity_tracker"

    @classmethod
    def from_kwargs(cls, from_json_kwargs) -> Self:
        """
        Retrieve the tracker from the kwargs, or initialize a new one if it doesn't exist.
        Adds itself to the kwargs so that it is available for future from_json calls.
        :param from_json_kwargs: the **kwargs of a from_json call.
        """
        tracker = from_json_kwargs.get(cls.__world_entity_tracker) or cls()
        tracker.add_to_kwargs(from_json_kwargs)
        return tracker

    @classmethod
    def from_world(cls, world: World) -> Self:
        """
        Create a new tracker from a world.
        :param world: A world instance that will be used as a backup to look for world entities.
        """
        tracker = cls()
        tracker._world = world
        return tracker

    def create_kwargs(self) -> Dict[str, Self]:
        """
        Creates a new kwargs that contains the tracker.
        The top-level object that calls from_json should add this to its kwargs.
        :return: A new kwargs dict with the tracker.
        """
        return {self.__world_entity_tracker: self}

    def add_to_kwargs(self, kwargs: Dict[str, Any]):
        """
        Adds the current instance to the provided keyword arguments dictionary,
        using a specific key internally defined within the instance.

        :param kwargs: A dictionary to which the current instance will be added.
                       The specific key is determined by the internal attribute of
                       the instance.
        :return: None
        """
        kwargs[self.__world_entity_tracker] = self

    def add_kinematic_structure_entity(
        self, kinematic_structure_entity: KinematicStructureEntity
    ):
        """
        Add a new kinematic structure entity to the tracker, to make it available for parsing in future from_json calls.
        """
        self._kinematic_structure_entities[kinematic_structure_entity.name] = (
            kinematic_structure_entity
        )

    def has_kinematic_structure_entity(self, name: PrefixedName) -> bool:
        try:
            self.get_kinematic_structure_entity(name)
            return True
        except KinematicStructureEntityNotInKwargs:
            return False

    def get_kinematic_structure_entity(
        self, name: PrefixedName
    ) -> KinematicStructureEntity:
        """
        Retrieve a kinematic structure entity by its name.

        This method attempts to find a kinematic structure entity from the internal
        collection. If the entity is not found and a world object is available,
        it will try to retrieve the entity by its name from the world object.

        :param name: The name of the kinematic structure entity to retrieve.
        :return: The kinematic structure entity corresponding to the specified name,
                 or None if not found.
        """
        kinematic_structure_entity = self._kinematic_structure_entities.get(name)
        if kinematic_structure_entity is not None:
            return kinematic_structure_entity
        if self._world is not None:
            try:
                return self._world.get_kinematic_structure_entity_by_name(name)
            except WorldEntityNotFoundError:
                pass
        raise KinematicStructureEntityNotInKwargs(name)
