from abc import ABC
from dataclasses import dataclass
from functools import lru_cache
from typing_extensions import Dict, Any, Self, List

from krrood.adapters.json_serializer import SubclassJSONSerializer, to_json, from_json

from ...datastructures.prefixed_name import PrefixedName
from ...world_description.world_modification import (
    WorldModelModificationBlock,
)


@dataclass
class MetaData(SubclassJSONSerializer):
    """
    Class for data describing the origin of a message.
    """

    node_name: str
    """
    The name of the node that published this message
    """

    process_id: int
    """
    The id of the process that published this message
    """

    object_id: int
    """
    The id of the object in the process that issues this publishing call
    """

    @lru_cache(maxsize=None)
    def to_json(self) -> Dict[str, Any]:
        return {
            **super().to_json(),
            "node_name": to_json(self.node_name),
            "process_id": to_json(self.process_id),
            "object_id": to_json(self.object_id),
        }

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        return cls(
            node_name=from_json(data["node_name"], **kwargs),
            process_id=from_json(data["process_id"], **kwargs),
            object_id=from_json(data["object_id"], **kwargs),
        )

    def __hash__(self):
        return hash((self.node_name, self.process_id, self.object_id))


@dataclass
class Message(SubclassJSONSerializer, ABC):

    meta_data: MetaData
    """
    Message origin meta data.
    """

    def to_json(self) -> Dict[str, Any]:
        return {
            **super().to_json(),
            "meta_data": to_json(self.meta_data.to_json()),
        }


@dataclass
class WorldStateUpdate(Message):
    """
    Class describing the updates to the free variables of a world state.
    """

    prefixed_names: List[PrefixedName]
    """
    The prefixed names of the changed free variables.
    """

    states: List[float]
    """
    The states of the changed free variables.
    """

    def to_json(self) -> Dict[str, Any]:
        return {
            **super().to_json(),
            "prefixed_names": to_json(self.prefixed_names),
            "states": to_json(list(self.states)),
        }

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        return cls(
            meta_data=from_json(data["meta_data"], **kwargs),
            prefixed_names=from_json(data["prefixed_names"], **kwargs),
            states=from_json(data["states"], **kwargs),
        )


@dataclass
class ModificationBlock(Message):
    """
    Message describing the modifications done to a world.
    """

    modifications: WorldModelModificationBlock
    """
    The modifications done to a world.
    """

    def to_json(self) -> Dict[str, Any]:
        return {
            **super().to_json(),
            "modifications": to_json(self.modifications),
        }

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        return cls(
            meta_data=from_json(data["meta_data"], **kwargs),
            modifications=from_json(data["modifications"], **kwargs),
        )


@dataclass
class LoadModel(Message):
    """
    Message for requesting the loading of a model identified by its primary key.
    """

    primary_key: int
    """
    The primary key identifying the model to be loaded.
    """

    def to_json(self) -> Dict[str, Any]:
        return {
            **super().to_json(),
            "primary_key": self.primary_key,
        }

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        return cls(
            meta_data=MetaData.from_json(data["meta_data"], **kwargs),
            primary_key=data["primary_key"],
        )


@dataclass
class WorldModelSnapshot(SubclassJSONSerializer):
    """
    Snapshot containing the complete modification history and the latest world state.
    """

    modifications: List[WorldModelModificationBlock]
    """
    The ordered list of world model modification blocks.
    """

    prefixed_names: List[PrefixedName]
    """
    The names of the free variables contained in the state snapshot.
    """

    states: List[float]
    """
    The values of the free variables contained in the state snapshot.
    """

    def to_json(self) -> Dict[str, Any]:
        return {
            **super().to_json(),
            "modifications": to_json(self.modifications),
            "state": {
                "prefixed_names": to_json(self.prefixed_names),
                "states": to_json(list(self.states)),
            },
        }

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        state = data.get("state", {})
        return cls(
            modifications=[from_json(data["modifications"], **kwargs)],
            prefixed_names=[from_json(data["prefixed_names"], **kwargs)],
            states=state.get("states", []),
        )
