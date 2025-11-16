import uuid
from dataclasses import dataclass, field

from krrood.entity_query_language.predicate import Symbol
from typing_extensions import Optional, Dict, Any, Self

from krrood.adapters.json_serializer import SubclassJSONSerializer


@dataclass
class PrefixedName(Symbol, SubclassJSONSerializer):
    """
    A class that represents an entity with a name and an optional prefix.
    We have the assumption every PrefixedName can only exist once in a given world.
    PrefixedName.name may be duplicate, but PrefixedName.prefix is always unique.
    """

    name: str
    """
    The name of the entity.
    """

    prefix: Optional[str] = None
    """
    Unique identifier of the entity.
    """

    _unique_by_uuid: bool = field(init=False, default=False)

    def __hash__(self):
        return hash((self.prefix, self.name))

    def __str__(self):
        return f"{self.prefix}/{self.name}"

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        return self.prefix == other.prefix and self.name == other.name

    def to_json(self) -> Dict[str, Any]:
        return {**super().to_json(), "name": self.name, "prefix": self.prefix}

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        return cls(name=data["name"], prefix=data["prefix"])

    def __lt__(self, other):
        return str(self) < str(other)

    def __le__(self, other):
        return str(self) <= str(other)

    def __gt__(self, other):
        return str(self) > str(other)

    def __ge__(self, other):
        return str(self) >= str(other)

    def ensure_unique(self):
        """Appends a random UUIDv4 (~2¹²² possibilities) to the name."""
        if not self._unique_by_uuid:
            self.name = f"{self.name}_{uuid.uuid4()}"
            self._unique_by_uuid = True
