from dataclasses import dataclass

from krrood.entity_query_language.predicate import Symbol
from typing_extensions import Optional, Dict, Any, Self

from krrood.adapters.json_serializer import SubclassJSONSerializer, to_json, from_json


@dataclass
class PrefixedName(Symbol, SubclassJSONSerializer):
    name: str
    prefix: Optional[str] = None

    def __hash__(self):
        return hash((self.prefix, self.name))

    def __str__(self):
        if self.prefix is None:
            return self.name
        return f"{self.prefix}/{self.name}"

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        return self.prefix == other.prefix and self.name == other.name

    def to_json(self) -> Dict[str, Any]:
        return {
            **super().to_json(),
            "name": to_json(self.name),
            "prefix": to_json(self.prefix),
        }

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        return cls(
            name=from_json(data["name"], **kwargs),
            prefix=from_json(data["prefix"], **kwargs),
        )

    def __lt__(self, other):
        return str(self) < str(other)

    def __le__(self, other):
        return str(self) <= str(other)

    def __gt__(self, other):
        return str(self) > str(other)

    def __ge__(self, other):
        return str(self) >= str(other)
