from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from enum import IntEnum
from typing_extensions import Generic, TypeVar, List, Optional, Dict, Any

from krrood.adapters.json_serializer import SubclassJSONSerializer, to_json, from_json

T = TypeVar("T")


class Derivatives(IntEnum):
    """
    Enumaration of interpretation for the order of derivativeson the spatial positions
    """

    position = 0
    velocity = 1
    acceleration = 2
    jerk = 3
    snap = 4
    crackle = 5
    pop = 6

    @classmethod
    def range(cls, start: Derivatives, stop: Derivatives, step: int = 1):
        """
        Includes stop!
        """
        return [item for item in cls if start <= item <= stop][::step]


@dataclass
class DerivativeMap(Generic[T], SubclassJSONSerializer):
    """
    A container class that maps derivatives (position, velocity, acceleration, jerk) to values of type T.

    This class provides a structured way to store and access different orders of derivatives
    using properties. Each derivative order can hold a value of type T or None.

    Type Parameters:
        T: The type of values stored for each derivative order.

    Attributes:
        data (List[Optional[T]]): Internal list storing the derivative values, initialized with None values.
    """

    data: List[Optional[T]] = field(default_factory=lambda: [None] * len(Derivatives))
    """
    Internal list storing the derivative values, initialized with None values.
    Order corresponds to the order of the Derivatives enum.
    """

    def __hash__(self):
        return hash(tuple(self.data))

    @property
    def position(self) -> T:
        return self.data[Derivatives.position]

    @position.setter
    def position(self, value: T):
        self.data[Derivatives.position] = value

    @property
    def velocity(self) -> T:
        return self.data[Derivatives.velocity]

    @velocity.setter
    def velocity(self, value: T):
        self.data[Derivatives.velocity] = value

    @property
    def acceleration(self) -> T:
        return self.data[Derivatives.acceleration]

    @acceleration.setter
    def acceleration(self, value: T):
        self.data[Derivatives.acceleration] = value

    @property
    def jerk(self) -> T:
        return self.data[Derivatives.jerk]

    @jerk.setter
    def jerk(self, value: T):
        self.data[Derivatives.jerk] = value

    def to_json(self) -> Dict[str, Any]:
        return {**super().to_json(), "data": to_json(self.data)}

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> DerivativeMap[T]:
        return cls(from_json(data=data["data"], **kwargs))

    def __mul__(self, other: float) -> DerivativeMap[T]:
        result = DerivativeMap()
        for i, v in enumerate(self.data):
            if v is not None:
                result.data[i] = v * other
        return result

    def __add__(self, other: float) -> DerivativeMap[T]:
        result = DerivativeMap()
        for i, v in enumerate(self.data):
            if v is not None:
                result.data[i] = v + other
        return result
