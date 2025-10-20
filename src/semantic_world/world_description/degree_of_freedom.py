from __future__ import annotations

from dataclasses import dataclass, field
from typing_extensions import Dict, Any

from random_events.utils import SubclassJSONSerializer

from ..datastructures.prefixed_name import PrefixedName
from ..spatial_types import spatial_types as cas
from ..spatial_types.derivatives import Derivatives, DerivativeMap
from ..spatial_types.symbol_manager import symbol_manager
from .world_entity import WorldEntity


@dataclass
class DegreeOfFreedom(WorldEntity, SubclassJSONSerializer):
    """
    A class representing a degree of freedom in a world model with associated derivatives and limits.

    This class manages a variable that can freely change within specified limits, tracking its position,
    velocity, acceleration, and jerk. It maintains symbolic representations for each derivative order
    and provides methods to get and set limits for these derivatives.
    """

    lower_limits: DerivativeMap[float] = field(default_factory=DerivativeMap)
    upper_limits: DerivativeMap[float] = field(default_factory=DerivativeMap)
    """
    Lower and upper bounds for each derivative
    """

    symbols: DerivativeMap[cas.Symbol] = field(
        default_factory=DerivativeMap, init=False
    )
    """
    Symbolic representations for each derivative
    """

    has_hardware_interface: bool = False
    """
    Whether this DOF is linked to a controller and can therefore respond to control commands.

    E.g. the caster wheels of a PR2 have dofs, but they are not directly controlled. 
    Instead a the omni drive connection is directly controlled and a low level controller translates these commands
    to commands for the caster wheels.

    A door hinge also has a dof that cannot be controlled.
    """

    def __post_init__(self):
        self.lower_limits = self.lower_limits or DerivativeMap()
        self.upper_limits = self.upper_limits or DerivativeMap()

    def create_and_register_symbols(self):
        assert self._world is not None
        for derivative in Derivatives.range(Derivatives.position, Derivatives.jerk):
            s = symbol_manager.register_symbol_provider(
                f"{self.name}_{derivative}",
                lambda d=derivative: self._world.state[self.name][d],
            )
            self.symbols.data[derivative] = s

    def has_position_limits(self) -> bool:
        try:
            lower_limit = self.lower_limits.position
            upper_limit = self.upper_limits.position
            return lower_limit is not None or upper_limit is not None
        except KeyError:
            return False

    def __hash__(self):
        return hash(id(self))

    def to_json(self) -> Dict[str, Any]:
        return {
            **super().to_json(),
            "lower_limits": self.lower_limits.to_json(),
            "upper_limits": self.upper_limits.to_json(),
            "name": self.name.to_json(),
        }

    @classmethod
    def _from_json(cls, data: Dict[str, Any]) -> DegreeOfFreedom:
        lower_limits = DerivativeMap.from_json(data["lower_limits"])
        upper_limits = DerivativeMap.from_json(data["upper_limits"])
        return cls(
            name=PrefixedName.from_json(data["name"]),
            lower_limits=lower_limits,
            upper_limits=upper_limits,
        )
