from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from probabilistic_model.probabilistic_circuit.rx.helper import uniform_measure_of_event
from typing_extensions import List

from ..world_description.shape_collection import BoundingBoxCollection
from ..spatial_types import Point3
from ..datastructures.variables import SpatialVariables
from ..world_description.world_entity import SemanticAnnotation, Body, Region


@dataclass(eq=False)
class HasDrawers:
    """
    A mixin class for semantic annotations that have drawers.
    """

    drawers: List[Drawer] = field(default_factory=list, hash=False)


@dataclass(eq=False)
class HasDoors:
    """
    A mixin class for semantic annotations that have doors.
    """

    doors: List[Door] = field(default_factory=list, hash=False)


@dataclass(eq=False)
class Handle(SemanticAnnotation):
    body: Body


@dataclass(eq=False)
class Container(SemanticAnnotation):
    body: Body


@dataclass(eq=False)
class Fridge(SemanticAnnotation):
    """
    A semantic annotation representing a fridge that has a door and a body.
    """

    body: Body
    door: Door


@dataclass(eq=False)
class Table(SemanticAnnotation):
    """
    A semantic annotation that represents a table.
    """

    top: Body
    """
    The body that represents the table's top surface.
    """

    def points_on_table(self, amount: int = 100) -> List[Point3]:
        """
        Get points that are on the table.

        :amount: The number of points to return.
        :returns: A list of points that are on the table.
        """
        area_of_table = BoundingBoxCollection.from_shapes(self.top.collision)
        event = area_of_table.event
        p = uniform_measure_of_event(event)
        p = p.marginal(SpatialVariables.xy)
        samples = p.sample(amount)
        z_coordinate = np.full(
            (amount, 1), max([b.max_z for b in area_of_table]) + 0.01
        )
        samples = np.concatenate((samples, z_coordinate), axis=1)
        return [Point3(*s, reference_frame=self.top) for s in samples]


################################


@dataclass(eq=False)
class Components(SemanticAnnotation): ...


@dataclass(eq=False)
class Furniture(SemanticAnnotation): ...


@dataclass(eq=False)
class SupportingSurface(SemanticAnnotation):
    """
    A semantic annotation that represents a supporting surface.
    """

    region: Region
    """
    The region that represents the supporting surface.
    """


#################### subclasses von Components


@dataclass(eq=False)
class EntryWay(Components):
    body: Body


@dataclass(eq=False)
class Door(EntryWay):
    handle: Handle


@dataclass(eq=False)
class DoubleDoor(EntryWay):
    left_door: Door
    right_door: Door


@dataclass(eq=False)
class Drawer(Components):
    container: Container
    handle: Handle


############################### subclasses to Furniture
@dataclass(eq=False)
class Cabinet(Furniture):
    container: Container
    drawers: List[Drawer] = field(default_factory=list, hash=False)
    doors: List[Door] = field(default_factory=list)


@dataclass(eq=False)
class Dresser(Furniture):
    container: Container
    drawers: List[Drawer] = field(default_factory=list, hash=False)
    doors: List[Door] = field(default_factory=list)


@dataclass(eq=False)
class Cupboard(Furniture):
    container: Container
    doors: List[Door] = field(default_factory=list)


@dataclass(eq=False)
class Wardrobe(Furniture):
    container: Container
    drawers: List[Drawer] = field(default_factory=list, hash=False)
    doors: List[Door] = field(default_factory=list)


class Floor(SupportingSurface): ...


@dataclass(eq=False)
class Room(SemanticAnnotation):
    """
    A semantic annotation that represents a closed area with a specific purpose
    """

    floor: Floor
    """
    The room's floor.
    """


@dataclass(eq=False)
class Wall(SemanticAnnotation):
    body: Body
    doors: List[Door] = field(default_factory=list)
