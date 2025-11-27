from __future__ import annotations

import re
from dataclasses import dataclass
from dataclasses import field
from typing import Optional, Type

from typing_extensions import List

from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Table,
    Container,
    IsPerceivable,
)
from semantic_digital_twin.semantic_annotations.mixins import (
    HasBody,
    HasSupportingSurface,
    Furniture,
)
from ...world_description.world_entity import SemanticAnnotation


class AmbiguousNameError(ValueError):
    """Raised when more than one semantic annotation class matches a given name with the same score."""


class UnresolvedNameError(ValueError):
    """Raised when no semantic annotation class matches a given name."""


@dataclass
class ProcthorResolver:
    """Central resolver that deterministically maps a ProcTHOR name to exactly one class."""

    classes: List[Type[HasBody]] = field(default_factory=list)

    def resolve(self, name: str) -> Optional[Type[SemanticAnnotation]]:
        """
        Resolve a given name to a class based on the number of matching tokens
        with the class name tokens or synonyms. The method preprocesses the
        name by removing numbers and splitting it into tokens, and then compares
        these tokens with the corresponding data in the available classes to
        find the best match.

        :param name: The name to resolve, represented as a string.
        :return: The class with the best match to the given name, or None if no matches are found.
        """
        # remove all numbers from the name
        name_tokens = set(n.lower() for n in re.sub(r"\d+", "", name).split("_"))
        possible_results = []
        for cls in self.classes:
            matches = cls.class_name_tokens().intersection(
                name_tokens
            ) or cls._synonyms.intersection(name_tokens)
            possible_results.append((cls, matches))

        if len(possible_results) == 0:
            return None
        # sort by max number of matches
        possible_results = sorted(
            possible_results, key=lambda x: len(x[1]), reverse=True
        )

        # if there are no matches, don't choose a class
        if len(possible_results[0][1]) == 0:
            return None
        best_cls, best_matches = possible_results[0]
        return best_cls


@dataclass(eq=False)
class Bottle(Container):
    """
    Abstract class for bottles.
    """


@dataclass(eq=False)
class Statue(HasBody): ...


@dataclass(eq=False)
class SoapBottle(Bottle):
    """
    A soap bottle.
    """


@dataclass(eq=False)
class WineBottle(Bottle):
    """
    A wine bottle.
    """


@dataclass(eq=False)
class MustardBottle(Bottle):
    """
    A mustard bottle.
    """


@dataclass(eq=False)
class DrinkingContainer(Container, HasBody): ...


@dataclass(eq=False)
class Cup(DrinkingContainer, IsPerceivable):
    """
    A cup.
    """


@dataclass(eq=False)
class Mug(DrinkingContainer):
    """
    A mug.
    """


@dataclass(eq=False)
class CookingContainer(Container, HasBody): ...


@dataclass(eq=False)
class Lid(HasBody): ...


@dataclass(eq=False)
class Pan(CookingContainer):
    """
    A pan.
    """


@dataclass(eq=False)
class PanLid(Lid):
    """
    A pan lid.
    """


@dataclass(eq=False)
class Pot(CookingContainer):
    """
    A pot.
    """


@dataclass(eq=False)
class PotLid(Lid):
    """
    A pot lid.
    """


@dataclass(eq=False)
class Plate(HasBody, HasSupportingSurface):
    """
    A plate.
    """


@dataclass(eq=False)
class Bowl(HasBody, IsPerceivable, HasSupportingSurface):
    """
    A bowl.
    """


# Food Items
@dataclass(eq=False)
class Food(HasBody): ...


@dataclass(eq=False)
class TunaCan(Food):
    """
    A tuna can.
    """


@dataclass(eq=False)
class Bread(Food):
    """
    Bread.
    """

    _synonyms = {
        "bumpybread",
        "whitebread",
        "loafbread",
        "honeybread",
        "grainbread",
    }


@dataclass(eq=False)
class CheezeIt(Food):
    """
    Some type of cracker.
    """


@dataclass(eq=False)
class Pringles(Food):
    """
    Pringles chips
    """


@dataclass(eq=False)
class GelatinBox(Food):
    """
    Gelatin box.
    """


@dataclass(eq=False)
class TomatoSoup(Food):
    """
    Tomato soup.
    """


@dataclass(eq=False)
class Produce(Food):
    """
    In American English, produce generally refers to fresh fruits and vegetables intended to be eaten by humans.
    """

    pass


@dataclass(eq=False)
class Tomato(Produce):
    """
    A tomato.
    """


@dataclass(eq=False)
class Lettuce(Produce):
    """
    Lettuce.
    """


@dataclass(eq=False)
class Apple(Produce):
    """
    An apple.
    """


@dataclass(eq=False)
class Banana(Produce):
    """
    A banana.
    """


@dataclass(eq=False)
class Orange(Produce):
    """
    An orange.
    """


@dataclass(eq=False)
class CoffeeTable(Table):
    """
    A coffee table.
    """


@dataclass(eq=False)
class DiningTable(Table):
    """
    A dining table.
    """


@dataclass(eq=False)
class SideTable(Table):
    """
    A side table.
    """


@dataclass(eq=False)
class Desk(Table):
    """
    A desk.
    """


@dataclass(eq=False)
class Chair(HasBody, Furniture):
    """
    Abstract class for chairs.
    """


@dataclass(eq=False)
class OfficeChair(Chair):
    """
    An office chair.
    """


@dataclass(eq=False)
class Armchair(Chair):
    """
    An armchair.
    """


@dataclass(eq=False)
class ShelvingUnit(HasBody, Furniture):
    """
    A shelving unit.
    """


@dataclass(eq=False)
class Bed(HasBody, Furniture):
    """
    A bed.
    """


@dataclass(eq=False)
class Sofa(HasBody, Furniture):
    """
    A sofa.
    """


@dataclass(eq=False)
class Sink(HasBody):
    """
    A sink.
    """


@dataclass(eq=False)
class Kettle(CookingContainer): ...


@dataclass(eq=False)
class Decor(HasBody): ...


@dataclass(eq=False)
class WallDecor(Decor):
    """
    Wall decorations.
    """


@dataclass(eq=False)
class Cloth(HasBody): ...


@dataclass(eq=False)
class Poster(WallDecor):
    """
    A poster.
    """


@dataclass(eq=False)
class WallPanel(HasBody):
    """
    A wall panel.
    """


@dataclass(eq=False)
class Potato(Produce): ...


@dataclass(eq=False)
class GarbageBin(Container):
    """
    A garbage bin.
    """


@dataclass(eq=False)
class Drone(HasBody): ...


@dataclass(eq=False)
class ProcthorBox(Container): ...


@dataclass(eq=False)
class Houseplant(HasBody):
    """
    A houseplant.
    """


@dataclass(eq=False)
class SprayBottle(HasBody):
    """
    A spray bottle.
    """


@dataclass(eq=False)
class Vase(HasBody):
    """
    A vase.
    """


@dataclass(eq=False)
class Book(HasBody):
    """
    A book.
    """

    book_front: Optional[BookFront] = None


@dataclass(eq=False)
class BookFront(HasBody): ...


@dataclass(eq=False)
class SaltPepperShaker(HasBody):
    """
    A salt and pepper shaker.
    """


@dataclass(eq=False)
class Cuttlery(HasBody): ...


@dataclass(eq=False)
class Fork(Cuttlery):
    """
    A fork.
    """


@dataclass(eq=False)
class Knife(Cuttlery):
    """
    A butter knife.
    """


@dataclass(eq=False)
class Spoon(Cuttlery, IsPerceivable): ...


@dataclass(eq=False)
class Milk(Cuttlery, IsPerceivable): ...


@dataclass(eq=False)
class Pencil(HasBody):
    """
    A pencil.
    """


@dataclass(eq=False)
class Pen(HasBody):
    """
    A pen.
    """


@dataclass(eq=False)
class Baseball(HasBody):
    """
    A baseball.
    """


@dataclass(eq=False)
class LiquidCap(HasBody):
    """
    A liquid cap.
    """
