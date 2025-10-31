# ----------------------------------------------------------------------------------------------------------------------
# This script generates the ORM classes for the semantic_digital_twin package.
# Dataclasses can be mapped automatically to the ORM model
# using the ORMatic library, they just have to be registered in the classes list.
# Classes that are self_mapped and explicitly_mapped are already mapped in the model.py file. Look there for more
# information on how to map them.
# ----------------------------------------------------------------------------------------------------------------------
from __future__ import annotations

import os
from dataclasses import is_dataclass

import krrood.entity_query_language.orm.model
import trimesh
from krrood.class_diagrams import ClassDiagram
from krrood.entity_query_language.predicate import HasTypes, HasType, Symbol
from krrood.entity_query_language.symbol_graph import SymbolGraph
from krrood.ormatic.dao import AlternativeMapping
from krrood.ormatic.ormatic import ORMatic
from krrood.ormatic.utils import classes_of_module
from krrood.utils import recursive_subclasses

import semantic_digital_twin.orm.model
import semantic_digital_twin.robots.abstract_robot
import semantic_digital_twin.semantic_annotations.semantic_annotations
import semantic_digital_twin.world  # ensure the module attribute exists on the package
import semantic_digital_twin.world_description.degree_of_freedom
import semantic_digital_twin.world_description.geometry
import semantic_digital_twin.world_description.shape_collection
import semantic_digital_twin.world_description.world_entity
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_computations.forward_kinematics import (
    ForwardKinematicsVisitor,
)
from semantic_digital_twin.world import (
    ResetStateContextManager,
    WorldModelUpdateContextManager,
)
from semantic_digital_twin.world_description.connections import (
    FixedConnection,
    HasUpdateState,
)


# collect all semantic digital twin classes that should be mapped
all_classes = set(classes_of_module(semantic_digital_twin.orm.model))
all_classes |= set(
    classes_of_module(semantic_digital_twin.world_description.world_entity)
)
all_classes |= set(classes_of_module(semantic_digital_twin.world_description.geometry))
all_classes |= set(
    classes_of_module(semantic_digital_twin.world_description.shape_collection)
)
all_classes |= set(classes_of_module(semantic_digital_twin.world))
all_classes |= set(
    classes_of_module(semantic_digital_twin.datastructures.prefixed_name)
)

all_classes |= set(
    classes_of_module(semantic_digital_twin.world_description.connections)
)
all_classes |= set(
    classes_of_module(semantic_digital_twin.semantic_annotations.semantic_annotations)
)
all_classes |= set(
    classes_of_module(semantic_digital_twin.world_description.degree_of_freedom)
)
all_classes |= set(classes_of_module(semantic_digital_twin.robots.abstract_robot))


# remove classes that should not be mapped
all_classes -= {
    ResetStateContextManager,
    WorldModelUpdateContextManager,
    HasUpdateState,
    ForwardKinematicsVisitor,
}

# build the symbol graph
symbol_graph = SymbolGraph()

# collect all KRROOD classes
all_classes |= {c.clazz for c in symbol_graph.class_diagram.wrapped_classes}
all_classes |= {am.original_class() for am in recursive_subclasses(AlternativeMapping)}
all_classes |= set(classes_of_module(krrood.entity_query_language.symbol_graph))
all_classes |= {Symbol}

# remove classes that don't need persistence
all_classes -= {HasType, HasTypes}


# keep only dataclasses that are NOT AlternativeMapping subclasses
all_classes = {
    c for c in all_classes if is_dataclass(c) and not issubclass(c, AlternativeMapping)
}

# ensure we have the original classes of the mappings (ORMatic uses these)
all_classes |= {am.original_class() for am in recursive_subclasses(AlternativeMapping)}


def generate_orm():
    """
    Generate the ORM classes for the pycram package.
    """
    class_diagram = ClassDiagram(
        list(sorted(all_classes, key=lambda c: c.__name__, reverse=True))
    )

    instance = ORMatic(
        class_dependency_graph=class_diagram,
        type_mappings={trimesh.Trimesh: semantic_digital_twin.orm.model.TrimeshType},
        alternative_mappings=recursive_subclasses(AlternativeMapping),
    )

    instance.make_all_tables()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.abspath(
        os.path.join(script_dir, "..", "src", "semantic_digital_twin", "orm")
    )
    with open(os.path.join(path, "ormatic_interface.py"), "w") as f:
        instance.to_sqlalchemy_file(f)


if __name__ == "__main__":
    generate_orm()
