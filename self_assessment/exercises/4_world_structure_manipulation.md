---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.3
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

(world-structure-manipulation-exercise)=
# World Structure Manipulation

This exercise demonstrates adding and removing entities from the kinematic structure.

You will:
- Create a simple world with a passive 6DoF connection and an active revolute joint
- Remove the revolute connection and its child body in a single modification block

## 0. Setup

```{code-cell} ipython3
:tags: [remove-input]
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body
from semantic_digital_twin.world_description.connections import Connection6DoF, RevoluteConnection
from semantic_digital_twin.world_description.degree_of_freedom import DegreeOfFreedom
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types.spatial_types import Vector3
from entity_query_language import entity, an, let, symbolic_mode, the
```

## 1. Create a simple kinematic structure
Your goal:
- Create a `World` and three bodies named `root`, `base`, and `body`
- Add a passive `Connection6DoF` between `root` (parent) and `base` (child)
- Add a `RevoluteConnection` between `base` (parent) and `body` (child) using a Z axis on `base`
- Store the world in a variable named `world` and the revolute connection in `revolute_connection`

```{code-cell} ipython3
:tags: [exercise]
# TODO: create the world and connections described above
world: World = ...
root: Body = ...
base: Body = ...
body: Body = ...
root_C_base: Connection6DoF = ...
revolute_connection_dof: DegreeOfFreedom = ...
base_C_body: RevoluteConnection = ...


```

```{code-cell} ipython3
:tags: [example-solution]
world = World()
root = Body(name=PrefixedName(name="root", prefix="world"))
base = Body(name=PrefixedName("base"))
body = Body(name=PrefixedName("body"))
root_C_base = Connection6DoF(parent=root, child=base)
revolute_connection_dof = DegreeOfFreedom(name=PrefixedName("joint_z"))
base_C_body = RevoluteConnection(
        parent=base,
        child=body,
        dof_name=revolute_connection_dof.name,
        axis=Vector3.Z(reference_frame=base),
    )

with world.modify_world():
    world.add_connection(root_C_base)    
    world.add_connection(base_C_body)
```

## 2. Remove a connection and its child
Your goal:
- In a single `with world.modify_world():` block, remove the revolute connection and the now disconnected child body `body`.

```{code-cell} ipython3
:tags: [exercise]
body = ...
base_C_body = ...
# TODO: remove the revolute connection and its child body in one modification block, without using the objects you defined above, but instead querying the world directly

```

```{code-cell} ipython3
:tags: [example-solution]

with symbolic_mode():
    base_C_body_query = the(entity(let(RevoluteConnection, world.connections)))

base_C_body = base_C_body_query.evaluate()
body = base_C_body.child

with world.modify_world():
    world.remove_connection(base_C_body)
    world.remove_kinematic_structure_entity(body)
```

```{code-cell} ipython3
:tags: [verify-solution, remove-input]
# After removal, the body must be gone and only one connection should remain
assert all(b is not body for b in world.bodies), "The child body `body` should be removed."
assert len(world.connections) == 1, "There should be exactly one connection left (root -> base)."
```