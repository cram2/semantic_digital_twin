import os
import time
import unittest
from typing import Optional

import sqlalchemy
from krrood.ormatic.utils import drop_database
from semantic_digital_twin.semantic_annotations.semantic_annotations import Handle, Door
from sqlalchemy import select
from sqlalchemy.orm import Session

from semantic_digital_twin.adapters.ros.world_synchronizer import (
    StateSynchronizer,
    ModelReloadSynchronizer,
    ModelSynchronizer,
)
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.orm.ormatic_interface import Base, WorldMappingDAO
from semantic_digital_twin.spatial_types import Vector3
from semantic_digital_twin.testing import rclpy_node
from semantic_digital_twin.utils import get_semantic_digital_twin_directory_root
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    FixedConnection,
    PrismaticConnection,
)
from semantic_digital_twin.world_description.degree_of_freedom import DegreeOfFreedom
from semantic_digital_twin.world_description.world_entity import Body


def create_dummy_world(w: Optional[World] = None) -> World:
    if w is None:
        w = World()
    b1 = Body(name=PrefixedName("b1"))
    b2 = Body(name=PrefixedName("b2"))
    with w.modify_world():
        w.add_connection(Connection6DoF.create_with_dofs(parent=b1, child=b2, world=w))
    return w


def test_state_synchronization(rclpy_node):

    w1 = create_dummy_world()
    w2 = create_dummy_world()

    synchronizer_1 = StateSynchronizer(
        node=rclpy_node,
        world=w1,
    )
    synchronizer_2 = StateSynchronizer(
        node=rclpy_node,
        world=w2,
    )

    # Allow time for publishers/subscribers to connect on unique topics
    time.sleep(0.2)

    w1.state.data[0, 0] = 1.0
    w1.notify_state_change()
    time.sleep(0.2)
    assert w1.state.data[0, 0] == 1.0
    assert w1.state.data[0, 0] == w2.state.data[0, 0]

    synchronizer_1.close()
    synchronizer_2.close()


def test_state_synchronization_world_model_change_after_init(rclpy_node):
    w1 = World()
    w2 = World()

    synchronizer_1 = StateSynchronizer(
        node=rclpy_node,
        world=w1,
    )
    create_dummy_world(w1)
    create_dummy_world(w2)
    synchronizer_2 = StateSynchronizer(
        node=rclpy_node,
        world=w2,
    )

    # Allow time for publishers/subscribers to connect on unique topics
    time.sleep(0.2)

    w1.state.data[0, 0] = 1.0
    w1.notify_state_change()
    time.sleep(0.2)
    assert w1.state.data[0, 0] == 1.0
    assert w1.state.data[0, 0] == w2.state.data[0, 0]

    synchronizer_1.close()
    synchronizer_2.close()


def test_model_reload(rclpy_node):

    engine = sqlalchemy.create_engine(
        "sqlite+pysqlite:///file::memory:?cache=shared",
        connect_args={"check_same_thread": False, "uri": True},
    )
    drop_database(engine)
    Base.metadata.create_all(engine)
    session_maker = sqlalchemy.orm.sessionmaker(bind=engine)
    session1 = session_maker()
    session2 = session_maker()

    w1 = create_dummy_world()
    w2 = World()

    synchronizer_1 = ModelReloadSynchronizer(
        rclpy_node,
        w1,
        session=session1,
    )
    synchronizer_2 = ModelReloadSynchronizer(
        rclpy_node,
        w2,
        session=session2,
    )

    synchronizer_1.publish_reload_model()
    time.sleep(1.0)
    assert len(w2.kinematic_structure_entities) == 2

    query = session1.scalars(select(WorldMappingDAO)).all()
    assert len(query) == 1
    assert w2.get_kinematic_structure_entity_by_name("b2")

    synchronizer_1.close()
    synchronizer_2.close()


def test_model_synchronization_body_only(rclpy_node):

    w1 = World(name="w1")
    w2 = World(name="w2")

    synchronizer_1 = ModelSynchronizer(
        node=rclpy_node,
        world=w1,
    )
    synchronizer_2 = ModelSynchronizer(
        node=rclpy_node,
        world=w2,
    )

    with w1.modify_world():
        new_body = Body(name=PrefixedName("b3"))
        w1.add_kinematic_structure_entity(new_body)
        assert len(w1.kinematic_structure_entities) == 1

    time.sleep(0.2)
    assert len(w1.kinematic_structure_entities) == 1
    assert len(w2.kinematic_structure_entities) == 1

    assert w2.get_kinematic_structure_entity_by_name("b3")

    synchronizer_1.close()
    synchronizer_2.close()


def test_model_synchronization_creation_only(rclpy_node):

    w1 = World(name="w1")
    w2 = World(name="w2")

    synchronizer_1 = ModelSynchronizer(
        node=rclpy_node,
        world=w1,
    )
    synchronizer_2 = ModelSynchronizer(
        node=rclpy_node,
        world=w2,
    )

    with w1.modify_world():
        b2 = Body(name=PrefixedName("b2"))
        w1.add_kinematic_structure_entity(b2)

        new_body = Body(name=PrefixedName("b3"))
        w1.add_kinematic_structure_entity(new_body)

        c = Connection6DoF.create_with_dofs(parent=b2, child=new_body, world=w1)
        w1.add_connection(c)
    time.sleep(0.1)
    assert len(w1.kinematic_structure_entities) == 2
    assert len(w2.kinematic_structure_entities) == 2
    assert len(w1.connections) == 1
    assert len(w2.connections) == 1

    synchronizer_1.close()
    synchronizer_2.close()


def test_model_synchronization_merge_full_world(rclpy_node):

    w1 = World(name="w1")
    w2 = World(name="w2")

    synchronizer_1 = ModelSynchronizer(
        node=rclpy_node,
        world=w1,
    )
    synchronizer_2 = ModelSynchronizer(
        node=rclpy_node,
        world=w2,
    )

    pr2_world = URDFParser.from_file(
        os.path.join(
            get_semantic_digital_twin_directory_root(os.getcwd()),
            "resources",
            "urdf",
            "pr2_kinematic_tree.urdf",
        )
    ).parse()

    def wait_for_sync(timeout=3.0, interval=0.05):
        start = time.time()
        while time.time() - start < timeout:
            body_names_1 = [body.name for body in w1.kinematic_structure_entities]
            body_names_2 = [body.name for body in w2.kinematic_structure_entities]
            if body_names_1 == body_names_2:
                return body_names_1, body_names_2
            time.sleep(interval)

        body_names_1 = [body.name for body in w1.kinematic_structure_entities]
        body_names_2 = [body.name for body in w2.kinematic_structure_entities]
        return body_names_1, body_names_2

    with w1.modify_world():
        new_body = Body(name=PrefixedName("b3"))
        w1.add_kinematic_structure_entity(new_body)

    fixed_connection = FixedConnection(child=new_body, parent=pr2_world.root)
    w1.merge_world(pr2_world, fixed_connection)

    body_names_1, body_names_2 = wait_for_sync()

    assert body_names_1 == body_names_2
    assert len(w1.kinematic_structure_entities) == len(w2.kinematic_structure_entities)

    w1_connection_names = [c.name for c in w1.connections]
    w2_connection_names = [c.name for c in w2.connections]
    assert w1_connection_names == w2_connection_names
    assert len(w1.connections) == len(w2.connections)
    assert len(w2.degrees_of_freedom) == len(w1.degrees_of_freedom)

    synchronizer_1.close()
    synchronizer_2.close()


def test_callback_pausing(rclpy_node):

    w1 = World(name="w1")
    w2 = World(name="w2")

    model_synchronizer_1 = ModelSynchronizer(node=rclpy_node, world=w1)
    model_synchronizer_2 = ModelSynchronizer(node=rclpy_node, world=w2)
    state_synchronizer_1 = StateSynchronizer(node=rclpy_node, world=w1)
    state_synchronizer_2 = StateSynchronizer(node=rclpy_node, world=w2)

    model_synchronizer_2.pause()
    state_synchronizer_2.pause()
    assert model_synchronizer_2._is_paused
    assert state_synchronizer_2._is_paused

    with w1.modify_world():
        b2 = Body(name=PrefixedName("b2"))
        w1.add_kinematic_structure_entity(b2)

        new_body = Body(name=PrefixedName("b3"))
        w1.add_kinematic_structure_entity(new_body)

        c = Connection6DoF.create_with_dofs(parent=b2, child=new_body, world=w1)
        w1.add_connection(c)

    time.sleep(0.1)
    assert len(model_synchronizer_2.missed_messages) == 1
    assert len(w1.kinematic_structure_entities) == 2
    assert len(w2.kinematic_structure_entities) == 0
    assert len(w1.connections) == 1
    assert len(w2.connections) == 0

    state_synchronizer_2.resume()
    model_synchronizer_2.resume()
    state_synchronizer_2.apply_missed_messages()
    model_synchronizer_2.apply_missed_messages()

    time.sleep(0.1)
    assert len(w1.kinematic_structure_entities) == 2
    assert len(w2.kinematic_structure_entities) == 2
    assert len(w1.connections) == 1
    assert len(w2.connections) == 1


def test_ChangeDifHasHardwareInterface(rclpy_node):

    w1 = World(name="w1")
    w2 = World(name="w2")

    synchronizer_1 = ModelSynchronizer(
        node=rclpy_node,
        world=w1,
    )
    synchronizer_2 = ModelSynchronizer(
        node=rclpy_node,
        world=w2,
    )

    with w1.modify_world():
        body1 = Body(name=PrefixedName("b1"))
        body2 = Body(name=PrefixedName("b2"))
        w1.add_kinematic_structure_entity(body1)
        w1.add_kinematic_structure_entity(body2)
        dof = DegreeOfFreedom(name=PrefixedName("dof"))
        w1.add_degree_of_freedom(dof)
        connection = PrismaticConnection(
            dof_name=dof.name, parent=body1, child=body2, axis=Vector3(1, 1, 1)
        )
        w1.add_connection(connection)
    assert len(w1.kinematic_structure_entities) == 2
    assert len(w1.connections) == 1

    time.sleep(0.2)
    assert len(w1.kinematic_structure_entities) == 2
    assert len(w2.kinematic_structure_entities) == 2
    assert len(w2.connections) == 1
    assert not w2.connections[0].dof.has_hardware_interface
    assert not w2.connections[0].dof.has_hardware_interface

    assert w2.get_kinematic_structure_entity_by_name("b2")

    with w1.modify_world():
        w1.set_dofs_has_hardware_interface(w1.degrees_of_freedom, True)

    time.sleep(0.2)
    assert w1.connections[0].dof.has_hardware_interface
    assert w2.connections[0].dof.has_hardware_interface

    synchronizer_1.close()
    synchronizer_2.close()


def test_semantic_annotation_modifications(rclpy_node):
    w1 = World(name="w1")
    w2 = World(name="w2")

    synchronizer_1 = ModelSynchronizer(
        node=rclpy_node,
        world=w1,
    )
    synchronizer_2 = ModelSynchronizer(
        node=rclpy_node,
        world=w2,
    )

    b1 = Body(name=PrefixedName("b1"))
    v1 = Handle(body=b1)
    v2 = Door(body=b1, handle=v1)

    with w1.modify_world():
        w1.add_body(b1)
        w1.add_semantic_annotation(v1)
        w1.add_semantic_annotation(v2)

    time.sleep(0.2)
    assert [sa.name for sa in w1.semantic_annotations] == [
        sa.name for sa in w2.semantic_annotations
    ]


if __name__ == "__main__":
    unittest.main()
