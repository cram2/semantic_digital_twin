import json

from std_srvs.srv import Trigger

from semantic_digital_twin.adapters.ros.world_fetcher import (
    FetchWorldServer,
    fetch_world_from_service,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.semantic_annotations.semantic_annotations import Handle, Door
from semantic_digital_twin.testing import rclpy_node
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.world_entity import Body
from semantic_digital_twin.world_description.world_modification import (
    WorldModelModificationBlock,
)


def create_dummy_world():
    """
    Create a simple world with two bodies and a connection.
    """
    world = World()
    body_1 = Body(name=PrefixedName("body_1"))
    body_2 = Body(name=PrefixedName("body_2"))
    with world.modify_world():
        world.add_kinematic_structure_entity(body_1)
        world.add_kinematic_structure_entity(body_2)
        world.add_connection(Connection6DoF(body_1, body_2, _world=world))
    return world


def test_get_modifications_as_json_empty_world(rclpy_node):
    """
    Test that get_modifications_as_json returns an empty list for a world with no modifications.
    """
    world = World()
    fetcher = FetchWorldServer(node=rclpy_node, world=world)

    modifications_json = fetcher.get_modifications_as_json()
    modifications_list = json.loads(modifications_json)

    assert modifications_list == []
    fetcher.close()


def test_service_callback_success(rclpy_node):
    """
    Test that the service callback returns success with the modifications JSON.
    """
    world = create_dummy_world()
    fetcher = FetchWorldServer(node=rclpy_node, world=world)

    # Create a mock request and response
    request = Trigger.Request()
    response = Trigger.Response()

    # Call the service callback directly
    result = fetcher.service_callback(request, response)

    assert result.success is True

    # Verify the message is valid JSON
    modifications_list = [
        WorldModelModificationBlock.from_json(d) for d in json.loads(result.message)
    ]

    assert modifications_list == world.get_world_model_manager().model_modification_blocks

    fetcher.close()


def test_service_callback_with_multiple_modifications(rclpy_node):
    """
    Test that the service callback returns all modifications when multiple changes are made.
    """
    world = World()
    fetcher = FetchWorldServer(node=rclpy_node, world=world)

    # Make multiple modifications
    body_1 = Body(name=PrefixedName("body_1"))
    body_2 = Body(name=PrefixedName("body_2"))
    body_3 = Body(name=PrefixedName("body_3"))

    with world.modify_world():
        world.add_kinematic_structure_entity(body_1)

    with world.modify_world():
        world.add_kinematic_structure_entity(body_2)
        world.add_kinematic_structure_entity(body_3)
        world.add_connection(Connection6DoF(body_1, body_2, _world=world))
        world.add_connection(Connection6DoF(body_2, body_3, _world=world))

    request = Trigger.Request()
    response = Trigger.Response()

    result = fetcher.service_callback(request, response)

    assert result.success is True
    # Verify the message is valid JSON
    modifications_list = [
        WorldModelModificationBlock.from_json(d) for d in json.loads(result.message)
    ]
    assert modifications_list == world.get_world_model_manager().model_modification_blocks
    fetcher.close()


def test_world_fetching(rclpy_node):
    world = create_dummy_world()
    fetcher = FetchWorldServer(node=rclpy_node, world=world)

    world2 = fetch_world_from_service(
        rclpy_node,
    )
    assert world2.get_world_model_manager().model_modification_blocks == world.get_world_model_manager().model_modification_blocks


def test_semantic_annotation_modifications(rclpy_node):
    w1 = World(name="w1")
    b1 = Body(name=PrefixedName("b1"))
    v1 = Handle(body=b1)
    v2 = Door(body=b1, handle=v1)

    with w1.modify_world():
        w1.add_body(b1)
        w1.add_semantic_annotation(v1)
        w1.add_semantic_annotation(v2)

    fetcher = FetchWorldServer(node=rclpy_node, world=w1)

    w2 = fetch_world_from_service(
        rclpy_node,
    )

    assert [sa.name for sa in w1.semantic_annotations] == [
        sa.name for sa in w2.semantic_annotations
    ]
