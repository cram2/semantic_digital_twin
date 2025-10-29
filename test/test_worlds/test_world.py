from copy import deepcopy

import numpy as np
import pytest
from semantic_digital_twin.semantic_annotations.semantic_annotations import Handle

from semantic_digital_twin.spatial_types import Vector3
from semantic_digital_twin.world_description.connections import (
    PrismaticConnection,
    RevoluteConnection,
    Connection6DoF,
    FixedConnection,
)
from semantic_digital_twin.exceptions import (
    AddingAnExistingSemanticAnnotationError,
    DuplicateSemanticAnnotationError,
    SemanticAnnotationNotFoundError,
    DuplicateKinematicStructureEntityError,
    UsageError,
    MissingWorldModificationContextError,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types.derivatives import Derivatives, DerivativeMap

# from semantic_digital_twin.spatial_types.math import rotation_matrix_from_rpy
from semantic_digital_twin.spatial_types.spatial_types import (
    TransformationMatrix,
    RotationMatrix,
)
from semantic_digital_twin.spatial_types.symbol_manager import symbol_manager
from semantic_digital_twin.testing import world_setup, pr2_world
from semantic_digital_twin.world_description.world_entity import (
    SemanticAnnotation,
    Body,
)


def test_set_state(world_setup):
    world, l1, l2, bf, r1, r2 = world_setup
    c1: PrismaticConnection = world.get_connection(l1, l2)
    c1.position = 1.0
    assert c1.position == 1.0
    c2: RevoluteConnection = world.get_connection(r1, r2)
    c2.position = 1337
    assert c2.position == 1337
    c3: Connection6DoF = world.get_connection(world.root, bf)
    transform = RotationMatrix.from_rpy(1, 0, 0).to_np()
    transform[0, 3] = 69
    c3.origin = transform
    assert np.allclose(world.compute_forward_kinematics_np(world.root, bf), transform)

    world.set_positions_1DOF_connection({c1: 2})
    assert c1.position == 2.0

    transform[0, 3] += c1.position
    assert np.allclose(l2.global_pose.to_np(), transform)


def test_construction(world_setup):
    world, l1, l2, bf, r1, r2 = world_setup
    world.validate()
    assert len(world.connections) == 5
    assert len(world.kinematic_structure_entities) == 6
    assert world.state.positions[0] == 0
    assert (
        world.get_connection(l1, l2).dof.name == world.get_connection(r1, r2).dof.name
    )


def test_chain_of_bodies(world_setup):
    world, _, l2, _, _, _ = world_setup
    result = world.compute_chain_of_kinematic_structure_entities(
        root=world.root, tip=l2
    )
    result = [x.name for x in result]
    assert result == [
        PrefixedName(name="root", prefix="world"),
        PrefixedName(name="bf", prefix=None),
        PrefixedName(name="l1", prefix=None),
        PrefixedName(name="l2", prefix=None),
    ]


def test_chain_of_connections(world_setup):
    world, _, l2, _, _, _ = world_setup
    result = world.compute_chain_of_connections(root=world.root, tip=l2)
    result = [x.name for x in result]
    assert result == [
        PrefixedName(name="root_T_bf", prefix=None),
        PrefixedName(name="bf_T_l1", prefix=None),
        PrefixedName(name="l1_T_l2", prefix=None),
    ]


def test_split_chain_of_bodies(world_setup):
    world, _, l2, _, _, r2 = world_setup
    result = world.compute_split_chain_of_kinematic_structure_entities(root=r2, tip=l2)
    result = tuple([x.name for x in y] for y in result)
    assert result == (
        [PrefixedName(name="r2", prefix=None), PrefixedName(name="r1", prefix=None)],
        [PrefixedName(name="bf", prefix=None)],
        [PrefixedName(name="l1", prefix=None), PrefixedName(name="l2", prefix=None)],
    )


def test_split_chain_of_bodies_adjacent1(world_setup):
    world, _, _, _, r1, r2 = world_setup
    result = world.compute_split_chain_of_kinematic_structure_entities(root=r2, tip=r1)
    result = tuple([x.name for x in y] for y in result)
    assert result == (
        [PrefixedName(name="r2", prefix=None)],
        [PrefixedName(name="r1", prefix=None)],
        [],
    )


def test_split_chain_of_bodies_adjacent2(world_setup):
    world, _, _, _, r1, r2 = world_setup
    result = world.compute_split_chain_of_kinematic_structure_entities(root=r1, tip=r2)
    result = tuple([x.name for x in y] for y in result)
    assert result == (
        [],
        [PrefixedName(name="r1", prefix=None)],
        [PrefixedName(name="r2", prefix=None)],
    )


def test_split_chain_of_bodies_identical(world_setup):
    world, _, _, _, r1, _ = world_setup
    result = world.compute_split_chain_of_kinematic_structure_entities(root=r1, tip=r1)
    result = tuple([x.name for x in y] for y in result)
    assert result == ([], [PrefixedName(name="r1", prefix=None)], [])


def test_split_chain_of_connections(world_setup):
    world, _, l2, _, _, r2 = world_setup
    result = world.compute_split_chain_of_connections(root=r2, tip=l2)
    result = tuple([x.name for x in y] for y in result)
    assert result == (
        [
            PrefixedName(name="r1_T_r2", prefix=None),
            PrefixedName(name="bf_T_r1", prefix=None),
        ],
        [
            PrefixedName(name="bf_T_l1", prefix=None),
            PrefixedName(name="l1_T_l2", prefix=None),
        ],
    )


def test_split_chain_of_connections_adjacent1(world_setup):
    world, _, _, _, r1, r2 = world_setup
    result = world.compute_split_chain_of_connections(root=r2, tip=r1)
    result = tuple([x.name for x in y] for y in result)
    assert result == ([PrefixedName(name="r1_T_r2", prefix=None)], [])


def test_split_chain_of_connections_adjacent2(world_setup):
    world, _, _, _, r1, r2 = world_setup
    result = world.compute_split_chain_of_connections(root=r1, tip=r2)
    result = tuple([x.name for x in y] for y in result)
    assert result == ([], [PrefixedName(name="r1_T_r2", prefix=None)])


def test_split_chain_of_connections_identical(world_setup):
    world, _, _, _, r1, _ = world_setup
    result = world.compute_split_chain_of_connections(root=r1, tip=r1)
    result = tuple([x.name for x in y] for y in result)
    assert result == ([], [])


@pytest.mark.skip(
    reason="readding of 1dof connection broken because reference to dof is lost"
)
def test_nested_with_blocks_illegal_state(world_setup):
    world, l1, l2, bf, r1, r2 = world_setup

    with world.modify_world():
        connection1 = world.get_connection(l1, l2)
        world.remove_connection(connection1)
        with world.modify_world():
            connection2 = world.get_connection(r1, r2)
            world.remove_connection(connection2)
        world.add_connection(connection1)
        world.add_connection(connection2)


def test_compute_fk_connection6dof(world_setup):
    world, _, _, bf, _, _ = world_setup
    fk = world.compute_forward_kinematics_np(world.root, bf)
    np.testing.assert_array_equal(fk, np.eye(4))

    connection: Connection6DoF = world.get_connection(world.root, bf)

    world.state[connection.x.name].position = 1.0
    world.state[connection.qw.name].position = 0
    world.state[connection.qz.name].position = 1
    world.notify_state_change()
    fk = world.compute_forward_kinematics_np(world.root, bf)
    np.testing.assert_array_equal(
        fk,
        [
            [-1.0, 0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    )


def test_compute_fk(world_setup):
    world, l1, l2, bf, r1, r2 = world_setup
    fk = world.compute_forward_kinematics_np(l2, r2)
    np.testing.assert_array_equal(fk, np.eye(4))

    connection: PrismaticConnection = world.get_connection(r1, r2)

    world.state[connection.dof.name].position = 1.0
    world.notify_state_change()
    fk = world.compute_forward_kinematics_np(l2, r2)
    assert np.allclose(
        fk,
        np.array(
            [
                [0.540302, -0.841471, 0.0, -1.0],
                [0.841471, 0.540302, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
    )


def test_compute_ik(world_setup):
    world, l1, l2, bf, r1, r2 = world_setup
    target = np.array(
        [
            [0.540302, -0.841471, 0.0, -1.0],
            [0.841471, 0.540302, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    joint_state = world.compute_inverse_kinematics(
        l2, r2, TransformationMatrix(target, reference_frame=l2)
    )
    for joint, state in joint_state.items():
        world.state[joint.name].position = state
    world.notify_state_change()
    assert np.allclose(world.compute_forward_kinematics_np(l2, r2), target, atol=1e-3)


def test_compute_fk_expression(world_setup):
    world, l1, l2, bf, r1, r2 = world_setup
    connection: PrismaticConnection = world.get_connection(r1, r2)
    world.state[connection.dof.name].position = 1.0
    world.notify_state_change()
    fk = world.compute_forward_kinematics_np(r2, l2)
    fk_expr = world.compose_forward_kinematics_expression(r2, l2)
    fk_expr_compiled = fk_expr.compile()
    fk2 = fk_expr_compiled(
        *symbol_manager.resolve_symbols(fk_expr_compiled.symbol_parameters)
    )
    np.testing.assert_array_almost_equal(fk, fk2)


def test_apply_control_commands(world_setup):
    world, l1, l2, bf, r1, r2 = world_setup
    connection: PrismaticConnection = world.get_connection(r1, r2)
    cmd = np.array([100.0, 0, 0, 0, 0, 0, 0, 0])
    dt = 0.1
    world.apply_control_commands(cmd, dt, Derivatives.jerk)
    assert world.state[connection.dof.name].jerk == 100.0
    assert world.state[connection.dof.name].acceleration == 100.0 * dt
    assert world.state[connection.dof.name].velocity == 100.0 * dt * dt
    assert world.state[connection.dof.name].position == 100.0 * dt * dt * dt


def test_compute_relative_pose(world_setup):
    world, l1, l2, bf, r1, r2 = world_setup
    connection: PrismaticConnection = world.get_connection(l1, l2)
    world.state[connection.dof.name].position = 1.0
    world.notify_state_change()

    pose = TransformationMatrix(reference_frame=l2)
    relative_pose = world.transform(pose, l1)
    expected_pose = TransformationMatrix(
        [
            [1.0, 0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    np.testing.assert_array_almost_equal(relative_pose.to_np(), expected_pose.to_np())


def test_compute_relative_pose_both(world_setup):
    world, l1, l2, bf, r1, r2 = world_setup
    world.get_connection(world.root, bf).origin = np.array(
        [
            [0.0, -1.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    world.notify_state_change()

    pose = TransformationMatrix.from_xyz_rpy(x=1.0, reference_frame=bf)
    relative_pose = world.transform(pose, world.root)
    # Rotation is 90 degrees around z-axis, translation is 1 along x-axis
    expected_pose = np.array(
        [
            [0.0, -1.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    np.testing.assert_array_almost_equal(relative_pose.to_np(), expected_pose)


def test_compute_relative_pose_only_translation(world_setup):
    world, l1, l2, bf, r1, r2 = world_setup
    connection: PrismaticConnection = world.get_connection(l1, l2)
    world.state[connection.dof.name].position = 1.0
    world.notify_state_change()

    pose = TransformationMatrix.from_xyz_rpy(x=2.0, reference_frame=l2)
    relative_pose = world.transform(pose, l1)
    expected_pose = np.array(
        [
            [1.0, 0, 0.0, 3.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    np.testing.assert_array_almost_equal(relative_pose.to_np(), expected_pose)


def test_compute_relative_pose_only_rotation(world_setup):
    world, l1, l2, bf, r1, r2 = world_setup
    connection: RevoluteConnection = world.get_connection(r1, r2)
    world.state[connection.dof.name].position = np.pi / 2  # 90 degrees
    world.notify_state_change()

    pose = TransformationMatrix(reference_frame=r2)
    relative_pose = world.transform(pose, r1)
    expected_pose = np.array(
        [
            [0.0, -1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    np.testing.assert_array_almost_equal(relative_pose.to_np(), expected_pose)


def test_add_semantic_annotation(world_setup):
    world, l1, l2, bf, r1, r2 = world_setup
    v = SemanticAnnotation(name=PrefixedName("muh"))
    with world.modify_world():
        world.add_semantic_annotation(v)
    with pytest.raises(AddingAnExistingSemanticAnnotationError):
        world.add_semantic_annotation(v, exists_ok=False)
    assert world.get_semantic_annotation_by_name(v.name) == v


def test_duplicate_semantic_annotation(world_setup):
    world, l1, l2, bf, r1, r2 = world_setup
    v = SemanticAnnotation(name=PrefixedName("muh"))
    with world.modify_world():
        world.add_semantic_annotation(v)
        world.semantic_annotations.append(v)
    with pytest.raises(DuplicateSemanticAnnotationError):
        world.get_semantic_annotation_by_name(v.name)


def test_merge_world(world_setup, pr2_world):
    world, l1, l2, bf, r1, r2 = world_setup

    base_link = pr2_world.get_kinematic_structure_entity_by_name(
        PrefixedName("base_link")
    )
    r_gripper_tool_frame = pr2_world.get_kinematic_structure_entity_by_name(
        PrefixedName("r_gripper_tool_frame")
    )
    torso_lift_link = pr2_world.get_kinematic_structure_entity_by_name(
        PrefixedName("torso_lift_link")
    )
    r_shoulder_pan_joint = pr2_world.get_connection(
        torso_lift_link,
        pr2_world.get_kinematic_structure_entity_by_name(
            PrefixedName("r_shoulder_pan_link")
        ),
    )

    l_shoulder_pan_joint = pr2_world.get_connection(
        torso_lift_link,
        pr2_world.get_kinematic_structure_entity_by_name(
            PrefixedName("l_shoulder_pan_link")
        ),
    )

    world.merge_world(pr2_world)

    assert base_link in world.kinematic_structure_entities
    assert r_gripper_tool_frame in world.kinematic_structure_entities
    assert l_shoulder_pan_joint in world.connections
    assert torso_lift_link._world == world
    assert r_shoulder_pan_joint._world == world


def test_merge_with_connection(world_setup, pr2_world):
    world, l1, l2, bf, r1, r2 = world_setup

    base_link = pr2_world.get_kinematic_structure_entity_by_name(
        PrefixedName("base_link")
    )
    r_gripper_tool_frame = pr2_world.get_kinematic_structure_entity_by_name(
        PrefixedName("r_gripper_tool_frame")
    )
    torso_lift_link = pr2_world.get_kinematic_structure_entity_by_name(
        PrefixedName("torso_lift_link")
    )
    r_shoulder_pan_joint = pr2_world.get_connection(
        torso_lift_link,
        pr2_world.get_kinematic_structure_entity_by_name(
            PrefixedName("r_shoulder_pan_link")
        ),
    )

    pose = np.eye(4)
    pose[0, 3] = 1.0

    origin = TransformationMatrix(pose)

    connection = pr2_world.get_connection_by_name("l_gripper_l_finger_joint")
    pr2_world.state[connection.dof.name].position = 0.55
    pr2_world.notify_state_change()
    expected_fk = pr2_world.compute_forward_kinematics(
        connection.parent, connection.child
    ).to_np()

    new_connection = FixedConnection(
        parent=world.root,
        child=pr2_world.root,
        parent_T_connection_expression=origin,
    )

    world.merge_world(pr2_world, new_connection)
    assert base_link in world.kinematic_structure_entities
    assert r_gripper_tool_frame in world.kinematic_structure_entities
    assert new_connection in world.connections
    assert torso_lift_link._world == world
    assert r_shoulder_pan_joint._world == world
    assert world.state[connection.dof.name].position == 0.55
    assert world.compute_forward_kinematics_np(world.root, base_link)[
        0, 3
    ] == pytest.approx(1.0, abs=1e-6)
    actual_fk = world.compute_forward_kinematics(
        connection.parent, connection.child
    ).to_np()
    assert np.allclose(actual_fk, expected_fk)


def test_merge_with_pose(world_setup, pr2_world):
    world, l1, l2, bf, r1, r2 = world_setup

    base_link = pr2_world.get_kinematic_structure_entity_by_name(
        PrefixedName("base_link")
    )
    r_gripper_tool_frame = pr2_world.get_kinematic_structure_entity_by_name(
        PrefixedName("r_gripper_tool_frame")
    )
    torso_lift_link = pr2_world.get_kinematic_structure_entity_by_name(
        PrefixedName("torso_lift_link")
    )
    r_shoulder_pan_joint = pr2_world.get_connection(
        torso_lift_link,
        pr2_world.get_kinematic_structure_entity_by_name(
            PrefixedName("r_shoulder_pan_link")
        ),
    )

    pose = np.eye(4)
    pose[0, 3] = 1.0  # Translate along x-axis

    world.merge_world_at_pose(pr2_world, TransformationMatrix(pose))

    assert base_link in world.kinematic_structure_entities
    assert r_gripper_tool_frame in world.kinematic_structure_entities
    assert torso_lift_link._world == world
    assert r_shoulder_pan_joint._world == world
    assert world.compute_forward_kinematics_np(world.root, base_link)[
        0, 3
    ] == pytest.approx(1.0, abs=1e-6)


def test_merge_with_pose_rotation(world_setup, pr2_world):
    world, l1, l2, bf, r1, r2 = world_setup

    base_link = pr2_world.get_kinematic_structure_entity_by_name(
        PrefixedName("base_link")
    )
    r_gripper_tool_frame = pr2_world.get_kinematic_structure_entity_by_name(
        PrefixedName("r_gripper_tool_frame")
    )
    torso_lift_link = pr2_world.get_kinematic_structure_entity_by_name(
        PrefixedName("torso_lift_link")
    )
    r_shoulder_pan_joint = pr2_world.get_connection(
        torso_lift_link,
        pr2_world.get_kinematic_structure_entity_by_name(
            PrefixedName("r_shoulder_pan_link")
        ),
    )
    base_footprint = pr2_world.get_kinematic_structure_entity_by_name(
        PrefixedName("base_footprint")
    )

    # Rotation is 90 degrees around z-axis, translation is 1 along x-axis
    pose = np.array(
        [
            [0.0, -1.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    world.merge_world_at_pose(pr2_world, TransformationMatrix(pose))

    assert base_link in world.kinematic_structure_entities
    assert r_gripper_tool_frame in world.kinematic_structure_entities
    assert torso_lift_link._world == world
    assert r_shoulder_pan_joint._world == world
    fk_base = world.compute_forward_kinematics_np(world.root, base_footprint)
    assert fk_base[0, 3] == pytest.approx(1.0, abs=1e-6)
    assert fk_base[1, 3] == pytest.approx(1.0, abs=1e-6)
    assert fk_base[2, 3] == pytest.approx(0.0, abs=1e-6)
    np.testing.assert_array_almost_equal(
        RotationMatrix.from_rpy(0, 0, np.pi / 2).to_np()[:3, :3],
        fk_base[:3, :3],
        decimal=6,
    )


def test_remove_connection(world_setup):
    world, l1, l2, bf, r1, r2 = world_setup
    connection = world.get_connection(l1, l2)
    num_dofs = len(world.degrees_of_freedom)
    with world.modify_world():
        world.remove_connection(connection)
        world.remove_kinematic_structure_entity(l2)
    assert connection not in world.connections
    # dof should still exist because it was a mimic connection, so the number didn't change.
    assert num_dofs == len(world.degrees_of_freedom)

    with world.modify_world():
        world.remove_connection(world.get_connection(r1, r2))
        new_connection = FixedConnection(r1, r2)
        world.add_connection(new_connection)

    with pytest.raises(ValueError):
        # if you remove a connection, the child must be connected some other way or deleted
        world.remove_connection(world.get_connection(r1, r2))


def test_copy_world(world_setup):
    world, l1, l2, bf, r1, r2 = world_setup
    world_copy = deepcopy(world)
    assert l2 not in world_copy.bodies
    assert bf.parent_connection not in world_copy.connections
    bf.parent_connection.origin = np.array(
        [[1, 0, 0, 1.5], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    )
    assert (
        float(
            world_copy.get_kinematic_structure_entity_by_name("bf").global_pose.to_np()[
                0, 3
            ]
        )
        == 0.0
    )
    assert float(bf.global_pose.to_np()[0, 3]) == 1.5


def test_copy_world_state(world_setup):
    world, l1, l2, bf, r1, r2 = world_setup
    connection: PrismaticConnection = world.get_connection(r1, r2)
    world.state[connection.dof.name].position = 1.0
    world.notify_state_change()
    world_copy = deepcopy(world)

    assert world.get_connection(r1, r2).position == 1.0
    assert world_copy.get_connection(r1, r2).position == 1.0


def test_match_index(world_setup):
    world, l1, l2, bf, r1, r2 = world_setup
    world_copy = deepcopy(world)
    for body in world.bodies:
        new_body = world_copy.get_kinematic_structure_entity_by_name(body.name)
        assert body.index == new_body.index


def test_copy_dof(world_setup):
    world, l1, l2, bf, r1, r2 = world_setup
    world_copy = deepcopy(world)
    for dof in world.degrees_of_freedom:
        new_dof = world_copy.get_degree_of_freedom_by_name(dof.name)
        assert dof.name == new_dof.name
        assert dof.lower_limits == new_dof.lower_limits
        assert dof.upper_limits == new_dof.upper_limits


def test_copy_pr2_world(pr2_world):
    pr2_world.state[
        pr2_world.get_degree_of_freedom_by_name("torso_lift_joint").name
    ].position = 0.3
    pr2_world.notify_state_change()
    pr2_copy = deepcopy(pr2_world)


def test_copy_pr2_world_connection_origin(pr2_world):
    pr2_world.notify_state_change()
    pr2_copy = deepcopy(pr2_world)

    for body in pr2_world.bodies:
        pr2_body = pr2_world.get_kinematic_structure_entity_by_name(body.name)
        pr2_copy_body = pr2_copy.get_kinematic_structure_entity_by_name(body.name)
        np.testing.assert_array_almost_equal(
            pr2_body.global_pose.to_np(), pr2_copy_body.global_pose.to_np()
        )


def test_world_different_entities(world_setup):
    world, l1, l2, bf, r1, r2 = world_setup
    world_copy = deepcopy(world)
    for body in world_copy.bodies:
        assert body not in world.bodies
    for connection in world_copy.connections:
        assert connection not in world.connections
    for dof in world_copy.state:
        assert dof not in world.degrees_of_freedom


def test_copy_pr2(pr2_world):
    pr2_world.state[
        pr2_world.get_degree_of_freedom_by_name("torso_lift_joint").name
    ].position = 0.3
    pr2_world.notify_state_change()
    pr2_copy = deepcopy(pr2_world)
    assert pr2_world.get_kinematic_structure_entity_by_name(
        "head_tilt_link"
    ).global_pose.to_np()[2, 3] == pytest.approx(1.472, abs=1e-3)
    assert pr2_copy.get_kinematic_structure_entity_by_name(
        "head_tilt_link"
    ).global_pose.to_np()[2, 3] == pytest.approx(1.472, abs=1e-3)


def test_add_entity_with_duplicate_name(world_setup):
    world, l1, l2, bf, r1, r2 = world_setup
    body_duplicate = Body(name=PrefixedName("l1"))
    with pytest.raises(DuplicateKinematicStructureEntityError):
        with world.modify_world():
            world.add_kinematic_structure_entity(body_duplicate)


def test_overwrite_dof_limits(world_setup):
    world, l1, l2, bf, r1, r2 = world_setup
    connection: PrismaticConnection = world.get_connections_by_type(
        PrismaticConnection
    )[0]
    assert connection.dof.lower_limits.velocity == -1
    assert connection.dof.upper_limits.velocity == 1

    new_limits = DerivativeMap([0.69, 0.42, 1337, 23])

    connection.raw_dof._overwrite_dof_limits(
        new_lower_limits=new_limits * -1, new_upper_limits=new_limits
    )
    assert connection.dof.lower_limits.position == -new_limits.position
    assert connection.dof.upper_limits.position == new_limits.position
    assert connection.dof.lower_limits.velocity == -new_limits.velocity
    assert connection.dof.upper_limits.velocity == new_limits.velocity
    assert connection.dof.lower_limits.acceleration == -new_limits.acceleration
    assert connection.dof.upper_limits.acceleration == new_limits.acceleration
    assert connection.dof.lower_limits.jerk == -new_limits.jerk
    assert connection.dof.upper_limits.jerk == new_limits.jerk

    new_limits2 = DerivativeMap([3333, 3333, 3333, 3333])

    connection.raw_dof._overwrite_dof_limits(
        new_lower_limits=new_limits2 * -1, new_upper_limits=new_limits2
    )
    assert connection.dof.lower_limits.position == -new_limits.position
    assert connection.dof.upper_limits.position == new_limits.position
    assert connection.dof.lower_limits.velocity == -new_limits.velocity
    assert connection.dof.upper_limits.velocity == new_limits.velocity
    assert connection.dof.lower_limits.acceleration == -new_limits.acceleration
    assert connection.dof.upper_limits.acceleration == new_limits.acceleration
    assert connection.dof.lower_limits.jerk == -new_limits.jerk
    assert connection.dof.upper_limits.jerk == new_limits.jerk


def test_overwrite_dof_limits_mimic(world_setup):
    world, l1, l2, bf, r1, r2 = world_setup
    connection: PrismaticConnection = world.get_connections_by_type(
        PrismaticConnection
    )[0]
    with world.modify_world():
        body = Body(name=PrefixedName("muh"))
        mimic_connection = PrismaticConnection(
            parent=bf,
            child=body,
            offset=23,
            multiplier=-2,
            axis=Vector3(0, 0, 1),
            dof_name=connection.dof_name,
        )
        world.add_body(body)
        world.add_connection(mimic_connection)

    # when the multiplier is negative, the vel limits shouldn't be swapped
    assert np.isclose(
        mimic_connection.dof.lower_limits.velocity,
        connection.dof.lower_limits.velocity * 2,
    )
    assert np.isclose(
        mimic_connection.dof.upper_limits.velocity,
        connection.dof.upper_limits.velocity * 2,
    )

    new_limits = DerivativeMap([0.69, 0.42, 1337, 23])

    with pytest.raises(UsageError):
        mimic_connection.dof._overwrite_dof_limits(
            new_lower_limits=new_limits * -1, new_upper_limits=new_limits
        )

    mimic_connection.raw_dof._overwrite_dof_limits(
        new_lower_limits=new_limits * -1, new_upper_limits=new_limits
    )

    # Check that limits are correctly applied with negative multiplier and offset
    # Position limits: swapped due to negative multiplier, then scaled and offset applied
    # Lower becomes: new_limits.position * (-2) + 23 = 0.69 * (-2) + 23 = -1.38 + 23 = 21.62
    # Upper becomes: (new_limits * -1).position * (-2) + 23 = -0.69 * (-2) + 23 = 1.38 + 23 = 24.38
    assert np.isclose(mimic_connection.dof.lower_limits.position, 21.62)
    assert np.isclose(mimic_connection.dof.upper_limits.position, 24.38)

    # Velocity limits: only multiplier applied (no offset), but absolute value for limits
    # Since we're dealing with limits, velocity should use abs(multiplier) = 2
    assert np.isclose(
        mimic_connection.dof.lower_limits.velocity, (new_limits * -1).velocity * 2
    )
    assert np.isclose(
        mimic_connection.dof.upper_limits.velocity, new_limits.velocity * 2
    )

    # Acceleration limits: only multiplier applied (no offset), absolute value for limits
    assert np.isclose(
        mimic_connection.dof.lower_limits.acceleration,
        (new_limits * -1).acceleration * 2,
    )
    assert np.isclose(
        mimic_connection.dof.upper_limits.acceleration, new_limits.acceleration * 2
    )

    # Jerk limits: only multiplier applied (no offset), absolute value for limits
    assert np.isclose(
        mimic_connection.dof.lower_limits.jerk, (new_limits * -1).jerk * 2
    )
    assert np.isclose(mimic_connection.dof.upper_limits.jerk, new_limits.jerk * 2)

    # limits are only applied if the new ones are lower
    new_limits2 = DerivativeMap([3333, 3333, 3333, 3333])

    mimic_connection.raw_dof._overwrite_dof_limits(
        new_lower_limits=new_limits2 * -1, new_upper_limits=new_limits2
    )

    assert np.isclose(mimic_connection.dof.lower_limits.position, 21.62)
    assert np.isclose(mimic_connection.dof.upper_limits.position, 24.38)

    assert np.isclose(
        mimic_connection.dof.lower_limits.velocity, (new_limits * -1).velocity * 2
    )
    assert np.isclose(
        mimic_connection.dof.upper_limits.velocity, new_limits.velocity * 2
    )

    assert np.isclose(
        mimic_connection.dof.lower_limits.acceleration,
        (new_limits * -1).acceleration * 2,
    )
    assert np.isclose(
        mimic_connection.dof.upper_limits.acceleration, new_limits.acceleration * 2
    )

    assert np.isclose(
        mimic_connection.dof.lower_limits.jerk, (new_limits * -1).jerk * 2
    )
    assert np.isclose(mimic_connection.dof.upper_limits.jerk, new_limits.jerk * 2)


def test_missing_world_modification_context(world_setup):
    world, l1, l2, bf, r1, r2 = world_setup
    with pytest.raises(MissingWorldModificationContextError):
        world.add_semantic_annotation(Handle(l1))
