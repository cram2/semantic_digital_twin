import os
import time
import unittest

import numpy
from mujoco_connector import MultiverseMujocoConnector
from multiverse_simulator import MultiverseSimulatorState, MultiverseViewer

from semantic_world.adapters.multi_parser import MultiParser
from semantic_world.adapters.multi_sim import MultiSim
from semantic_world.datastructures.prefixed_name import PrefixedName
from semantic_world.spatial_types.spatial_types import TransformationMatrix
from semantic_world.world import World
from semantic_world.world_description.connections import Connection6DoF, FixedConnection
from semantic_world.world_description.geometry import Box, Scale, Color
from semantic_world.world_description.world_entity import Body, Region

urdf_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "resources", "urdf"
)
mjcf_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "resources", "mjcf"
)
headless = os.environ.get("CI", "false").lower() == "true"


class MultiverseMujocoConnectorTestCase(unittest.TestCase):
    file_path = os.path.normpath(os.path.join(mjcf_dir, "mjx_single_cube_no_mesh.xml"))
    Simulator = MultiverseMujocoConnector
    step_size = 5e-4

    def test_read_and_write_data_in_the_loop(self):
        viewer = MultiverseViewer()
        simulator = self.Simulator(
            viewer=viewer,
            file_path=self.file_path,
            headless=headless,
            step_size=self.step_size,
        )
        self.assertIs(simulator.state, MultiverseSimulatorState.STOPPED)
        self.assertIs(simulator.headless, headless)
        self.assertIsNone(simulator.stop_reason)
        self.assertIsNone(simulator.simulation_thread)
        simulator.start(simulate_in_thread=False)
        for step in range(1000):
            if step == 100:
                read_objects = {
                    "joint1": {
                        "joint_angular_position": [0.0],
                        "joint_angular_velocity": [0.0],
                    },
                    "joint2": {
                        "joint_angular_position": [0.0],
                        "joint_angular_velocity": [0.0],
                    },
                }
                viewer.read_objects = read_objects
            elif step == 101:
                read_objects = {
                    "joint1": {"joint_angular_velocity": [0.0]},
                    "joint2": {"joint_angular_position": [0.0], "joint_torque": [0.0]},
                }
                viewer.read_objects = read_objects
            elif step == 102:
                write_objects = {
                    "joint1": {"joint_angular_position": [1.0]},
                    "actuator2": {"cmd_joint_angular_position": [2.0]},
                    "box": {
                        "position": [1.1, 2.2, 3.3],
                        "quaternion": [0.707, 0.0, 0.707, 0.0],
                    },
                }
                read_objects = {
                    "joint1": {
                        "joint_angular_position": [0.0],
                        "joint_angular_velocity": [0.0],
                    },
                    "actuator2": {"cmd_joint_angular_position": [0.0]},
                    "box": {
                        "position": [0.0, 0.0, 0.0],
                        "quaternion": [0.0, 0.0, 0.0, 0.0],
                    },
                }
                viewer.write_objects = write_objects
                viewer.read_objects = read_objects
            else:
                viewer.read_objects = {}
            simulator.step()
            if step == 100:
                self.assertEqual(viewer.read_data.shape, (1, 4))
                self.assertEqual(
                    viewer.read_objects["joint1"][
                        "joint_angular_position"
                    ].values.shape,
                    (1, 1),
                )
                self.assertEqual(
                    viewer.read_objects["joint2"][
                        "joint_angular_position"
                    ].values.shape,
                    (1, 1),
                )
                self.assertEqual(
                    viewer.read_objects["joint1"][
                        "joint_angular_velocity"
                    ].values.shape,
                    (1, 1),
                )
                self.assertEqual(
                    viewer.read_objects["joint2"][
                        "joint_angular_velocity"
                    ].values.shape,
                    (1, 1),
                )
            elif step == 101:
                self.assertEqual(viewer.read_data.shape, (1, 3))
                self.assertEqual(
                    viewer.read_objects["joint1"][
                        "joint_angular_velocity"
                    ].values.shape,
                    (1, 1),
                )
                self.assertEqual(
                    viewer.read_objects["joint2"][
                        "joint_angular_position"
                    ].values.shape,
                    (1, 1),
                )
                self.assertEqual(
                    viewer.read_objects["joint2"]["joint_torque"].values.shape, (1, 1)
                )
            elif step == 102:
                self.assertEqual(viewer.write_data.shape, (1, 9))
                self.assertEqual(
                    viewer.write_objects["joint1"]["joint_angular_position"].values[0],
                    (1.0,),
                )
                self.assertEqual(
                    viewer.write_objects["actuator2"][
                        "cmd_joint_angular_position"
                    ].values[0],
                    (2.0,),
                )
                numpy.testing.assert_allclose(
                    viewer.write_objects["box"]["position"].values[0],
                    [1.1, 2.2, 3.3],
                    rtol=1e-5,
                    atol=1e-5,
                )
                numpy.testing.assert_allclose(
                    viewer.write_objects["box"]["quaternion"].values[0],
                    [0.707, 0.0, 0.707, 0.0],
                    rtol=1e-5,
                    atol=1e-5,
                )
                self.assertEqual(viewer.read_data.shape, (1, 10))
                self.assertAlmostEqual(
                    viewer.read_objects["joint1"]["joint_angular_position"].values[0][
                        0
                    ],
                    1.0,
                    places=3,
                )
                self.assertEqual(
                    viewer.read_objects["actuator2"][
                        "cmd_joint_angular_position"
                    ].values[0][0],
                    2.0,
                )
                numpy.testing.assert_allclose(
                    viewer.read_objects["box"]["position"].values[0],
                    [1.1, 2.2, 3.3],
                    rtol=1e-5,
                    atol=1e-5,
                )
                numpy.testing.assert_allclose(
                    viewer.read_objects["box"]["quaternion"].values[0],
                    [0.7071067811865475, 0.0, 0.7071067811865475, 0.0],
                    rtol=1e-5,
                    atol=1e-5,
                )
            else:
                self.assertEqual(viewer.read_data.shape, (1, 0))
        simulator.stop()
        self.assertIs(simulator.state, MultiverseSimulatorState.STOPPED)


class MultiSimTestCase(unittest.TestCase):
    test_urdf = os.path.normpath(os.path.join(urdf_dir, "simple_two_arm_robot.urdf"))
    test_mjcf = os.path.normpath(os.path.join(mjcf_dir, "mjx_single_cube_no_mesh.xml"))
    step_size = 1e-3

    def setUp(self):
        self.test_urdf_world = MultiParser(self.test_urdf).parse()
        self.test_mjcf_world = MultiParser(self.test_mjcf).parse()

    def test_empty_multi_sim_in_5s(self):
        world = World()
        viewer = MultiverseViewer()
        multi_sim = MultiSim(viewer=viewer, world=world, headless=headless)
        self.assertIsInstance(multi_sim.simulator, MultiverseMujocoConnector)
        self.assertEqual(multi_sim.simulator.file_path, "/tmp/scene.xml")
        self.assertIs(multi_sim.simulator.headless, headless)
        self.assertEqual(multi_sim.simulator.step_size, self.step_size)
        multi_sim.start_simulation()
        start_time = time.time()
        time.sleep(5.0)
        multi_sim.stop_simulation()
        self.assertAlmostEqual(time.time() - start_time, 5.0, delta=0.5)

    def test_world_multi_sim_in_5s(self):
        viewer = MultiverseViewer()
        multi_sim = MultiSim(
            viewer=viewer, world=self.test_urdf_world, headless=headless
        )
        self.assertIsInstance(multi_sim.simulator, MultiverseMujocoConnector)
        self.assertEqual(multi_sim.simulator.file_path, "/tmp/scene.xml")
        self.assertIs(multi_sim.simulator.headless, headless)
        self.assertEqual(multi_sim.simulator.step_size, self.step_size)
        multi_sim.start_simulation()
        start_time = time.time()
        time.sleep(5.0)
        multi_sim.stop_simulation()
        self.assertAlmostEqual(time.time() - start_time, 5.0, delta=0.5)

    def test_world_multi_sim_with_change(self):
        viewer = MultiverseViewer()
        multi_sim = MultiSim(
            viewer=viewer, world=self.test_urdf_world, headless=headless
        )
        self.assertIsInstance(multi_sim.simulator, MultiverseMujocoConnector)
        self.assertEqual(multi_sim.simulator.file_path, "/tmp/scene.xml")
        self.assertIs(multi_sim.simulator.headless, headless)
        self.assertEqual(multi_sim.simulator.step_size, self.step_size)
        multi_sim.start_simulation()
        start_time = time.time()
        time.sleep(1.0)

        current_time = time.time()
        new_body = Body(name=PrefixedName("test_body"))
        box_origin = TransformationMatrix.from_xyz_rpy(
            x=0.2, y=0.4, z=-0.3, roll=0, pitch=0.5, yaw=0, reference_frame=new_body
        )
        box = Box(
            origin=box_origin,
            scale=Scale(1.0, 1.5, 0.5),
            color=Color(
                1.0,
                0.0,
                0.0,
                1.0,
            ),
        )
        new_body.collision = [box]

        with self.test_urdf_world.modify_world():
            self.test_urdf_world.add_connection(
                Connection6DoF(
                    parent=self.test_urdf_world.root,
                    child=new_body,
                    _world=self.test_urdf_world,
                )
            )
        print(f"Time to add new body: {time.time() - current_time}s")
        self.assertIn(
            new_body.name.name, multi_sim.simulator.get_all_body_names().result
        )

        time.sleep(1)

        current_time = time.time()
        region = Region(name=PrefixedName("test_region"))
        region_box = Box(
            scale=Scale(0.1, 0.5, 0.2),
            origin=TransformationMatrix.from_xyz_rpy(reference_frame=region),
            color=Color(
                0.0,
                1.0,
                0.0,
                0.8,
            ),
        )
        region.area = [region_box]

        with self.test_urdf_world.modify_world():
            self.test_urdf_world.add_connection(
                FixedConnection(
                    parent=self.test_urdf_world.root,
                    child=region,
                    _world=self.test_urdf_world,
                    origin_expression=TransformationMatrix.from_xyz_rpy(z=0.5),
                )
            )
        print(f"Time to add new region: {time.time() - current_time}s")
        self.assertIn(region.name.name, multi_sim.simulator.get_all_body_names().result)

        time.sleep(4.0)

        multi_sim.stop_simulation()
        # self.assertAlmostEqual(time.time() - start_time, 1.0, delta=0.5)

    @unittest.skip("Dynamics not there yet")
    def test_multi_sim_in_5s(self):
        viewer = MultiverseViewer()
        multi_sim = MultiSim(
            world=self.test_mjcf_world,
            viewer=viewer,
            headless=headless,
            step_size=self.step_size,
        )
        self.assertIsInstance(multi_sim.simulator, MultiverseMujocoConnector)
        self.assertIs(multi_sim.simulator.headless, headless)
        self.assertEqual(multi_sim.simulator.step_size, self.step_size)
        multi_sim.start_simulation()
        start_time = time.time()
        time.sleep(5.0)
        multi_sim.stop_simulation()
        self.assertAlmostEqual(time.time() - start_time, 5.0, delta=0.1)

    @unittest.skip("Dynamics not there yet")
    def test_read_objects_from_multi_sim_in_5s(self):
        read_objects = {
            "joint1": {
                "joint_angular_position": [0.0],
                "joint_angular_velocity": [0.0],
            },
            "joint2": {
                "joint_angular_position": [0.0],
                "joint_angular_velocity": [0.0],
            },
            "actuator1": {"cmd_joint_angular_position": [0.0]},
            "actuator2": {"cmd_joint_angular_position": [0.0]},
            "world": {"energy": [0.0, 0.0]},
            "box": {"position": [0.0, 0.0, 0.0], "quaternion": [1.0, 0.0, 0.0, 0.0]},
        }
        viewer = MultiverseViewer(read_objects=read_objects)
        multi_sim = MultiSim(
            world=self.test_mjcf_world,
            viewer=viewer,
            headless=headless,
            step_size=self.step_size,
        )
        self.assertIsInstance(multi_sim.simulator, MultiverseMujocoConnector)
        self.assertIs(multi_sim.simulator.headless, headless)
        self.assertEqual(multi_sim.simulator.step_size, self.step_size)
        multi_sim.start_simulation()
        start_time = time.time()
        for _ in range(5):
            print(
                f"Time: {multi_sim.simulator.current_simulation_time} - Objects: {multi_sim.get_read_objects()}"
            )
            time.sleep(1)
        multi_sim.stop_simulation()
        self.assertAlmostEqual(time.time() - start_time, 5.0, delta=0.1)

    @unittest.skip("Dynamics not there yet")
    def test_write_objects_to_multi_sim_in_5s(self):
        write_objects = {"box": {"position": [0.0, 0.0, 0.0]}}
        viewer = MultiverseViewer(write_objects=write_objects)
        multi_sim = MultiSim(
            world=self.test_mjcf_world,
            viewer=viewer,
            headless=headless,
            step_size=self.step_size,
        )
        self.assertIsInstance(multi_sim.simulator, MultiverseMujocoConnector)
        self.assertIs(multi_sim.simulator.headless, headless)
        self.assertEqual(multi_sim.simulator.step_size, self.step_size)
        multi_sim.start_simulation()
        box_positions = [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0],
        ]
        multi_sim.pause_simulation()
        start_time = time.time()
        for box_position in box_positions:
            write_objects["box"]["position"] = box_position
            multi_sim.set_write_objects(write_objects=write_objects)
            time.sleep(1)
        multi_sim.stop_simulation()
        self.assertAlmostEqual(time.time() - start_time, 5.0, delta=0.1)

    @unittest.skip("Dynamics not there yet")
    def test_write_objects_to_multi_sim_in_10s_with_pause_and_unpause(self):
        write_objects = {
            "box": {"position": [0.0, 0.0, 0.0], "quaternion": [1.0, 0.0, 0.0, 0.0]}
        }
        viewer = MultiverseViewer()
        multi_sim = MultiSim(
            world=self.test_mjcf_world,
            viewer=viewer,
            headless=headless,
            step_size=self.step_size,
        )
        self.assertIsInstance(multi_sim.simulator, MultiverseMujocoConnector)
        self.assertIs(multi_sim.simulator.headless, headless)
        self.assertEqual(multi_sim.simulator.step_size, self.step_size)
        multi_sim.start_simulation()
        time.sleep(1)  # Ensure the simulation is running before setting objects
        box_positions = [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0],
        ]
        start_time = time.time()
        for box_position in box_positions:
            write_objects["box"]["position"] = box_position
            multi_sim.pause_simulation()
            multi_sim.set_write_objects(write_objects=write_objects)
            time.sleep(1)
            multi_sim.unpause_simulation()
            multi_sim.set_write_objects(write_objects={})
            time.sleep(1)
        multi_sim.stop_simulation()
        self.assertAlmostEqual(time.time() - start_time, 10.0, delta=0.1)

    @unittest.skip("Dynamics not there yet")
    def test_stable(self):
        write_objects = {
            "box": {"position": [0.0, 0.0, 0.0], "quaternion": [1.0, 0.0, 0.0, 0.0]}
        }
        viewer = MultiverseViewer()
        multi_sim = MultiSim(
            world=self.test_mjcf_world,
            viewer=viewer,
            headless=headless,
            step_size=self.step_size,
            real_time_factor=-1.0,
        )
        multi_sim.start_simulation()
        time.sleep(1)  # Ensure the simulation is running before setting objects
        stable_box_poses = [
            [[0.0, 0.0, 0.03], [1.0, 0.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.02], [0.707, 0.0, 0.707, 0.0]],
            [[1.0, 1.0, 0.02], [0.5, 0.5, 0.5, 0.5]],
            [[0.0, 1.0, 0.03], [0.0, 0.707, 0.707, 0.0]],
        ]
        for _ in range(100):
            for stable_box_pose in stable_box_poses:
                write_objects["box"]["position"] = stable_box_pose[0]
                write_objects["box"]["quaternion"] = stable_box_pose[1]
                multi_sim.pause_simulation()
                multi_sim.set_write_objects(write_objects=write_objects)
                multi_sim.set_write_objects(write_objects={})
                multi_sim.unpause_simulation()
                self.assertTrue(
                    multi_sim.is_stable(
                        body_names=["box"], max_simulation_steps=1000, atol=1e-3
                    )
                )

        unstable_box_poses = [
            [[0.0, 0.0, 1.03], [1.0, 0.0, 0.0, 0.0]],
            [[1.0, 0.0, 1.03], [0.707, 0.0, 0.707, 0.0]],
            [[1.0, 1.0, 1.03], [0.5, 0.5, 0.5, 0.5]],
            [[0.0, 1.0, 1.03], [0.0, 0.707, 0.707, 0.0]],
            [[0.0, 0.0, 1.03], [1.0, 0.0, 0.0, 0.0]],
        ]
        for _ in range(100):
            for unstable_box_pose in unstable_box_poses:
                write_objects["box"]["position"] = unstable_box_pose[0]
                write_objects["box"]["quaternion"] = unstable_box_pose[1]
                multi_sim.pause_simulation()
                multi_sim.set_write_objects(write_objects=write_objects)
                multi_sim.set_write_objects(write_objects={})
                multi_sim.unpause_simulation()
                self.assertFalse(
                    multi_sim.is_stable(
                        body_names=["box"], max_simulation_steps=1000, atol=1e-3
                    )
                )
        multi_sim.stop_simulation()


if __name__ == "__main__":
    unittest.main()
