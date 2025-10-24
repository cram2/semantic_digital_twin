import os
import unittest

import numpy as np
import sqlalchemy
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import RevoluteConnection
from semantic_digital_twin.world_description.geometry import Box, Scale, Color
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types.spatial_types import TransformationMatrix
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body
from semantic_digital_twin.orm.ormatic_interface import *
from ormatic.dao import to_dao


class ORMTest(unittest.TestCase):
    engine: sqlalchemy.engine
    session: Session

    urdf_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "resources", "urdf"
    )
    table = os.path.join(urdf_dir, "table.urdf")

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.engine = create_engine("sqlite:///:memory:")
        cls.table_world = URDFParser.from_file(file_path=cls.table).parse()

    def setUp(self):
        super().setUp()
        self.session = Session(self.engine)
        Base.metadata.create_all(bind=self.session.bind)

    def tearDown(self):
        super().tearDown()
        Base.metadata.drop_all(self.session.bind)
        self.session.close()

    def test_table_world(self):
        revolute_connection = self.table_world.get_connections_by_type(
            RevoluteConnection
        )[0]
        revolute_connection.position = 1
        revolute_connection.velocity = 23
        revolute_connection.acceleration = 42
        revolute_connection.jerk = 69
        fk = self.table_world.compute_forward_kinematics_np(
            root=revolute_connection.parent, tip=revolute_connection.child
        )
        world_dao: WorldMappingDAO = to_dao(self.table_world)

        self.session.add(world_dao)
        self.session.commit()

        bodies_from_db = self.session.scalars(select(BodyDAO)).all()
        self.assertEqual(
            len(bodies_from_db), len(self.table_world.kinematic_structure_entities)
        )

        connections_from_db = self.session.scalars(select(ConnectionDAO)).all()
        self.assertEqual(len(connections_from_db), len(self.table_world.connections))

        queried_world = self.session.scalar(select(WorldMappingDAO))
        reconstructed: World = queried_world.from_dao()

        fk2 = reconstructed.compute_forward_kinematics_np(
            root=revolute_connection.parent, tip=revolute_connection.child
        )
        assert np.allclose(fk, fk2)
        reconstructed_connection = reconstructed.get_connections_by_type(
            RevoluteConnection
        )[0]
        assert reconstructed_connection.position == revolute_connection.position
        assert reconstructed_connection.velocity == revolute_connection.velocity
        assert reconstructed_connection.acceleration == revolute_connection.acceleration
        assert reconstructed_connection.jerk == revolute_connection.jerk

    def test_insert(self):
        origin = TransformationMatrix.from_xyz_rpy(1, 2, 3, 1, 2, 3)
        scale = Scale(1.0, 1.0, 1.0)
        color = Color(0.0, 1.0, 1.0)
        shape1 = Box(origin=origin, scale=scale, color=color)
        b1 = Body(name=PrefixedName("b1"), collision=ShapeCollection([shape1]))

        dao: BodyDAO = to_dao(b1)

        self.session.add(dao)
        self.session.commit()
        queried_body = self.session.scalar(select(BodyDAO))
        reconstructed_body = queried_body.from_dao()
        self.assertIs(
            reconstructed_body, reconstructed_body.collision[0].origin.reference_frame
        )

        result = self.session.scalar(select(ShapeDAO))
        self.assertIsInstance(result, BoxDAO)
        box = result.from_dao()
