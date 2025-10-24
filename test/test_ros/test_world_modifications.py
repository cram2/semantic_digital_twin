import unittest

from semantic_digital_twin.world_description.degree_of_freedom import DegreeOfFreedom

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types.spatial_types import Vector3
from semantic_digital_twin.semantic_annotations.semantic_annotations import Handle, Door
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    FixedConnection,
    Connection6DoF,
    PrismaticConnection,
    RevoluteConnection,
)
from semantic_digital_twin.world_description.world_entity import Body
from semantic_digital_twin.world_description.world_modification import (
    WorldModelModificationBlock,
    AddKinematicStructureEntityModification,
    AddConnectionModification,
    AddDegreeOfFreedomModification,
    AddSemanticAnnotationModification,
    RemoveSemanticAnnotationModification,
)


class ConnectionModificationTestCase(unittest.TestCase):

    def test_single_modification(self):
        w = World()

        with w.modify_world():
            b1 = Body(name=PrefixedName("b1"))
            b2 = Body(name=PrefixedName("b2"))
            w.add_kinematic_structure_entity(b1)
            w.add_kinematic_structure_entity(b2)

            connection = FixedConnection(b1, b2)
            w.add_connection(connection)

    def test_ChangeDofHasHardwareInterface(self):
        w = World()

        with w.modify_world():
            b1 = Body(name=PrefixedName("b1"))
            b2 = Body(name=PrefixedName("b2"))
            w.add_kinematic_structure_entity(b1)
            w.add_kinematic_structure_entity(b2)

            dof = DegreeOfFreedom(name=PrefixedName("dofyboi"))
            w.add_degree_of_freedom(dof)
            connection = RevoluteConnection(
                b1, b2, axis=Vector3.from_iterable([0, 0, 1]), dof_name=dof.name
            )
            w.add_connection(connection)
        assert connection.dof.has_hardware_interface is False

        with w.modify_world():
            w.set_dofs_has_hardware_interface(connection.dofs, True)
        assert connection.dof.has_hardware_interface is True

    def test_many_modifications(self):
        w = World()

        with w.modify_world():
            b1 = Body(name=PrefixedName("b1"))
            b2 = Body(name=PrefixedName("b2"))
            b3 = Body(name=PrefixedName("b3"))
            w.add_kinematic_structure_entity(b1)
            w.add_kinematic_structure_entity(b2)
            w.add_kinematic_structure_entity(b3)
            w.add_connection(
                Connection6DoF.with_auto_generated_dofs(parent=b1, child=b2, world=w)
            )
            dof = DegreeOfFreedom(name=PrefixedName("dofyboi"))
            w.add_degree_of_freedom(dof)
            w.add_connection(
                PrismaticConnection(
                    parent=b2,
                    child=b3,
                    axis=Vector3.from_iterable([0, 0, 1]),
                    dof_name=dof.name,
                )
            )

        modifications = w._model_modification_blocks[-1]
        self.assertEqual(len(modifications.modifications), 13)

        add_body_modifications = [
            m
            for m in modifications.modifications
            if isinstance(m, AddKinematicStructureEntityModification)
        ]
        self.assertEqual(len(add_body_modifications), 3)

        add_dof_modifications = [
            m
            for m in modifications.modifications
            if isinstance(m, AddDegreeOfFreedomModification)
        ]
        self.assertEqual(len(add_dof_modifications), 8)

        add_connection_modifications = [
            m
            for m in modifications.modifications
            if isinstance(m, AddConnectionModification)
        ]
        self.assertEqual(len(add_connection_modifications), 2)

        # reconstruct this world
        w2 = World()

        # copy modifications
        modifications_copy = WorldModelModificationBlock.from_json(
            modifications.to_json()
        )
        modifications_copy.apply(w2)
        self.assertEqual(len(w2.bodies), 3)
        self.assertEqual(len(w2.connections), 2)

        with w.modify_world():
            w.remove_connection(w.connections[-1])
            w.remove_kinematic_structure_entity(
                w.get_kinematic_structure_entity_by_name("b3")
            )

        modifications = w._model_modification_blocks[-1]
        self.assertEqual(len(modifications.modifications), 3)

        modifications_copy = WorldModelModificationBlock.from_json(
            modifications.to_json()
        )
        modifications_copy.apply(w2)
        self.assertEqual(len(w2.bodies), 2)
        self.assertEqual(len(w2.connections), 1)

    def test_semantic_annotation_modifications(self):
        w = World()
        b1 = Body(name=PrefixedName("b1"))
        v1 = Handle(body=b1)
        v2 = Door(body=b1, handle=v1)

        add_v1 = AddSemanticAnnotationModification(v1)
        add_v2 = AddSemanticAnnotationModification(v2)

        self.assertNotIn(v1, w.semantic_annotations)
        self.assertNotIn(v2, w.semantic_annotations)

        with w.modify_world():
            add_v1.apply(w)
            add_v2.apply(w)

        self.assertIn(v1, w.semantic_annotations)
        self.assertIn(v2, w.semantic_annotations)

        rm_v1 = RemoveSemanticAnnotationModification(v1)
        rm_v2 = RemoveSemanticAnnotationModification(v2)
        with w.modify_world():
            rm_v1.apply(w)
            rm_v2.apply(w)

        self.assertNotIn(v1, w.semantic_annotations)
        self.assertNotIn(v2, w.semantic_annotations)


if __name__ == "__main__":
    unittest.main()
