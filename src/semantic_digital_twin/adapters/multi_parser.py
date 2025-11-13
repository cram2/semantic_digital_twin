import logging
import os
from dataclasses import dataclass

import numpy
from typing_extensions import Optional, Set, List

from ..exceptions import WorldEntityNotFoundError
from ..spatial_types.spatial_types import (
    TransformationMatrix,
    Quaternion,
    RotationMatrix,
    Point3,
)
from ..world_description.geometry import (
    Box,
    Sphere,
    Cylinder,
    Scale,
    Shape,
    Color,
    TriangleMesh,
)
from ..world_description.inertial_properties import (
    Inertial,
    InertiaTensor,
    PrincipalMoments,
    PrincipalAxes,
)
from ..world_description.connection_properties import JointDynamics
from ..world_description.shape_collection import ShapeCollection
from ..world_description.world_entity import KinematicStructureEntity

from multiverse_parser import (
    InertiaSource,
    UsdImporter,
    MjcfImporter,
    BodyBuilder,
    JointBuilder,
    JointType,
    Factory,
    GeomType,
)
from multiverse_parser.utils import get_relative_transform
from pxr import UsdUrdf, UsdMujoco, UsdGeom, UsdPhysics, Gf  # type: ignore

from ..world_description.connections import (
    RevoluteConnection,
    PrismaticConnection,
    FixedConnection,
    Connection6DoF,
)
from ..world_description.degree_of_freedom import DegreeOfFreedom
from ..datastructures.prefixed_name import PrefixedName
from ..spatial_types import spatial_types as cas
from ..spatial_types.derivatives import DerivativeMap
from ..world import World, Body, Connection


def usd_pose_to_cas_pose(usd_transform: Gf.Matrix4d) -> cas.TransformationMatrix:
    """
    Convert a USD Gf.Matrix4d transform to a cas.TransformationMatrix. This assumes that the USD transform does not contain any scale or shear.

    :param usd_transform: The USD Gf.Matrix4d transform to convert.
    :return: A cas.TransformationMatrix representing the same transformation.
    """
    translation: Gf.Vec3d = usd_transform.ExtractTranslation()  # type: ignore
    rotation: Gf.Rotation = usd_transform.ExtractRotation()  # type: ignore
    quat: Gf.Quatd = rotation.GetQuat()  # type: ignore
    return cas.TransformationMatrix.from_xyz_quaternion(
        pos_x=translation[0],
        pos_y=translation[1],
        pos_z=translation[2],
        quat_x=quat.GetImaginary()[0],
        quat_y=quat.GetImaginary()[1],
        quat_z=quat.GetImaginary()[2],
        quat_w=quat.GetReal(),
    )


def get_free_body_names(factory: Factory) -> Set[str]:
    """
    Get the names of all free bodies in the world.
    :param factory: The factory instance that contains the world builder.
    :return: A set of names of free bodies.
    """
    constrained_body_names = set()
    stage = factory.world_builder.stage
    for joint in [
        UsdPhysics.Joint(joint_prim)  # type: ignore
        for joint_prim in stage.TraverseAll()  # type: ignore
        if joint_prim.IsA(UsdPhysics.Joint)  # type: ignore
    ]:  # type: ignore
        for child_body_path in joint.GetBody1Rel().GetTargets():
            child_body_prim = stage.GetPrimAtPath(child_body_path)
            constrained_body_names.add(child_body_prim.GetName())
            constrained_body_names.update(
                child.GetName()
                for child in child_body_prim.GetAllChildren()
                if child.IsA(UsdGeom.Xform)  # type: ignore
            )

    free_body_names = {
        prim.GetName()
        for prim in stage.TraverseAll()
        if prim.IsA(UsdGeom.Xform)  # type: ignore
        and prim.HasAPI(UsdPhysics.MassAPI)  # type: ignore
        and prim.HasAPI(UsdPhysics.RigidBodyAPI)  # type: ignore
        and prim.GetName() not in constrained_body_names
    }

    return free_body_names


def parse_box(
    gprim: UsdGeom.Gprim, origin_transform: TransformationMatrix, color: Color
) -> Shape:
    """
    Parses a box geometry from a UsdGeom.Gprim instance.

    :param gprim: The UsdGeom.Gprim instance representing the box geometry.
    :param origin_transform: The origin transformation matrix for the box.
    :param color: The color of the box.
    :return: A Box shape representing the parsed box geometry.
    """
    size = (
        numpy.array(
            [gprim.GetLocalTransformation().GetRow(i).GetLength() for i in range(3)]
        )
        * 2
    )
    return Box(
        origin=origin_transform,
        scale=Scale(*size),
        color=color,
    )


def parse_sphere(sphere: UsdGeom.Sphere, origin_transform: TransformationMatrix, color: Color) -> Shape:  # type: ignore
    """
    Parses a sphere geometry from a UsdGeom.Sphere instance.

    :param sphere: The UsdGeom.Sphere instance representing the sphere geometry.
    :param origin_transform: The origin transformation matrix for the sphere.
    :param color: The color of the sphere.
    :return: A Sphere shape representing the parsed sphere geometry.
    """
    return Sphere(
        origin=origin_transform,
        radius=sphere.GetRadiusAttr().Get(),
        color=color,
    )


def parse_cylinder(cylinder: UsdGeom.Cylinder, origin_transform: TransformationMatrix, color: Color) -> Shape:  # type: ignore
    """
    Parses a cylinder geometry from a UsdGeom.Cylinder instance.

    :param cylinder: The UsdGeom.Cylinder instance representing the cylinder geometry.
    :param origin_transform: The origin transformation matrix for the cylinder.
    :param color: The color of the cylinder.
    :return: A Cylinder shape representing the parsed cylinder geometry.
    """
    return Cylinder(
        origin=origin_transform,
        width=cylinder.GetRadiusAttr().Get() * 2,
        height=cylinder.GetHeightAttr().Get(),
        color=color,
    )


def parse_plane(
    plane: UsdGeom.Mesh, origin_transform: TransformationMatrix, color: Color
) -> Shape:
    """
    Parses a plane geometry from a UsdGeom.Mesh instance.

    :param plane: The UsdGeom.Mesh instance representing the plane geometry.
    :param origin_transform: The origin transformation matrix for the plane.
    :param color: The color of the plane.
    :return: A Plane shape representing the parsed plane geometry.
    """
    size_x = plane.GetExtentAttr().Get()[1][0] - plane.GetExtentAttr().Get()[0][0]
    size_y = plane.GetExtentAttr().Get()[1][1] - plane.GetExtentAttr().Get()[0][1]
    size_z = plane.GetExtentAttr().Get()[1][2] - plane.GetExtentAttr().Get()[0][2]
    if numpy.isclose(size_z, 0.0):
        size_x = 0.0
        size_y = 0.0
        size_z = 0.05
    return Box(
        origin=origin_transform,
        scale=Scale(size_x, size_y, size_z),
        color=color,
    )


def parse_mesh(
    gprim: UsdGeom.Gprim,
    local_transformation: Gf.Matrix4d,
    translation: Gf.Vec3d,
    quat: Gf.Quatd,
) -> Shape:
    """
    Parses a mesh geometry from a UsdGeom.Gprim instance.

    :param gprim: The UsdGeom.Gprim instance representing the mesh geometry.
    :param local_transformation: The local transformation matrix of the mesh.
    :param translation: The translation vector of the mesh.
    :param quat: The quaternion representing the rotation of the mesh.
    :return: A TriangleMesh shape representing the parsed mesh geometry.
    """
    scale = [local_transformation.GetRow(i).GetLength() for i in range(3)]
    data = {"mesh": {}, "origin": {}, "scale": {}}
    data["mesh"]["vertices"] = gprim.GetVertices()
    data["mesh"]["faces"] = gprim.GetFaceVertexCounts()
    data["origin"]["position"] = [
        translation[0],
        translation[1],
        translation[2],
    ]
    data["origin"]["quaternion"] = [
        quat.GetReal(),
        quat.GetImaginary()[0],
        quat.GetImaginary()[1],
        quat.GetImaginary()[2],
    ]
    data["scale"]["x"] = scale[0]
    data["scale"]["y"] = scale[1]
    data["scale"]["z"] = scale[2]
    return TriangleMesh.from_json(data=data)


def parse_geometry(body_builder: BodyBuilder) -> tuple[List[Shape], List[Shape]]:
    """
    Parses the visual and collision geometry from a BodyBuilder instance.

    :param body_builder: The BodyBuilder instance to parse.
    :return: A tuple containing two lists: the first list contains the visual shapes, and the second list contains the collision shapes.
    """

    visuals = []
    collisions = []
    for geom_builder in body_builder.geom_builders:
        gprim = geom_builder.gprim
        gprim_prim = gprim.GetPrim()
        local_transformation: Gf.Matrix4d = (  # type: ignore
            gprim.GetLocalTransformation().RemoveScaleShear()
        )
        translation: Gf.Vec3d = local_transformation.ExtractTranslation()  # type: ignore
        rotation: Gf.Rotation = local_transformation.ExtractRotation()  # type: ignore
        quat: Gf.Quatd = rotation.GetQuat()  # type: ignore
        origin_transform = TransformationMatrix.from_xyz_quaternion(
            pos_x=translation[0],
            pos_y=translation[1],
            pos_z=translation[2],
            quat_x=quat.GetImaginary()[0],
            quat_y=quat.GetImaginary()[1],
            quat_z=quat.GetImaginary()[2],
            quat_w=quat.GetReal(),
        )
        color = Color(*geom_builder.rgba)
        shape = None
        match geom_builder.type:
            case GeomType.CUBE:
                shape = parse_box(gprim, origin_transform, color)
            case GeomType.SPHERE:
                shape = parse_sphere(UsdGeom.Sphere(gprim_prim), origin_transform, color)  # type: ignore
            case GeomType.CYLINDER:
                shape = parse_cylinder(UsdGeom.Cylinder(gprim_prim), origin_transform, color)  # type: ignore
            case GeomType.PLANE:
                shape = parse_plane(UsdGeom.Mesh(gprim_prim), origin_transform, color)  # type: ignore
            case GeomType.MESH:
                shape = parse_mesh(gprim, local_transformation, translation, quat)
        if shape is None:
            logging.warning(f"Geometry type {geom_builder.type} is not supported yet.")
        else:
            if gprim_prim.HasAPI(UsdPhysics.CollisionAPI):  # type: ignore
                collisions.append(shape)
            else:
                visuals.append(shape)
    return visuals, collisions


def parse_non_fixed_joint(
    world: World, joint_builder: JointBuilder
) -> tuple[KinematicStructureEntity, KinematicStructureEntity, TransformationMatrix]:
    """
    Parses a non-fixed joint from a JointBuilder instance.

    :param world: The World instance to get the kinematic structure entities from.
    :param joint_builder: The JointBuilder instance to parse.
    :return: A tuple containing the parent KinematicStructureEntity, the child KinematicStructure
    Entity, and the origin TransformationMatrix of the joint.
    """
    parent_body = world.get_kinematic_structure_entity_by_name(
        joint_builder.parent_prim.GetName()
    )
    child_body = world.get_kinematic_structure_entity_by_name(
        joint_builder.child_prim.GetName()
    )
    transform = get_relative_transform(
        from_prim=joint_builder.parent_prim, to_prim=joint_builder.child_prim
    )
    origin = usd_pose_to_cas_pose(transform)
    return parent_body, child_body, origin


def parse_fixed_joint(
    world: World, body_builder: BodyBuilder
) -> tuple[KinematicStructureEntity, KinematicStructureEntity, TransformationMatrix]:
    """
    Parses a fixed joint from a BodyBuilder instance.

    :param world: The World instance to get the kinematic structure entities from.
    :param body_builder: The BodyBuilder instance to parse.
    :return: A tuple containing the parent KinematicStructureEntity, the child KinematicStructure
    Entity, and the origin TransformationMatrix of the joint.
    """
    parent_body = world.get_kinematic_structure_entity_by_name(
        body_builder.xform.GetPrim().GetParent().GetName()
    )
    child_body = world.get_kinematic_structure_entity_by_name(
        body_builder.xform.GetPrim().GetName()
    )
    transform = body_builder.xform.GetLocalTransformation()
    origin = usd_pose_to_cas_pose(transform)
    return parent_body, child_body, origin


def parse_dof(
    world: World, free_variable_name: str, joint_builder: JointBuilder, joint_name: str
) -> DegreeOfFreedom:
    """
    Parses a degree of freedom from a JointBuilder instance.

    :param world: The World instance to get the degree of freedom from.
    :param free_variable_name: The name of the free variable associated with the degree of freedom
    :param joint_builder: The JointBuilder instance to parse.
    :param joint_name: The name of the joint associated with the degree of freedom.
    :return: A DegreeOfFreedom instance representing the parsed degree of freedom.
    """
    try:
        return world.get_degree_of_freedom_by_name(free_variable_name)
    except WorldEntityNotFoundError:
        if joint_builder.type == JointType.CONTINUOUS:
            dof = DegreeOfFreedom(
                name=PrefixedName(joint_name),
            )
        else:
            lower_limits = DerivativeMap()
            lower_limits.position = joint_builder.joint.GetLowerLimitAttr().Get()
            upper_limits = DerivativeMap()
            upper_limits.position = joint_builder.joint.GetUpperLimitAttr().Get()
            dof = DegreeOfFreedom(
                name=PrefixedName(joint_name),
                lower_limits=lower_limits,
                upper_limits=upper_limits,
            )
        world.add_degree_of_freedom(dof)
        return dof


def parse_inertial(body_builder: BodyBuilder) -> Optional[Inertial]:
    """
    Parses the inertial properties from a BodyBuilder instance.

    :param body_builder: The BodyBuilder instance to parse.
    :return: An Inertial instance representing the parsed inertial properties, or None if no inertial properties are found.
    """
    xform_prim = body_builder.xform.GetPrim()
    if not xform_prim.HasAPI(UsdPhysics.MassAPI):  # type: ignore
        return None
    physics_mass_api = UsdPhysics.MassAPI(xform_prim)  # type: ignore
    mass = physics_mass_api.GetMassAttr().Get()
    center_of_mass = physics_mass_api.GetCenterOfMassAttr().Get()
    center_of_mass = Point3.from_iterable(center_of_mass)
    principle_axes_quat = physics_mass_api.GetPrincipalAxesAttr().Get()
    principle_axes_quat = Quaternion.from_iterable(
        [principle_axes_quat.GetReal(), *principle_axes_quat.GetImaginary()]
    )
    principle_moments = physics_mass_api.GetDiagonalInertiaAttr().Get()
    inertia_tensor = InertiaTensor.from_principal_moments_and_axes(
        moments=PrincipalMoments.from_values(*principle_moments),
        axes=PrincipalAxes.from_rotation_matrix(
            RotationMatrix.from_quaternion(principle_axes_quat)
        ),
    )
    inertial = Inertial(
        mass=mass, center_of_mass=center_of_mass, inertia=inertia_tensor
    )
    return inertial


@dataclass
class MultiParser:
    """
    Class to parse any scene description file to World.
    """

    file_path: str
    """
    The file path of the scene.
    """

    prefix: Optional[str] = None
    """
    The prefix for every name used in this world.
    """

    def __post_init__(self):
        if self.prefix is None:
            self.prefix = os.path.basename(self.file_path).split(".")[0]

    def create_factory(
        self,
        fixed_base: bool = True,
        root_name: Optional[str] = None,
        with_physics=True,
        with_visual=True,
        with_collision=True,
        inertia_source=InertiaSource.FROM_SRC,
        default_rgba=numpy.array([0.9, 0.9, 0.9, 1.0]),
    ) -> Factory:
        """
        Creates a Factory instance for the specific parser.

        :param fixed_base: Whether to fix the base of the root body.
        :param root_name: The name of the root body.
        :param with_physics: Whether to include physics properties.
        :param with_visual: Whether to include visual geometry.
        :param with_collision: Whether to include collision geometry.
        :param inertia_source: The source of inertia properties.
        :param default_rgba: The default RGBA color for visual geometry.
        :return: A Factory instance for the specific parser.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def parse(self, fixed_base=True) -> World:
        """
        Parses the file at `file_path` and returns a World instance.

        :param fixed_base: Whether to fix the base of the root body.
        :return: A World instance representing the parsed scene. The root will be named "world", regardless of the original root name.
        """
        factory = self.create_factory()
        factory.import_model()
        bodies = [
            self.parse_body(body_builder)
            for body_builder in factory.world_builder.body_builders
        ]
        world = World()

        root = bodies[0]
        if root.name.name != "world":
            with world.modify_world():
                root = Body(name=PrefixedName("world"))
                world.add_body(root)

        with world.modify_world():
            for body in bodies:
                world.add_kinematic_structure_entity(body)
            joints = []
            for body_builder in factory.world_builder.body_builders:
                joints += self.parse_joints(body_builder=body_builder, world=world)
            for joint in joints:
                world.add_connection(joint)

            free_body_names = get_free_body_names(factory=factory)

            for free_body_name in free_body_names:
                body = world.get_body_by_name(free_body_name)
                if body.name.name == root.name.name:
                    continue
                if fixed_base:
                    joint = FixedConnection(parent=root, child=body)
                else:
                    joint = Connection6DoF.create_with_dofs(
                        world=world, parent=root, child=body
                    )
                world.add_connection(joint)

        return world

    def parse_joints(self, body_builder: BodyBuilder, world: World) -> list[Connection]:
        """
        Parses joints from a BodyBuilder instance.
        :param body_builder: The BodyBuilder instance to parse.
        :param world: The World instance to add the connections to.
        :return: A list of Connection instances representing the parsed joints.
        """
        connections = []
        for joint_builder in body_builder.joint_builders:
            parent_body, child_body, origin = parse_non_fixed_joint(
                world, joint_builder
            )
            connection = self.parse_joint(
                joint_builder, parent_body, child_body, origin, world
            )
            connections.append(connection)
        if (
            len(body_builder.joint_builders) == 0
            and not body_builder.xform.GetPrim().GetParent().IsPseudoRoot()
        ):
            parent_body, child_body, origin = parse_fixed_joint(world, body_builder)
            connection = FixedConnection(
                parent=parent_body,
                child=child_body,
                parent_T_connection_expression=origin,
            )
            connections.append(connection)

        return connections

    def parse_joint(
        self,
        joint_builder: JointBuilder,
        parent_body: KinematicStructureEntity,
        child_body: KinematicStructureEntity,
        origin: TransformationMatrix,
        world: World,
    ) -> Connection:
        """
        Parses a joint from a JointBuilder instance.

        :param joint_builder: The JointBuilder instance to parse.
        :param parent_body: The parent KinematicStructureEntity instance of the joint.
        :param child_body: The child KinematicStructureEntity instance of the joint.
        :param origin: The origin TransformationMatrix of the joint.
        :param world: The World instance to add the connections to.
        :return: A Connection instance representing the parsed joint.
        """
        joint_prim = joint_builder.joint.GetPrim()
        joint_name = joint_prim.GetName()
        free_variable_name = joint_name
        offset = None
        multiplier = None
        if joint_prim.HasAPI(UsdUrdf.UrdfJointAPI):  # type: ignore
            urdf_joint_api = UsdUrdf.UrdfJointAPI(joint_prim)  # type: ignore
            if len(urdf_joint_api.GetJointRel().GetTargets()) > 0:
                free_variable_name = urdf_joint_api.GetJointRel().GetTargets()[0].name
                offset = urdf_joint_api.GetOffsetAttr().Get()
                multiplier = urdf_joint_api.GetMultiplierAttr().Get()

        armature = 0.0
        dry_friction = 0.0
        damping = 0.0
        if joint_prim.HasAPI(UsdMujoco.MujocoJointAPI):  # type: ignore
            mujoco_joint_api = UsdMujoco.MujocoJointAPI(joint_prim)  # type: ignore
            armature = mujoco_joint_api.GetArmatureAttr().Get()
            dry_friction = mujoco_joint_api.GetFrictionlossAttr().Get()
            damping = mujoco_joint_api.GetDampingAttr().Get()
        match joint_builder.type:
            case JointType.FREE:
                raise NotImplementedError("Free joints are not supported yet.")
            case JointType.FIXED:
                return FixedConnection(
                    parent=parent_body,
                    child=child_body,
                    parent_T_connection_expression=origin,
                )
            case JointType.REVOLUTE | JointType.CONTINUOUS | JointType.PRISMATIC:
                axis = cas.Vector3(
                    float(joint_builder.axis.to_array()[0]),
                    float(joint_builder.axis.to_array()[1]),
                    float(joint_builder.axis.to_array()[2]),
                    reference_frame=parent_body,
                )
                dof = parse_dof(
                    world=world,
                    free_variable_name=free_variable_name,
                    joint_builder=joint_builder,
                    joint_name=joint_name,
                )
                if joint_builder.type in [JointType.REVOLUTE, JointType.CONTINUOUS]:
                    JointConnection = RevoluteConnection
                else:
                    JointConnection = PrismaticConnection
                joint_prop = JointDynamics(
                    armature=armature,
                    dry_friction=dry_friction,
                    damping=damping,
                )
                return JointConnection(
                    name=PrefixedName(joint_name),
                    parent=parent_body,
                    child=child_body,
                    parent_T_connection_expression=origin,
                    multiplier=multiplier,
                    offset=offset,
                    axis=axis,
                    dof_name=dof.name,
                    dynamics=joint_prop,
                )
        raise NotImplementedError(
            f"Joint type {joint_builder.type} is not supported yet."
        )

    def parse_body(self, body_builder: BodyBuilder) -> Body:
        """
        Parses a body from a BodyBuilder instance.

        :param body_builder: The BodyBuilder instance to parse.
        :return: A Body instance representing the parsed body.
        """
        name = PrefixedName(
            prefix=self.prefix, name=body_builder.xform.GetPrim().GetName()
        )
        visuals, collisions = parse_geometry(body_builder)
        result = Body(name=name)
        inertial = parse_inertial(body_builder)
        if inertial is not None:
            result.inertial = inertial
        visuals = ShapeCollection(visuals, reference_frame=result)
        collisions = ShapeCollection(collisions, reference_frame=result)
        result.visual = visuals
        result.collision = collisions
        return result


class MJCFParser(MultiParser):
    """
    Class to parse MJCF scene description files to worlds.
    """

    def create_factory(
        self,
        fixed_base: bool = True,
        root_name: Optional[str] = None,
        with_physics=True,
        with_visual=True,
        with_collision=True,
        inertia_source=InertiaSource.FROM_SRC,
        default_rgba=numpy.array([0.9, 0.9, 0.9, 1.0]),
    ) -> Factory:
        if root_name is None:
            root_name = "world"
        return MjcfImporter(
            file_path=self.file_path,
            fixed_base=False,
            root_name=root_name,
            with_physics=with_physics,
            with_visual=with_visual,
            with_collision=with_collision,
            inertia_source=inertia_source,
            default_rgba=default_rgba,
        )


class USDParser(MultiParser):
    """
    Class to parse USD scene description files to worlds.
    """

    def create_factory(
        self,
        fixed_base: bool = True,
        root_name: Optional[str] = None,
        with_physics=True,
        with_visual=True,
        with_collision=True,
        inertia_source=InertiaSource.FROM_SRC,
        default_rgba=numpy.array([0.9, 0.9, 0.9, 1.0]),
    ) -> Factory:
        return UsdImporter(
            file_path=self.file_path,
            fixed_base=True,
            root_name=root_name,
            with_physics=with_physics,
            with_visual=with_visual,
            with_collision=with_collision,
            inertia_source=inertia_source,
            default_rgba=default_rgba,
        )
