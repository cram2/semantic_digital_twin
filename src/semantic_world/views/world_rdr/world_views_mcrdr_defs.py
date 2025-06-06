from typing_extensions import List, Set, Union
from ..views import Cabinet, Container, Door, Drawer, Fridge, Handle, Roots, Table, Windows
from ...connections import FixedConnection, PrismaticConnection, RevoluteConnection
from ...world import World


def conditions_90574698325129464513441443063592862114(case) -> bool:
    return True


def conclusion_90574698325129464513441443063592862114(case) -> List[Handle]:
    def get_value_for_world_views_of_type_handle(case: World) -> Union[set, list, Handle]:
        """Get possible value(s) for World.views of types list/set of Handle"""
        return [Handle(b) for b in case.bodies if "handle" in b.name.name.lower()]
    
    return get_value_for_world_views_of_type_handle(case)


def conditions_14920098271685635920637692283091167284(case) -> bool:
    return len([v for v in case.views if type(v) is Handle]) > 0


def conclusion_14920098271685635920637692283091167284(case) -> List[Container]:
    def get_value_for_world_views_of_type_container(case: World) -> Union[set, Container, list]:
        """Get possible value(s) for World.views of types list/set of Container"""
        prismatic_connections = [c for c in case.connections if isinstance(c, PrismaticConnection)]
        fixed_connections = [c for c in case.connections if isinstance(c, FixedConnection)]
        children_of_prismatic_connections = [c.child for c in prismatic_connections]
        handles = [v for v in case.views if type(v) is Handle]
        fixed_connections_with_handle_child = [fc for fc in fixed_connections if fc.child in [h.body for h in handles]]
        drawer_containers = set(children_of_prismatic_connections).intersection(
            set([fc.parent for fc in fixed_connections_with_handle_child]))
        return [Container(b) for b in drawer_containers]
    
    return get_value_for_world_views_of_type_container(case)


def conditions_331345798360792447350644865254855982739(case) -> bool:
    return len([v for v in case.views if type(v) is Handle]) > 0 and len(
        [v for v in case.views if type(v) is Container]) > 0


def conclusion_331345798360792447350644865254855982739(case) -> List[Drawer]:
    def get_value_for_world_views_of_type_drawer(case: World) -> Union[set, list, Drawer]:
        """Get possible value(s) for World.views of types list/set of Drawer"""
        handles = [v for v in case.views if type(v) is Handle]
        containers = [v for v in case.views if type(v) is Container]
        fixed_connections = [c for c in case.connections if
                             isinstance(c, FixedConnection) and c.parent in [cont.body for cont in
                                                                             containers] and c.child in [
                                 h.body for h in handles]]
        prismatic_connections = [c for c in case.connections if
                                 isinstance(c, PrismaticConnection) and c.child in [cont.body for cont in containers]]
        drawer_handle_connections = [fc for fc in fixed_connections if
                                     fc.parent in [pc.child for pc in prismatic_connections]]
        drawers = [Drawer([cont for cont in containers if dc.parent == cont.body][0],
                          [h for h in handles if dc.child == h.body][0]) for dc in drawer_handle_connections]
        return drawers
    
    return get_value_for_world_views_of_type_drawer(case)


def conditions_35528769484583703815352905256802298589(case) -> bool:
    return len([v for v in case.views if type(v) is Drawer]) > 0


def conclusion_35528769484583703815352905256802298589(case) -> List[Cabinet]:
    def get_value_for_world_views_of_type_cabinet(case: World) -> Union[set, Cabinet, list]:
        """Get possible value(s) for World.views of types list/set of Cabinet"""
        drawers = [v for v in case.views if type(v) is Drawer]
        prismatic_connections = [c for c in case.connections if
                                 isinstance(c, PrismaticConnection) and c.child in [drawer.container.body for drawer in
                                                                                    drawers]]
        cabinet_container_bodies = [pc.parent for pc in prismatic_connections]
        cabinets = []
        for ccb in cabinet_container_bodies:
            if ccb in [cabinet.container.body for cabinet in cabinets]:
                continue
            cc_prismatic_connections = [pc for pc in prismatic_connections if pc.parent is ccb]
            cabinet_drawer_container_bodies = [pc.child for pc in cc_prismatic_connections]
            cabinet_drawers = [d for d in drawers if d.container.body in cabinet_drawer_container_bodies]
            cabinets.append(Cabinet(Container(ccb), cabinet_drawers))
    
        return cabinets
    
    return get_value_for_world_views_of_type_cabinet(case)


def conditions_59112619694893607910753808758642808601(case) -> bool:
    def conditions_for_world_views_of_type_door(case: World) -> bool:
        """Get conditions on whether it's possible to conclude a value for World.views  of type Door."""
        return len([v for v in case.views if isinstance(v, Handle)]) > 0
    return conditions_for_world_views_of_type_door(case)


def conclusion_59112619694893607910753808758642808601(case) -> List[Door]:
    def world_views_of_type_door(case: World) -> List[Door]:
        """Get possible value(s) for World.views  of type Door."""
        handles = [v for v in case.views if isinstance(v, Handle)]
        handle_bodies = [h.body for h in handles]
        connections_with_handles = [c for c in case.connections if isinstance(c, FixedConnection) and
                                    c.child in handle_bodies]
    
        revolute_connections = [c for c in case.connections if isinstance(c, RevoluteConnection)]
        bodies_connected_to_handles = [c.parent if c.child in handle_bodies else c.child for c in connections_with_handles]
        bodies_that_have_revolute_joints = [b for b in bodies_connected_to_handles for c in revolute_connections
                                            if b == c.child]
        body_handle_connections = [c for c in connections_with_handles if c.parent in bodies_that_have_revolute_joints]
        doors = [Door(c.parent, [h for h in handles if h.body == c.child][0]) for c in body_handle_connections]
        return doors
    return world_views_of_type_door(case)


def conditions_10840634078579061471470540436169882059(case) -> bool:
    def conditions_for_world_views_of_type_fridge(case: World) -> bool:
        """Get conditions on whether it's possible to conclude a value for World.views  of type Fridge."""
        return True
    return conditions_for_world_views_of_type_fridge(case)


def conclusion_10840634078579061471470540436169882059(case) -> List[Fridge]:
    def world_views_of_type_fridge(case: World) -> List[Fridge]:
        """Get possible value(s) for World.views  of type Fridge."""
        # Get fridge-related doors
        doors = [v for v in case.views if isinstance(v, Door) and "fridge" in v.body.name.name.lower()]
        # Precompute bodies of the fridge doors
        door_bodies = [d.body for d in doors]
        # Filter relevant revolute connections
        door_connections = [
            c for c in case.connections
            if isinstance(c, RevoluteConnection)
               and c.child in door_bodies
               and 'fridge' in c.parent.name.name.lower()
        ]
        return [Fridge(c.parent, doors[door_bodies.index(c.child)]) for c in door_connections]
    return world_views_of_type_fridge(case)


def conditions_100363513934529269865524874913381333593(case) -> bool:
    def conditions_for_world_views_of_type_roots(case: World) -> bool:
        """Get conditions on whether it's possible to conclude a value for World.views  of type Roots."""
        return len([Roots(r) for r in case.bodies if "root" in r.name.name.lower()]) > 0
    return conditions_for_world_views_of_type_roots(case)


def conclusion_100363513934529269865524874913381333593(case) -> List[Roots]:
    def world_views_of_type_roots(case: World) -> Roots:
        return [Roots(r) for r in case.bodies if "root" in r.name.name.lower()]
    return world_views_of_type_roots(case)


def conditions_32110951838731034027817851716447618523(case) -> bool:
    def conditions_for_world_views_of_type_table(case: World) -> bool:
        """Get conditions on whether it's possible to conclude a value for World.views  of type Table."""
        return len([r for r in case.views if isinstance(r, Roots)]) > 0
    return conditions_for_world_views_of_type_table(case)


def conclusion_32110951838731034027817851716447618523(case) -> List[Table]:
    def world_views_of_type_table(case: World) -> Table:
        """Get possible value(s) for World.views  of type Table."""
        all_roots = [r for r in case.views if isinstance(r, Roots)]
        root_bodies = [r.body for r in all_roots]
        conections_with_roots = [
            c for c in case.connections
            if isinstance(c, FixedConnection)
               and c.parent in root_bodies
        ]
        return [Table(b.child) for b in conections_with_roots if "table" in b.child.name.name.lower()]
    return world_views_of_type_table(case)


def conditions_284429630184552508120710178948116682797(case) -> bool:
    def conditions_for_world_views_of_type_windows(case: World) -> bool:
        """Get conditions on whether it's possible to conclude a value for World.views  of type Windows."""
        return len([r for r in case.views if isinstance(r, Roots)]) > 0
    return conditions_for_world_views_of_type_windows(case)


def conclusion_284429630184552508120710178948116682797(case) -> List[Windows]:
    def world_views_of_type_windows(case: World) -> Windows:
        """Get possible value(s) for World.views  of type Windows."""
        all_roots = [r for r in case.views if isinstance(r, Roots)]
        root_bodies = [r.body for r in all_roots]
        conections_with_roots = [
            c for c in case.connections
            if isinstance(c, FixedConnection)
               and c.parent in root_bodies
        ]
        return [Windows(b.child) for b in conections_with_roots if "windows" in b.child.name.name.lower()]
    return world_views_of_type_windows(case)


