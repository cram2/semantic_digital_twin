from dataclasses import dataclass, field
from itertools import combinations
from typing_extensions import Optional, Set, List, Dict, Iterable

import fcl
from trimesh.collision import CollisionManager, mesh_to_BVH

from .collision_detector import CollisionDetector, CollisionCheck, Collision
from ..world_description.world_entity import Body


@dataclass
class TrimeshCollisionDetector(CollisionDetector):
    collision_manager: CollisionManager = field(
        default_factory=CollisionManager, init=False
    )
    """
    The collision manager from trimesh to handle collision detection
    """
    _last_synced_state: Optional[int] = field(default=None, init=False)
    """
    Last synced state version of the world
    """
    _last_synced_model: Optional[int] = field(default=None, init=False)
    """
    Last synced model version of the world
    """
    _collision_objects: Dict[Body, fcl.CollisionObject] = field(
        default_factory=dict, init=False
    )
    """
    The FCL collision objects for each body in the world
    """

    def sync_world_model(self) -> None:
        """
        Synchronize the collision checker with the current world model
        """
        if self._last_synced_model == self._world.get_world_model_manager().model_version:
            return
        bodies_to_be_added = set(self._world.bodies_with_enabled_collision) - set(
            self._collision_objects.keys()
        )
        for body in bodies_to_be_added:
            self._collision_objects[body] = fcl.CollisionObject(
                mesh_to_BVH(body.collision.combined_mesh),
                fcl.Transform(
                    body.global_pose.to_np()[:3, :3], body.global_pose.to_np()[:3, 3]
                ),
            )
        bodies_to_be_removed = set(self._collision_objects.keys()) - set(
            self._world.bodies_with_enabled_collision
        )
        for body in bodies_to_be_removed:
            del self._collision_objects[body]

    def sync_world_state(self) -> None:
        """
        Synchronize the collision checker with the current world state
        """
        if self._last_synced_state == self._world.state._state_version:
            return
        for body, coll_obj in self._collision_objects.items():
            coll_obj.setTransform(
                fcl.Transform(
                    body.global_pose.to_np()[:3, :3], body.global_pose.to_np()[:3, 3]
                )
            )

    def check_collisions(
        self, collision_matrix: Optional[Iterable[CollisionCheck]] = None
    ) -> List[Collision]:
        """
        Checks for collisions in the current world state. The collision manager from trimesh returns all collisions,
        which are then filtered based on the provided collision matrix. If there are multiple contacts between two bodies,
        only the first contact is returned.

        :param collision_matrix: An optional set of CollisionCheck objects to filter the collisions. If None is provided, all collisions are checked.
        :return: A list of Collision objects representing the detected collisions.
        """
        self.sync_world_model()
        self.sync_world_state()

        if collision_matrix is None:
            return []

        collision_pairs = [
            (cc.body_a, cc.body_b, cc.distance) for cc in collision_matrix
        ]
        result = []
        for body_a, body_b, distance in collision_pairs:
            if (
                body_a not in self._collision_objects
                or body_b not in self._collision_objects
            ):
                raise ValueError(
                    f"One of the bodies {body_a.name}, {body_b.name} does not have collision enabled or is not part of the world."
                )
            distance_request = fcl.DistanceRequest(
                enable_nearest_points=True, enable_signed_distance=True
            )
            distance_result = fcl.DistanceResult()
            fcl.distance(
                self._collision_objects[body_a],
                self._collision_objects[body_b],
                distance_request,
                distance_result,
            )
            if distance_result.min_distance <= distance:
                result.append(
                    Collision(
                        distance_result.min_distance,
                        body_a,
                        body_b,
                        map_P_pa=distance_result.nearest_points[0],
                        map_P_pb=distance_result.nearest_points[1],
                        map_V_n_input=distance_result.nearest_points[0]
                        - distance_result.nearest_points[1],
                    )
                )

        return result

    def check_collision_between_bodies(
        self, body_a: Body, body_b: Body
    ) -> Optional[Collision]:
        collision = self.check_collisions(
            {CollisionCheck(body_a, body_b, 0.0, self._world)}
        )
        return collision[0] if collision else None

    def reset_cache(self):
        pass
