import jax.numpy as jp

from ..algebra import apply_hit_mask
from ..constants import EPSILON, FARAWAY


def intersect_canonical_plane(ray_origins, ray_directions):
    ray_origins_y = ray_origins[:, 1]
    ray_directions_y = ray_directions[:, 1]
    depths = - ray_origins_y / ray_directions_y
    hit_mask = jp.logical_and((depths > EPSILON), (depths < FARAWAY))
    depths = apply_hit_mask(hit_mask, depths)
    return hit_mask, None, jp.expand_dims(depths, -1)


def compute_canonical_normals_plane(shape_points):
    shape_normals = jp.array([[0.0, 1.0, 0.0]])
    shape_normals = jp.repeat(shape_normals, len(shape_points), axis=0)
    return shape_normals
