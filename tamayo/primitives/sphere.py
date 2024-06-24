import jax.numpy as jp

from ..algebra import (solve_quadratic, dot, apply_hit_mask,
                       compute_quadratic_is_hit, compute_quadratic_depths)


def compute_canonical_normals_sphere(points):
    normals = points - jp.array([0.0, 0.0, 0.0])
    return normals


def intersect_canonical_sphere(ray_origin, ray_directions):
    origin_minus_center = ray_origin - jp.array([0.0, 0.0, 0.0])
    a = dot(ray_directions, ray_directions)
    b = 2.0 * dot(ray_directions, origin_minus_center)
    c = dot(origin_minus_center, origin_minus_center) - 1.0
    depths_A, depths_B, is_valid = solve_quadratic(a, b, c)
    hit_mask = compute_quadratic_is_hit(depths_A, depths_B, is_valid)
    depth = compute_quadratic_depths(depths_A, depths_B)
    depth = apply_hit_mask(hit_mask, depth)
    depth = jp.expand_dims(depth, axis=-1)
    return hit_mask, None, depth
