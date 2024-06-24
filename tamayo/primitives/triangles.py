import jax.numpy as jp

from ..algebra import dot, normalize
from ..constants import FARAWAY, EPSILON


def extract_points(vertices, faces):
    points_A = vertices[faces[:, 0]]
    points_B = vertices[faces[:, 1]]
    points_C = vertices[faces[:, 2]]
    return points_A, points_B, points_C


def intersect_canonical_shape(vertices, faces, ray_origins, ray_directions):
    points_A, points_B, points_C = extract_points(vertices, faces)
    edges_AB = points_B - points_A
    edges_AC = points_C - points_A
    edges_AC = jp.expand_dims(edges_AC, axis=1)
    directions_cross_edges_AC = jp.cross(ray_directions, edges_AC)
    edges_AB = jp.expand_dims(edges_AB, axis=1)
    determinants = dot(edges_AB, directions_cross_edges_AC)
    f = 1.0 / (determinants + EPSILON)
    points_A = jp.expand_dims(points_A, 1)
    points_1_to_origin = ray_origins - points_A
    u = f * dot(points_1_to_origin, directions_cross_edges_AC)
    hit_mask_u = jp.logical_not(jp.logical_or(u < 0.0, u > 1.0))

    origins_cross_edge_1 = jp.cross(points_1_to_origin, edges_AB)
    v = f * dot(ray_directions, origins_cross_edge_1)
    hit_mask_v = jp.logical_not(jp.logical_or(v < 0.0, (u + v) > 1.0))
    hit_mask = jp.logical_and(hit_mask_u, hit_mask_v)

    valid_mask = jp.logical_not(jp.abs(determinants) < EPSILON)
    hit_mask = jp.logical_and(hit_mask, valid_mask)
    depth = f * dot(edges_AC, origins_cross_edge_1)
    depth = jp.where(hit_mask, depth, FARAWAY)
    return hit_mask, depth, depth


def compute_canonical_normals(vertices, faces, points):
    points_A, points_B, points_C = extract_points(vertices, faces)
    edges_AB = points_B - points_A
    edges_AC = points_C - points_A
    normals = jp.cross(edges_AC, edges_AB)
    normals = normalize(normals)
    num_triangles, num_rays, num_dimensions = points.shape
    normals = jp.expand_dims(normals, 1)
    normals = jp.repeat(normals, num_rays, axis=1)
    return normals
