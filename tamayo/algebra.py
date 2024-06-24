import jax.numpy as jp

from .constants import FARAWAY


def dot(vectors_A, vectors_B):
    """Computes dot product between vectors_A and vectors_B

    # Arguments
        vectors_A: Array (num_rays, 3)
        vectors_B: Array (num_rays, 3)

    # Returns
        Array (num_rays)
    """
    return jp.sum(vectors_A * vectors_B, axis=-1)


def compute_norm(vectors):
    return jp.linalg.norm(vectors, axis=-1, keepdims=True)


def normalize(vectors):
    """Normalizes vectors across last dimension

    # Arguments
        vectors: Array (num_vectors, 3)

    # Returns
        normalized vectors: Array (num_vectors, 3)
    """
    norms = compute_norm(vectors)
    vectors = vectors / norms
    return vectors


def solve_quadratic(a, b, c):
    """Solves quadratic equation

    # Arguments
        a: Array
        b: Array
        c: Array

    # Returns
        solution_A: Array
        solution_B: Array
        valid_mask: Boolean array
    """
    discriminator = (b**2) - (4.0 * a * c)
    valid_mask = discriminator > 0  # >= is bad for automatic differentiation
    discriminator = jp.where(valid_mask, discriminator, 0.0)
    sqrt_discriminator = jp.sqrt(discriminator)
    solution_A = (-b - sqrt_discriminator) / (2.0 * a)
    solution_B = (-b + sqrt_discriminator) / (2.0 * a)
    return solution_A, solution_B, valid_mask


def add_zeros(vectors):
    """Adds zeros to vectors in R^3 across last dimension

    # Arguments
        vectors: Array (num_vectors, 3)

    # Returns
        vectors: Array (num_vectors, 4)
    """
    zeros = jp.zeros((len(vectors), 1))
    vectors = jp.concatenate([vectors, zeros], axis=-1)
    return vectors


def add_ones(points):
    """Adds ones to vectors in R^3 across last dimension

    # Arguments
        vectors: Array (num_vectors, 3)

    # Returns
        vectors: Array (num_vectors, 4)
    """
    ones = jp.ones((len(points), 1))
    points = jp.concatenate([points, ones], axis=-1)
    return points


def transform_points(affine_transform, points):
    """Transform R^3 points with affine matrix

    # Arguments
        affine_matrix: (4, 4)
        rays: (num_rays, 3)

    # Returns
        Transformed vectors (num_rays, 3)
    """
    points = add_ones(points)
    points = jp.matmul(affine_transform, points.T).T
    return points[:, :3]


def transform_vectors(affine_matrix, vectors):
    """Transform R^3 vectors with affine matrix

    # Arguments
        affine_matrix: (4, 4)
        rays: (num_rays, 3)

    # Returns
        Transformed vectors (num_rays, 3)
    """
    vectors = add_zeros(vectors)
    vectors = jp.matmul(affine_matrix, vectors.T).T
    vectors = vectors[:, :3]
    return vectors


def compute_quadratic_is_hit(depths_A, depths_B, is_valid):
    """Computes is hit mask from quadratic depths solution

    # Arguments
        depths_A: Array
        depths_B: Array
        is_Valid: Array

    # Returns
        is_hit: Array

    # Notes
        TODO: check if it should not be >= instead of just >
    """
    is_positive = jp.logical_or(depths_A > 0, depths_B > 0)
    is_hit = jp.logical_and(is_valid, is_positive)
    return is_hit


def apply_hit_mask(hit_mask, depths):
    """Applies hit mask to depths

    # Arguments
        hit_mask: Array
        depths: Array

    # Returns
        depths: Array
    """
    depth = jp.where(hit_mask, depths, FARAWAY)
    return depth


def compute_quadratic_depths(depths_A, depths_B):
    """Computes closest positive depth from quadratic solutions

    # Arguments
        depth_A: Array
        depth_B: Array
        hit_mask: Array

    # Returns
        depth: Array
    """
    choose_A = jp.logical_and((depths_A > 0), (depths_A < depths_B))
    depth = jp.where(choose_A, depths_A, depths_B)
    return depth


def replace_misses(depth, hit_mask):
    """Replaces depths where rays miss hit with faraway values

    # Arguments
        depth: Array
        hit_mask:

    # Return
        depth: Array
    """
    depth = jp.where(hit_mask, depth, FARAWAY)
    return depth


def compute_points3D(origins, directions, depth):
    """Compute positions of 3D points i.e. pointcloud

    # Arguments
        ray_origins: Array (num_rays, 3)
        ray_directions: Array (num_rays, 3)
        depth: Array (???)

    # Returns
        Array (num_rays, 3)
    """
    position = origins + (depth * directions)
    return position


def compute_hits_to_light(light_position, hits):
    hits_to_light = light_position - hits
    hits_to_light = normalize(hits_to_light)
    return hits_to_light


def reflect(light_directions, normals):
    return light_directions - (
        normals * 2 * jp.expand_dims(dot(light_directions, normals), -1))


def compute_reflections_dot_eye(light, points, normals, eye):
    hits_to_light = compute_hits_to_light(light.position, points)
    reflections = reflect(-hits_to_light, normals)
    reflections_dot_eye = dot(reflections, eye)
    reflections_dot_eye = jp.maximum(reflections_dot_eye, 0.0)
    return reflections_dot_eye


def transform_rays(affine_transform, ray_origins, ray_directions):
    ray_origin = transform_points(affine_transform, ray_origins)
    ray_directions = transform_vectors(affine_transform, ray_directions)
    return ray_origin, ray_directions


def sort_depths(depths):
    depths = jp.vstack(depths)
    depths = jp.sort(depths, axis=0)
    return depths


def to_column(vector):
    return jp.reshape(vector, (-1, 1))
