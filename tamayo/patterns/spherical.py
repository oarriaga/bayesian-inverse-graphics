import jax.numpy as jp

from .image import compute_image_colors
from ..algebra import compute_norm


def spherical_map(points3D):
    x, y, z = jp.split(points3D, 3, axis=1)
    theta = jp.arctan2(x, z)
    radii = compute_norm(points3D)
    phi = jp.arccos(y / radii)

    raw_u = theta / (2 * jp.pi)
    u = 1 - (raw_u + 0.5)
    v = 1 - (phi / jp.pi)
    return u, v


def compute_colors(points3D, image):
    u, v = spherical_map(points3D)
    colors = compute_image_colors(u, v, image)
    return colors
