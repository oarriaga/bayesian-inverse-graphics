import jax.numpy as jp

from .image import compute_image_colors


def compute_colors(points3D, image):
    u, v = planar_map(points3D)
    colors = compute_image_colors(u, v, image)
    return colors


def planar_map(points3D):
    x, y, z = jp.split(points3D, 3, axis=1)
    u = jp.remainder(x, 1)
    v = jp.remainder(z, 1)
    return u, v
