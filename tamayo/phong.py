import jax
import jax.numpy as jp

from tamayo.algebra import (
    dot, compute_hits_to_light, compute_reflections_dot_eye)

from .patterns import empty_pattern
from .patterns import spherical
from .patterns import planar
from .algebra import transform_points


def compute_colors_in_shape(pattern_transform, shape_transform, points_world):
    world_to_shape = jp.linalg.inv(shape_transform)
    points_shape = transform_points(world_to_shape, points_world)

    shape_to_pattern = jp.linalg.inv(pattern_transform)
    points_pattern = transform_points(shape_to_pattern, points_shape)
    return points_pattern


def compute_pattern_colors(shape, points):
    pattern_transform = shape.pattern.transform
    pattern_image = shape.pattern.image
    points = compute_colors_in_shape(
        pattern_transform, shape.transform, points)
    cases = [empty_pattern.compute_colors, spherical.compute_colors, planar.compute_colors]
    pattern_colors = jax.lax.switch(shape.pattern.type, cases, points, pattern_image)
    return pattern_colors


def compute_material_colors(material, light, points):
    return jp.full_like(points, material.color * light.intensity)


def compute_base_color(shape, material, light, points):
    pattern_colors = compute_pattern_colors(shape, points)
    material_colors = compute_material_colors(material, light, points)
    base_colors = pattern_colors + material_colors
    return base_colors


def compute_ambient(shape, material, light, points):
    base_color = compute_base_color(shape, material, light, points)
    ambient = base_color * material.ambient
    return ambient


def compute_diffuse(shape, material, light, points, normals):
    hits_to_light = compute_hits_to_light(light.position, points)
    lambertian = dot(hits_to_light, normals)
    lambertian = jp.maximum(lambertian, 0.0)
    # base_color (3) light_dot_normal (num_rays, 1) outer product (num_rays, 3)
    lambertian = jp.expand_dims(lambertian, -1)
    base_color = compute_base_color(shape, material, light, points)
    return base_color * material.diffuse * lambertian


def compute_specular(material, light, points, normals, eye):
    reflections = compute_reflections_dot_eye(light, points, normals, eye)
    factor = jp.power(reflections, material.shininess)
    # light.intensity (3) factor (num_rays, 1) outer product (num_rays, 3)
    factor = jp.expand_dims(factor, -1)
    specular = light.intensity * material.specular * factor
    return specular


def compute_colors(shape, material, points, normals, eye, light):
    ambient = compute_ambient(shape, material, light, points)
    diffuse = compute_diffuse(shape, material, light, points, normals)
    specular = compute_specular(material, light, points, normals, eye)
    return ambient + diffuse + specular


def compute_colors_with_shadow(shape, material, points,
                               normals, eye, light, shadow_mask):
    ambient = compute_ambient(shape, material, light, points)
    diffuse = compute_diffuse(shape, material, light, points, normals)
    specular = compute_specular(material, light, points, normals, eye)
    colors = ambient + diffuse + specular
    shadow_mask = jp.expand_dims(shadow_mask, 1)
    color = jp.where(shadow_mask, ambient, colors)
    return color
