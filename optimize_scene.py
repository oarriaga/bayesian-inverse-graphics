import os
from functools import partial
import argparse
from collections import namedtuple

import jax
import jax.numpy as jp
import optax

from tamayo import PointLight, SE3, merge_shapes
from tamayo.render import Render
from tamayo.camera import build_rays
from tamayo.abstract import Material, Pattern, Shape
from tamayo.constants import (
    NO_PATTERN, PLANAR_PATTERN, PLANE, SPHERE, CUBE, CYLINDER)

from primitives import load, flatten, parse_metadata, parse_label
from logger import (write_dictionary, build_directory, print_progress,
                    write_pytree, make_directory)
from plotter import write_losses, write_image


SHAPE_TYPES = [SPHERE, CUBE, CYLINDER]
NAME_TO_TYPE = dict(zip(['sphere', 'cube', 'cylinder'], SHAPE_TYPES))
ARG_TO_TYPE = dict(zip([0, 1, 2], SHAPE_TYPES))
NAME_TO_RGB = {'gray': [87, 87, 87], 'red': [173, 35, 35],
               'blue': [42, 75, 215], 'green': [29, 105, 20],
               'brown': [129, 74, 25], 'purple': [129, 38, 192],
               'cyan': [41, 208, 208], 'yellow': [255, 238, 51]}
FLOOR = namedtuple('FLOOR', ['color', 'ambient', 'diffuse', 'image'])
LANDSCAPE_VARIABLE = namedtuple('SCENE', ['floor', 'lights'])
MATERIAL_FIELDS = ['color', 'ambient', 'diffuse', 'specular', 'shininess']
MATERIAL_VARIABLE = namedtuple('MATERIAL_VARIABLE', MATERIAL_FIELDS)
VARIABLE_STATE = namedtuple('VARIABLE_STATE', ['optimizer_state', 'variable'])

parser = argparse.ArgumentParser(description='scene optimization')
parser.add_argument('--seed', default=777, type=int)
parser.add_argument('--dataset_name', default='PRIMITIVES', type=str)
parser.add_argument('--dataset_path', default='datasets', type=str)
parser.add_argument('--shapes_directory', default='shapes', type=str)
parser.add_argument('--images_directory', default='images', type=str)
parser.add_argument('--viewport_factor', default=0.25, type=float)
parser.add_argument('--root', default='experiments', type=str)
parser.add_argument('--split', default='train', type=str)
parser.add_argument('--label', default='SCENE-OPTIMIZATION', type=str)
parser.add_argument('--num_lights', default=5, type=int)
parser.add_argument('--pattern_shape', nargs='+', default=[200, 200, 3])
parser.add_argument('--outer_epochs', default=7, type=int)
parser.add_argument('--inner_epochs', default=5, type=int)
parser.add_argument('--shadow', default=True, type=bool)
parser.add_argument('--learning_rate', default=0.01, type=float)
parser.add_argument('--ambient', default=0.1, type=float)
parser.add_argument('--diffuse', default=0.9, type=float)
parser.add_argument('--specular', default=0.5, type=float)
parser.add_argument('--shininess', default=4.0, type=float)
args = parser.parse_args()


def build_render(camera_origin, size, y_FOV, shadows):
    camera_origin = jp.array(camera_origin)
    camera_target = jp.array([0.0, 0.0, 0.0])
    camera_upward = jp.array([0.0, 1.0, 0.0])
    transform = SE3.view_transform(camera_origin, camera_target, camera_upward)
    ray_origins, ray_directions = build_rays(size, y_FOV, transform)
    return Render(ray_origins, ray_directions, *size, shadows)


def build_floor(floor):
    pattern = Pattern(jp.eye(4), PLANAR_PATTERN, floor.image)
    material = Material(floor.color, floor.ambient, floor.diffuse, 0.0, 200.0)
    return Shape(jp.eye(4), PLANE, pattern, material)


def build_shape(shape, label):
    material = Material(**shape._asdict())
    return Shape(label.transform, label.type, label.pattern, material)


def SceneLoss(render):
    def apply(scene, shape, shape_label, true_image):
        shape = build_shape(shape, shape_label)
        pred_image, pred_depth = render(shape, scene)
        return jp.mean((true_image - pred_image)**2)
    return apply


def Scene(size, y_FOV, camera_origin, shadows):
    render = build_render(camera_origin, size, y_FOV, shadows)

    def render_scene(shape, scene):
        floor = build_floor(scene.floor)
        shapes, mask = merge_shapes(shape, floor)
        image, depth = render(shapes, mask, scene.lights)
        return jp.clip(image, 0.0, 1.0), depth

    return render_scene


def label_to_shape(name_to_type, pattern_shape, label):
    label = list(label.values())[0]
    shift, theta, scale, color, shape_type = parse_label(label, name_to_type)
    shifts = SE3.translation(jp.array([shift[0], scale[1], shift[1]]))
    rotate = SE3.rotation_y(theta)
    scale = SE3.scaling(jp.array(scale))
    transform = shifts @ rotate @ scale
    material = Material(jp.array(color), 0.1, 0.9, 0.5, 4.0)
    pattern = Pattern(jp.eye(4), NO_PATTERN, jp.zeros(pattern_shape))
    arg_type = jp.argmax(shape_type).tolist()
    return Shape(transform, ARG_TO_TYPE[arg_type], pattern, material)


def parse_color(label):
    label = list(label.values())[0]
    return jp.array(label['RGB']) / 255.0


def label_to_material_type(label):
    label = list(label.values())[0]
    material_type = '_'.join([label['color'], label['material']])
    return material_type


def build_material_types(labels):
    material_types = {}
    for label in labels:
        material_type = label_to_material_type(label)
        material_types[material_type] = parse_color(label)
    return material_types


def build_MATERIAL(optimizer, color, ambient, diffuse, specular, shininess):
    variable = MATERIAL_VARIABLE(color, ambient, diffuse, specular, shininess)
    optimizer_state = optimizer.init(variable)
    return VARIABLE_STATE(optimizer_state, variable)


def initialize_materials(optimizer, labels, ambient,
                         diffuse, specular, shininess):
    material_types = build_material_types(labels)
    materials = {}
    for material_type, color in material_types.items():
        args = (optimizer, color, ambient, diffuse, specular, shininess)
        materials[material_type] = build_MATERIAL(*args)
    return materials


def initialize_landscape(optimizer, key, num_lights, pattern_shape):
    lights = []
    for arg in range(num_lights):
        key, subkey_0, subkey_1 = jax.random.split(key, 3)
        positions = jax.random.uniform(subkey_0, (3,), minval=-5.0, maxval=5.0)
        intensity = jax.random.uniform(subkey_1, (3,), minval=0.10, maxval=0.5)
        lights.append(PointLight(intensity, positions))
    floor = FLOOR(jp.array([0.5, 0.5, 0.5]), 0.1, 0.9, jp.zeros(pattern_shape))
    variable = LANDSCAPE_VARIABLE(floor, lights)
    optimizer_state = optimizer.init(variable)
    return VARIABLE_STATE(optimizer_state, variable)


def optimize_materials(scene, material, label, image, optimizer):
    state, variable = material.optimizer_state, material.variable
    loss, grads = materials_grad(scene.variable, variable, label, image)
    state, variable = update_state(optimizer, grads, state, variable)
    return loss, state, variable


def optimize_landscape(scene, material, label, image, optimizer):
    state, variable = scene.optimizer_state, scene.variable
    loss, grads = landscape_grad(variable, material.variable, label, image)
    state, variable = update_state(optimizer, grads, state, variable)
    return loss, state, variable


def update_state(optimizer, gradients, optimizer_state, variable):
    updates = optimizer.update(gradients, optimizer_state, variable)
    update, optimizer_state = updates
    variable = optax.apply_updates(variable, update)
    return optimizer_state, variable


def write_images(render, labels, materials, landscape, directory):
    for label_arg, label in enumerate(labels):
        material = materials[label_to_material_type(label)]
        label = label_to_shape(NAME_TO_TYPE, args.pattern_shape, label)
        shape = build_shape(material.variable, label)
        image, depth = render(shape, landscape.variable)
        write_image(image, directory, f'image_{label_arg:03d}.png')


key = jax.random.PRNGKey(args.seed)
root = build_directory(os.path.join(args.root, args.dataset_name), args.label)
write_dictionary(args.__dict__, root, 'parameters.json')

dataset_metadata = parse_metadata(args.dataset_path, args.dataset_name)
camera_origin = jp.array(dataset_metadata['camera_origin'])
H, W = dataset_metadata['image_shape']
y_FOV = dataset_metadata['y_FOV']
image_shape = [int(H * args.viewport_factor), int(W * args.viewport_factor)]
dataset_path = os.path.join(args.dataset_path, args.dataset_name)
dataset = load(dataset_path, args.split, image_shape)
images, depths, labels = flatten(dataset)

render = Scene(image_shape, y_FOV, camera_origin, args.shadow)
scene_loss = SceneLoss(render)
landscape_grad = jax.jit(jax.value_and_grad(scene_loss, argnums=(0)))
materials_grad = jax.jit(jax.value_and_grad(scene_loss, argnums=(1)))
landscape_optimizer = optax.adam(args.learning_rate)
materials_optimizer = optax.adam(args.learning_rate)

landscape_args = landscape_optimizer, key, args.num_lights, args.pattern_shape
landscape = initialize_landscape(*landscape_args)
materials = initialize_materials(materials_optimizer, labels, args.ambient,
                                 args.diffuse, args.specular, args.shininess)

print_progress = partial(print_progress, len(labels))
landscape_losses = []
materials_losses = {}
for material_type in materials.keys():
    materials_losses[material_type] = []

fast_render = jax.jit(render)
images_directory = os.path.join(root, args.images_directory)
epoch_directory = make_directory(os.path.join(images_directory, 'epoch_00'))
write_images(fast_render, labels, materials, landscape, epoch_directory)
for outer_epoch in range(1, args.outer_epochs + 1):
    print(f'Outer epoch {outer_epoch} / {args.outer_epochs}')
    for inner_epoch in range(args.inner_epochs):
        for label_arg, (label, image) in enumerate(zip(labels, images)):
            material_type = label_to_material_type(label)
            material = materials[material_type]
            shape = label_to_shape(NAME_TO_TYPE, args.pattern_shape, label)
            opt_args = (landscape, material, shape, image, materials_optimizer)
            loss, state, variable = optimize_materials(*opt_args)
            materials[material_type] = VARIABLE_STATE(state, variable)
            materials_losses[material_type].append(loss)
            print_progress(label_arg, f'material loss: {loss:.5f}')

    for inner_epoch in range(args.inner_epochs):
        for label_arg, (label, image) in enumerate(zip(labels, images)):
            material_type = label_to_material_type(label)
            material = materials[material_type]
            shape = label_to_shape(NAME_TO_TYPE, args.pattern_shape, label)
            opt_args = (landscape, material, shape, image, landscape_optimizer)
            loss, state, landscape = optimize_landscape(*opt_args)
            landscape = VARIABLE_STATE(state, landscape)
            landscape_losses.append(loss)
            print_progress(label_arg, f'landscape loss: {loss:.5f}')

    epoch_directory = f'epoch_{outer_epoch:02d}'
    epoch_directory = os.path.join(images_directory, epoch_directory)
    make_directory(epoch_directory)
    write_images(fast_render, labels, materials, landscape, epoch_directory)

write_pytree(landscape.variable.lights, root, 'lights.pkl')
write_pytree(build_floor(landscape.variable.floor), root, 'floor.pkl')

shapes_directory = make_directory(os.path.join(root, args.shapes_directory))
for label_arg, label in enumerate(labels):
    material = materials[label_to_material_type(label)]
    label = label_to_shape(NAME_TO_TYPE, args.pattern_shape, label)
    shape = build_shape(material.variable, label)
    write_pytree(shape, shapes_directory, f'shape_{label_arg:03d}.pkl')


def leaf_to_list(leaf):
    return leaf.tolist() if isinstance(leaf, jp.ndarray) else leaf


def write_materials(material_variables, path):
    materials = {}
    for material_type in material_variables.keys():
        variable = material_variables[material_type].variable
        variable = jax.tree_util.tree_map(leaf_to_list, variable)
        variable = variable._asdict()
        materials[material_type] = variable
    write_dictionary(materials, path, 'materials.json')


write_materials(materials, root)

losses_directory = make_directory(os.path.join(root, 'losses'))
write_pytree(landscape_losses, losses_directory, 'landscape_losses.pkl')
write_losses(landscape_losses, losses_directory, 'landscape_losses.pdf')
for material_type, losses in materials_losses.items():
    losses = jp.array(losses)
    write_pytree(losses, losses_directory, f'{material_type}_losses.pkl')
    write_losses(losses, losses_directory, f'{material_type}_losses.pdf')
