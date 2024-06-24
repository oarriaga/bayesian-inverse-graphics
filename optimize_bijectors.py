import os
import pickle
import argparse
from functools import partial

import jax
import jax.numpy as jp
import optax
import numpy as np
from sklearn.mixture import GaussianMixture as SKLGaussianMixture
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
from tamayo import SE3, Render, build_rays
from primitives import parse_metadata

from jaynes.observation_model import ObservationModel
from jaynes.logger import BIJECTOR, load_priors
from jaynes.variables import Shift, Theta, Scale, Classes, Affine, build_GMM
from jaynes.primitives import Independent

from plotter import write_image, plot_priors, write_losses
from logger import (make_directory, print_progress, build_directory,
                    write_dictionary, load_parameters, find_path)

parser = argparse.ArgumentParser(description='Optimize Bijectors')
parser.add_argument('--root', default='experiments', type=str)
parser.add_argument('--seed', default=777, type=int)
parser.add_argument('--label', default='PRIORS', type=str)
parser.add_argument('--dataset_name', default='PRIMITIVES', type=str)
parser.add_argument('--parameters_wildcard', default='*SCENE-OPTIMIZATION')
parser.add_argument('--parameters_filename', default='parameters.json')
parser.add_argument('--scene_filename', default='scene.json')
parser.add_argument('--materials_filename', default='materials.json')
parser.add_argument('--floor_filename', default='floor.npy')
parser.add_argument('--prior_filename', default='prior.json')
parser.add_argument('--num_samples', default=100_000, type=int)
parser.add_argument('--gradient_steps', default=25_000, type=int)
parser.add_argument('--learning_rate', default=1e-3, type=int)
parser.add_argument('--shift_mean', nargs='+', default=[0.0, 0.025])
parser.add_argument('--shift_scale', nargs='+', default=[0.08, 0.08])
parser.add_argument('--theta_mean', default=0.0, type=float)
parser.add_argument('--theta_concentration', default=0.0, type=float)
parser.add_argument('--scale_mean', nargs='+', default=[0.025, 0.025, 0.025])
parser.add_argument('--scale_scale', nargs='+', default=[1e-4, 1e-4, 1e-4])
parser.add_argument('--shadows', default=True, type=bool)
parser.add_argument('--num_images', default=100, type=int)
parser.add_argument('--classes_temperature', default=0.5, type=float)
parser.add_argument('--class_probabilities',
                    nargs='+', default=[1 / 3, 1 / 3, 1 / 3])
args = parser.parse_args()


def BijectionLoss(samples, distribution, Bijector):
    def compute(x):
        bijector = Bijector(x)
        pred_distribution = tfd.TransformedDistribution(distribution, bijector)
        negative_log_likelihood = - pred_distribution.log_prob(samples)
        return negative_log_likelihood.sum()
    return compute


def get_property_values(shape_parameters, name):
    values = []
    for type_name, type_values in shape_parameters.items():
        for property_name, property_value in type_values.items():
            if property_name == name:
                values.append(property_value)
    return np.array(values)


def optimize_bijector(x, samples, distribution, bijector, optimizer, steps):
    optimizer_state = optimizer.init(x)
    compute_loss = BijectionLoss(samples, distribution, bijector)
    compute_grad = jax.jit(jax.value_and_grad(compute_loss, argnums=(0)))
    losses = []
    for step in range(steps):
        loss, grads = compute_grad(x)
        update, optimizer_state = optimizer.update(grads, optimizer_state)
        x = optax.apply_updates(x, update)
        print_progress(steps, step, f' | {name} loss: {loss}')
        losses.append(loss)
    return losses, x


def get_mixture_parameters(model, unidimensional=True):
    weights = model.weights_.tolist()
    if unidimensional:
        mean = model.means_[:, 0].tolist()
        stdv = np.sqrt(model.covariances_[:, 0]).tolist()
    else:
        mean = np.moveaxis(np.array(model.means_), 1, 0).tolist()
        stdv = np.moveaxis(np.sqrt(model.covariances_), 1, 0).tolist()
    return {'weights': weights, 'mean': mean, 'stdv': stdv}


def fit_gaussian_mixture(seed, samples, unidimensional=True):
    kwargs = {'covariance_type': 'diag', 'max_iter': 1000, 'reg_covar': 1e-3,
              'n_init': 500, 'tol': 1e-4, 'random_state': seed}
    model = SKLGaussianMixture(2, **kwargs)
    if unidimensional:
        samples = samples.reshape(-1, 1)
    model = model.fit(samples)
    return get_mixture_parameters(model, unidimensional)


root = build_directory(os.path.join(args.root, args.dataset_name), args.label)
write_dictionary(args.__dict__, root, 'parameters.json')
priors = {}
wildcard = os.path.join(args.root, args.dataset_name, args.parameters_wildcard)
shape_parameters = load_parameters(wildcard, args.materials_filename)
key = jax.random.PRNGKey(args.seed)
optimize_bijector = partial(optimize_bijector, steps=args.gradient_steps)
for name in ['ambient', 'diffuse', 'specular', 'shininess', 'color']:
    one_dim = True if name != 'color' else False
    values = get_property_values(shape_parameters, name)
    mixture_params = fit_gaussian_mixture(args.seed, values, one_dim)
    key, key_n, key_g = jax.random.split(key, 3)
    shift_0 = 0.0 if name != 'color' else jp.full(3, 0.0)
    scale_0 = 1.0 if name != 'color' else jp.full(3, 1.0)
    samples = tfd.Normal(shift_0, scale_0).sample(args.num_samples, seed=key_n)
    x_0 = BIJECTOR(shift_0, scale_0)
    distribution = build_GMM(**mixture_params)
    optimizer = optax.adam(args.learning_rate)
    loss, x = optimize_bijector(x_0, samples, distribution, Affine, optimizer)
    write_losses(loss, root, f'loss_{name}.pdf')
    bijector_params = {'shift': x.shift.tolist(), 'scale': x.scale.tolist()}
    priors[name] = {'mixture': mixture_params, 'bijector': bijector_params}
write_dictionary(priors, root, args.prior_filename)


wildcard = os.path.join(args.root, args.dataset_name, args.parameters_wildcard)
optimization_metadata = load_parameters(wildcard, args.parameters_filename)
dataset_path = optimization_metadata['dataset_path']
viewport_factor = optimization_metadata['viewport_factor']
dataset_metadata = parse_metadata(dataset_path, args.dataset_name)
H, W = dataset_metadata['image_shape']
image_shape = [int(H * viewport_factor), int(W * viewport_factor)]
camera_origin = jp.array(dataset_metadata['camera_origin'])
y_FOV = dataset_metadata['y_FOV']

optimization_directory = find_path(wildcard)
floor_filename = os.path.join(optimization_directory, 'floor.pkl')
floor = pickle.load(open(floor_filename, 'rb'))

lights_filename = os.path.join(optimization_directory, 'lights.pkl')
lights = pickle.load(open(lights_filename, 'rb'))

camera_origin = jp.array(camera_origin)
camera_target = jp.array([0.0, 0.0, 0.0])
camera_upward = jp.array([0.0, 1.0, 0.0])
camera_pose = SE3.view_transform(camera_origin, camera_target, camera_upward)
ray_origins, ray_directions = build_rays(image_shape, y_FOV, camera_pose)
_render = Render(ray_origins, ray_directions, *image_shape, args.shadows)
render = ObservationModel(_render, floor, lights)
render = jax.jit(render)

priors_wildcard = os.path.join(args.root, args.dataset_name, '*' + args.label)
priors = load_priors(priors_wildcard, args.prior_filename)
priors.extend([Shift(args.shift_mean, args.shift_scale),
               Theta(args.theta_mean, args.theta_concentration),
               Scale(args.scale_mean, args.scale_scale),
               Classes(args.class_probabilities, args.classes_temperature)])

plot_priors(key, priors, root, args.num_samples)
image_directory = os.path.join(root, 'prior_predictive_samples')
make_directory(image_directory)
priors = Independent('priors', priors)
for image_arg, subkey in enumerate(jax.random.split(key, args.num_images)):
    sample = priors.sample(subkey, 1)
    image, depth = render(sample)
    filename = f'{image_arg:02d}_prior_predictive_sample.png'
    write_image(image, image_directory, filename)
