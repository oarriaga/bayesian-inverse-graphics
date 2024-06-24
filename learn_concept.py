import os
import time
import argparse
import pickle
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
description = 'Probabilistic inverse graphics for few short learning'
parser = argparse.ArgumentParser(description=description)
parser.add_argument('--seed', default=777, type=int)
parser.add_argument('--root', default='experiments', type=str)
parser.add_argument('--label', default='IMAGE')
parser.add_argument('--split', default='test', type=str)
parser.add_argument('--device', default='gpu', type=str)
parser.add_argument('--dataset_name', default='PRIMITIVES')
parser.add_argument('--prior_filename', default='prior.json')
parser.add_argument('--priors_wildcard', default='*PRIORS')
parser.add_argument('--params_filename', default='parameters.json')
parser.add_argument('--noise_wildcard', default='*LIKELIHOOD-CALIBRATION')
parser.add_argument('--noise_filename', default='noise_parameters.json')
parser.add_argument('--scene_wildcard', default='*SCENE-OPTIMIZATION')
parser.add_argument('--features_wildcard', default='*INVARIANT-MAPS')
parser.add_argument('--features_filename', default='invariances.json')
parser.add_argument('--shadows', default=False, type=bool)
parser.add_argument('--neural_model', default='VGG16', type=bool)
parser.add_argument('--weights_path', default='VGG16.eqx')
parser.add_argument('--concept', default=0, type=int)
parser.add_argument('--shot_arg', default=0, type=int)
parser.add_argument('--tune_steps', default=500, type=float)
parser.add_argument('--tune_episodes', default=5, type=float)
parser.add_argument('--num_posterior_samples', default=100, type=int)
parser.add_argument('--image_noise', default=1.0, type=float)
parser.add_argument('--neuro_weight', default=0.05, type=float)
parser.add_argument('--num_branches', default=3, type=int)
parser.add_argument('--num_chains', default=20, type=int)
parser.add_argument('--burn_in', default=1000, type=int)
parser.add_argument('--num_samples', default=30_000, type=int)
parser.add_argument('--sigma', default=0.05, type=float)
args = parser.parse_args()

import jax
jax.config.update('jax_platform_name', args.device)
import jax.numpy as jp
from tamayo import SE3, Render, build_rays
from lecun import VGG16, BRANCH_CNN

from jaynes.primitives import Independent, Posterior
from jaynes.observation_model import NeuroLikelihood
from jaynes.observation_model import ObservationModel, Likelihood
from jaynes.variables import Theta, Classes, Shift, Scale
from jaynes.logger import load_priors
from jaynes.tuner import Tuner, compute_acceptance_rate
from jaynes.observation_model import estimate_point
from jaynes.metropolis_hastings import AdditiveProposal, RMH, to_trace, chain
from jaynes.programs import sample_posterior
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

from primitives import load, parse_metadata
from logger import (write_trace, build_directory, write_dictionary,
                    load_parameters, find_path, write_summary, make_directory)

from plotter import (build_summary, write_image, plot_trace,
                     write_error_image,
                     write_true_pred_image, plot_shift_posterior,
                     plot_theta_posterior, plot_shift_posteriors,
                     plot_shift_x_posterior, plot_shift_y_posterior,
                     plot_scale_posterior, plot_scale_x_posterior,
                     plot_scale_y_posterior, plot_scale_z_posterior,
                     plot_color_posterior, plot_ambient_posterior,
                     plot_diffuse_posterior, plot_specular_posterior,
                     plot_shininess_posterior, plot_classes_posterior,
                     plot_dirichlet_posterior)


print('LOADING PRIORS')
root = os.path.join(args.root, args.dataset_name)
priors_wildcard = os.path.join(root, args.priors_wildcard)
priors_metadata = load_parameters(priors_wildcard, args.params_filename)
shift_mean = priors_metadata['shift_mean']
shift_stdv = priors_metadata['shift_scale']  # TODO change to stdv
theta_mean = priors_metadata['theta_mean']
theta_concentration = priors_metadata['theta_concentration']
theta = Theta(theta_mean, theta_concentration, 'theta')
probabilities = priors_metadata['class_probabilities']
temperature = priors_metadata['classes_temperature']
shape = Classes(probabilities, temperature)
shift = Shift(shift_mean, shift_stdv, 'shift')
scale = Scale(priors_metadata['scale_mean'], priors_metadata['scale_scale'])
material_priors = load_priors(priors_wildcard, args.prior_filename)
ambient, diffuse, specular, shine, color = material_priors
nodes = [shift, theta, scale, color, ambient, diffuse, specular, shine, shape]

print('LOADING SCENE PARAMETERS')
scene_wildcard = os.path.join(root, args.scene_wildcard)
scene_metadata = load_parameters(scene_wildcard, args.params_filename)
dataset_path = scene_metadata['dataset_path']
dataset_name = scene_metadata['dataset_name']
viewport_factor = scene_metadata['viewport_factor']

print('LOADING DATASET PARAMETERS')
dataset_metadata = parse_metadata(dataset_path, dataset_name)
camera_origin = jp.array(dataset_metadata['camera_origin'])
H, W = dataset_metadata['image_shape']
image_shape = [int(H * viewport_factor), int(W * viewport_factor)]
print('IMAGE_SHAPE', image_shape)
y_FOV = dataset_metadata['y_FOV']

print('SETTING DIFFERENTIABLE RENDERER')
camera_origin = jp.array(camera_origin)
camera_target = jp.array([0.0, 0.0, 0.0])
camera_upward = jp.array([0.0, 1.0, 0.0])
camera_pose = SE3.view_transform(camera_origin, camera_target, camera_upward)
ray_origins, ray_directions = build_rays(image_shape, y_FOV, camera_pose)
tamayo_render = Render(ray_origins, ray_directions, *image_shape, args.shadows)

print('LOADING OPTIMIZED SCENE')
optimization_directory = find_path(scene_wildcard)
floor_filename = os.path.join(optimization_directory, 'floor.pkl')
floor = pickle.load(open(floor_filename, 'rb'))
lights_filename = os.path.join(optimization_directory, 'lights.pkl')
lights = pickle.load(open(lights_filename, 'rb'))

print('SETTING OBSERVATION MODEL')
observation_model = ObservationModel(tamayo_render, floor, lights)

print('LOADING NEURAL LIKELIHOOD')
key = jax.random.PRNGKey(args.seed)
key_0, key_1, key_2, key_3, key_4, key_5 = jax.random.split(key, 6)
base_model = VGG16(key_0)
features_wildcard = os.path.join(root, args.features_wildcard)
invariances = load_parameters(features_wildcard, args.features_filename)
CNN = BRANCH_CNN(base_model, args.weights_path, invariances, args.num_branches)

print('LOADING DATASET')
observations = []
dataset_fullpath = os.path.join(dataset_path, args.dataset_name)
dataset = load(dataset_fullpath, args.split, image_shape)
image, depth, label = dataset[args.concept][args.shot_arg]

print('LOADING NOISE DISTRIBUTION')
noise_model = tfd.TruncatedNormal(0.0, args.image_noise, -1.0, 1.0)
neuro_model = NeuroLikelihood(args.neuro_weight, CNN)

print('SETTING PRIOR, LIKELIHOOD AND TARGET DISTRIBUTION')
likelihood = Likelihood(observation_model, noise_model, neuro_model)
priors = Independent('priors', nodes)
target = Posterior(priors, likelihood, image)

print('SAMPLING INVERSE SAMPLES')
samples = priors.sample_inverse(key_1, args.num_chains)

print('TUNING MCMC KERNEL')
tune = Tuner(lambda x: target(x).log_prob, samples, args.num_chains)
tuner_start_time = time.time()
tuner_state = tune(key_3, args.tune_steps, args.tune_episodes, args.sigma)
tuner_end_time = time.time()
tuner_duration = tuner_end_time - tuner_start_time
sigma = tuner_state.sigma[-1].mean()
mean_acceptance_rate = tuner_state.acceptance_rate[-1].mean()
print('MEAN ACCEPTANCE RATE', f'{mean_acceptance_rate:.03f}')
print('TUNING TIME', f'{tuner_duration:.02f} [s]')

print('INITIALIZING MCMC KERNEL WITH SIGMA:', sigma)
propose = AdditiveProposal(sigma)
kernel, initialize = RMH(lambda x: target(x).log_prob, propose)

print('STARTING MULTI-CHAIN INFERENCE')
states = jax.vmap(initialize, in_axes=(0))(samples)
chain_start_time = time.time()
states, infos = chain(key_4, kernel, states, args.num_samples, args.num_chains)
chain_end_time = time.time()
acceptance_rate = compute_acceptance_rate(infos).tolist()
chain_duration = chain_end_time - chain_start_time
total_duration = tuner_duration + chain_duration
print('SAMPLING TIME', f'{chain_duration:.02f} [s]')
print('TOTAL TIME', f'{total_duration:.02f} [s]')
print('ACCEPTANCE RATES', acceptance_rate)

print('LOG RESULTS')
trace = to_trace(nodes, states.position, args.burn_in)
summary = build_summary(trace)
print(summary)
results = {'sigma': float(sigma),
           'tuner_duration': tuner_duration,
           'chain_duration': chain_duration,
           'total_duration': total_duration,
           'acceptance_rate': acceptance_rate}
render = jax.jit(observation_model)
root_label = f'{args.label}_PROBABILISTIC-PROGRAMS'
root = os.path.join(args.root, args.dataset_name, root_label)
root = os.path.join(root, f'CONCEPT-{args.concept:02d}')
root = os.path.join(root, f'SHOT-{args.shot_arg:02d}')
root = build_directory(root)
write_dictionary(args.__dict__, root, 'parameters.json')
write_dictionary(results, root, 'results.json')
write_summary(summary, root)
write_trace(trace, root)
image = jp.clip(image, 0.0, 1.0)
write_image(image, root, 'true_image.png')
mean_image, _ = estimate_point(summary, render, 'mean')
medn_image, _ = estimate_point(summary, render, 'median')
write_image(mean_image, root, 'mean_point.png')
write_image(medn_image, root, 'median_point.png')
write_true_pred_image(image, mean_image, root, 'mean')
write_true_pred_image(image, medn_image, root, 'median')
write_error_image(image, mean_image, root)
write_error_image(image, medn_image, root)
write_image(mean_image, root, 'mean_image.png')
write_image(medn_image, root, 'median_image.png')

plot_trace(trace, root)
plot_shift_posterior(trace, label, root)
plot_theta_posterior(trace, root)
plot_shift_posteriors(trace, root)
plot_shift_x_posterior(trace, root)
plot_shift_y_posterior(trace, root)
plot_scale_posterior(trace, root)
plot_scale_x_posterior(trace, root)
plot_scale_y_posterior(trace, root)
plot_scale_z_posterior(trace, root)
plot_color_posterior(trace, root)
plot_ambient_posterior(trace, root)
plot_diffuse_posterior(trace, root)
plot_specular_posterior(trace, root)
plot_shininess_posterior(trace, root)
plot_classes_posterior(trace, root)
plot_dirichlet_posterior(trace, root)

print('LOG POSTERIOR SAMPLES')
posterior_directory = make_directory(os.path.join(root, 'posterior_samples'))
images = sample_posterior(key_5, args.num_posterior_samples, trace, render)
for arg, image in enumerate(images):
    write_image(image, posterior_directory, f'posterior_sample_{arg:02d}.png')
