from collections import namedtuple
import jax
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

SAMPLE_FIELDS = ['shift', 'theta', 'scale', 'color', 'ambient',
                 'diffuse', 'specular', 'shininess', 'classes']
SAMPLE = namedtuple('SAMPLE', SAMPLE_FIELDS)
# probabilistic prototypical program
P3_FIELDS = ['scale', 'color', 'ambient', 'diffuse',
             'specular', 'shininess', 'classes']
P3 = namedtuple('P3', P3_FIELDS)


def samples_to_distribution(chain_samples, name):
    shape = chain_samples.shape
    if len(shape) == 3:
        num_chains, num_samples, dimension = shape
        new_shape = (num_chains * num_samples, dimension)
        event_dim = 1
    if len(shape) == 2:
        num_chains, num_samples = shape
        new_shape = (num_chains * num_samples)
        event_dim = 0
    samples = chain_samples.reshape(new_shape)
    distribution = tfd.Empirical(samples, event_dim, name=name)
    return distribution


def build_program(posterior):
    program_properties = {}
    for distribution_name, distribution in posterior.items():
        if distribution_name in P3._fields:
            program_properties[distribution_name] = distribution
    return P3(**program_properties)


def sample_from_program(key, program):
    posterior = {}
    for name, distribution in program._asdict().items():
        key, subkey = jax.random.split(key)
        posterior[name] = distribution.sample(seed=subkey)
    return posterior


def render_program(key, shift, theta, program, render):
    program_samples = sample_from_program(key, program)
    program_samples['shift'] = shift
    program_samples['theta'] = theta
    image, depth = render(program_samples)
    return image, depth


def trace_to_posterior(trace):
    posterior = {}
    for name, chain_samples in trace.items():
        samples = samples_to_distribution(chain_samples, name)
        posterior[name] = samples
    return posterior


def sample_posterior(key, num_samples, trace, render):
    posterior = trace_to_posterior(trace)
    program = build_program(posterior)
    shift_distribution = posterior['shift']
    theta_distribution = posterior['theta']
    images = []
    for arg, key in enumerate(jax.random.split(key, num_samples)):
        keys = jax.random.split(key, 3)
        shift = shift_distribution.sample(seed=keys[0])
        theta = theta_distribution.sample(seed=keys[1])
        image, depth = render_program(keys[2], shift, theta, program, render)
        images.append(image)
    return images
