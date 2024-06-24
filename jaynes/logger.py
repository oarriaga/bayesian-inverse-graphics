# TODO change this to bijection optimization?
import os
import json
import glob
from collections import namedtuple

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

from .variables import Affine, GaussianMixture, StudentTNoise

# is this necessary?
BIJECTOR = namedtuple('BIJECTOR', ['shift', 'scale'])


def find_path(wildcard):
    filenames = glob.glob(wildcard)
    filepaths = []
    for filename in filenames:
        if os.path.isdir(filename):
            filepaths.append(filename)
    return max(filepaths, key=os.path.getmtime)


def load_noise_model(wildcard, filename):
    directory = find_path(wildcard)
    filedata = open(os.path.join(directory, filename), 'r')
    noise_parameters = json.load(filedata)
    return StudentTNoise(**noise_parameters)


def load_priors(wildcard, filename):
    directory = find_path(wildcard)
    filedata = open(os.path.join(directory, filename), 'r')
    priors_params = json.load(filedata)
    priors = []
    for name, prior_params in priors_params.items():
        mixture_params = prior_params['mixture']
        bijector_params = prior_params['bijector']
        bijector = tfb.Invert(Affine(BIJECTOR(**bijector_params)))
        prior = GaussianMixture(**mixture_params, bijector=bijector, name=name)
        priors.append(prior)
    return priors
