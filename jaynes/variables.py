import jax.numpy as jp
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

from .primitives import Node


def to_log_normal(mean, variance):
    scale = jp.log((variance / mean**2) + 1)
    mu = jp.log(mean) - (scale / 2)
    return mu, jp.sqrt(scale)


def build_GMM(weights, mean, stdv):
    return tfd.MixtureSameFamily(
        tfd.Categorical(probs=weights), tfd.Normal(loc=mean, scale=stdv))


def GaussianMixture(weights, mean, stdv, bijector, name):
    distribution = build_GMM(weights, mean, stdv)
    return Node(name, distribution, bijector)


def Shift(mean, scale, name='shift'):
    lower = jp.array([-0.14, -0.14,])
    upper = jp.array([0.14, 0.14,])
    distribution = tfd.TruncatedNormal(mean, scale, lower, upper)
    bijector = tfb.Chain([tfb.Shift(mean), tfb.Scale(scale)])
    return Node(name, distribution, bijector)


def Theta(mean, concentration, name='theta'):
    distribution = tfd.VonMises(mean, concentration)
    bijector = tfb.Chain([tfb.Sigmoid(-jp.pi, jp.pi), tfb.Scale(jp.pi / 2.0)])
    return Node(name, distribution, bijector)


def Scale(mean, variance, name='scale'):
    mean, variance = jp.array(mean), jp.array(variance)
    variance = jp.log((variance / mean**2) + 1)
    mean = jp.log(mean) - (variance / 2)
    scale = jp.sqrt(variance)

    distribution = tfd.LogNormal(mean, scale)
    bijector = tfb.Chain([tfb.Softplus(), tfb.Shift(mean), tfb.Scale(scale)])
    return Node(name, distribution, bijector)


def Classes(probabilities, temperature, name='classes'):
    distribution = tfd.RelaxedOneHotCategorical(
        temperature, probs=probabilities)
    bijector = tfb.Chain([tfb.SoftmaxCentered(), tfb.Scale(3.0)])
    return Node(name, distribution, bijector)


def Affine(x):
    return tfb.Chain([tfb.Shift(x.shift), tfb.Scale(x.scale)])


def StudentTNoise(degree, mean, scale, name='noise'):
    distribution = tfd.StudentT(degree, mean, scale)
    bijector = tfb.Chain([tfb.Shift(mean), tfb.Scale(scale)])
    return Node(name, distribution, bijector)


def GaussianNoise(mean, scale, name='noise'):
    distribution = tfd.Normal(mean, scale)
    bijector = tfb.Chain([tfb.Shift(mean), tfb.Scale(scale)])
    return Node(name, distribution, bijector)


def LaplaceNoise(mean, scale, name='noise'):
    distribution = tfd.Laplace(mean, scale)
    return Node(name, distribution)
