from typing import Callable
import jax
import jax.numpy as jp
from collections import namedtuple
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors


Distribution = tfd.Distribution
Variable = namedtuple(
    'Variable', ['apply', 'sample', 'sample_inverse', 'name'])
NodeState = namedtuple('NodeState', ['sample', 'log_prob'])
Observation = namedtuple('Observation', ['distribution', 'log_prob'])
PosteriorState = namedtuple(
    'PosteriorState', ['log_prob', 'prior_log_prob', 'likelihood_log_prob'])


def Node(name, distribution, bijector=None):
    # TODO this a node that returns a variable? Or a variable that gives a node

    if isinstance(distribution, Distribution):
        def model(*args):
            return distribution

    elif isinstance(distribution, Callable):
        def model(*args):
            return distribution(*args)

    else:
        raise ValueError('Invalid distribution type')

    if bijector is None:
        def forward_log_det_jac(x):
            return 0.0

        def forward_bijection(x):
            return x

        def inverse_bijection(x):
            return x
    else:
        def forward_log_det_jac(x):
            return bijector.forward_log_det_jacobian(x)

        def forward_bijection(x):
            return bijector(x)

        def inverse_bijection(x):
            return bijector.inverse(x)

    def apply(inverse_sample, *args):
        forward_sample = forward_bijection(inverse_sample)
        node_distribution = model(*args)
        log_det_jac = forward_log_det_jac(inverse_sample)
        log_prob = node_distribution.log_prob(forward_sample)
        log_prob = log_prob + log_det_jac
        return NodeState(forward_sample, log_prob.sum())

    def sample(key, num_samples, *args):
        distribution = model(*args)
        sample = distribution.sample(num_samples, seed=key)
        sample = jp.squeeze(sample, axis=0) if num_samples == 1 else sample
        return sample

    def sample_inverse(key, num_samples, *args):
        distribution = model(*args)
        sample = distribution.sample(num_samples, seed=key)
        sample = inverse_bijection(sample)
        sample = jp.squeeze(sample, axis=0) if num_samples == 1 else sample
        return sample

    return Variable(apply, sample, sample_inverse, name)


def Sequential(name, nodes):
    # TODO you need to return all chain samples. You might need them.

    def apply(inverse_samples):
        # TODO sequential only returns the last forward sample?
        previous_state = nodes[0].apply(inverse_samples[0])
        log_prob = previous_state.log_prob
        for node, inverse_sample in zip(nodes[1:], inverse_samples[1:]):
            previous_state = node.apply(inverse_sample, previous_state.sample)
            log_prob = log_prob + previous_state.log_prob.sum()
        return NodeState(previous_state.sample, log_prob)

    def sample(key, num_samples):
        # TODO you only need to sample from the root nodes?
        return nodes[0].sample(key, num_samples)

    def sample_inverse(key, num_samples):
        return nodes[0].sample_inverse(key, num_samples)

    return Variable(apply, sample, sample_inverse, name)


def Independent(name, nodes):
    node_names = [node.name for node in nodes]
    name_to_node = dict(zip(node_names, nodes))

    def apply(inverse_samples):
        forward_samples, log_prob = [], 0.0
        for node_name in node_names:
            inverse_sample = inverse_samples[node_name]
            state = name_to_node[node_name].apply(inverse_sample)
            log_prob = log_prob + state.log_prob
            forward_samples.append(state.sample)
        return NodeState(dict(zip(node_names, forward_samples)), log_prob)

    def sample(key, num_samples):
        samples = []
        for node in nodes:
            samples.append(node.sample(key, num_samples))
        return dict(zip(node_names, samples))

    def sample_inverse(key, num_samples):
        samples = []
        for node in nodes:
            samples.append(node.sample_inverse(key, num_samples))
        return dict(zip(node_names, samples))

    return Variable(apply, sample, sample_inverse, name)


def Likelihood(observation_model):

    def apply(forward_samples, observation):
        distribution, pred_image = observation_model(forward_samples)
        true_image = observation / 255.0
        pred_image = pred_image / 255.0
        log_prob = -25_000 * ((true_image - pred_image)**2).mean()
        # log_prob = distribution.log_prob(observation)
        return Observation(distribution, log_prob.sum())

    return apply


def Posterior(priors, likelihood, observation):

    def apply(inverse_samples):
        prior_state = priors.apply(inverse_samples)
        likelihood_log_prob = likelihood(prior_state.sample, observation)
        prior_log_prob = prior_state.log_prob
        log_prob = prior_log_prob + likelihood_log_prob
        return PosteriorState(log_prob, prior_log_prob, likelihood_log_prob)

    return apply


if __name__ == "__main__":
    from tensorflow_probability.substrates import jax as tfp
    tfd = tfp.distributions
    tfb = tfp.bijectors

    key = jax.random.PRNGKey(777)
    tfp_model = tfd.JointDistributionSequential([
        tfd.Beta(2.0, 5.0, name='beta'),
        lambda beta: tfd.Bernoulli(probs=beta, name='active')])

    node_1 = Node('beta', tfd.Beta(2.0, 5.0))
    node_2 = Node('bernoulli', lambda beta: tfd.Bernoulli(probs=beta))
    model = Sequential('is_active', [node_1, node_2])

    sample = tfp_model.sample(seed=key)
    x = model.apply(sample)
    print(jp.allclose(tfp_model.log_prob(sample), x.log_prob))

    forward_samples_node_1 = node_1.sample(key, 10_000)
    inverse_samples_node_1 = node_1.sample_inverse(key, 10_000)
    forward_samples_node_2 = node_2.sample_inverse(key, 10_000, 0.5)
    forward_samples_sequence_1 = model.sample(key, 10_000)
