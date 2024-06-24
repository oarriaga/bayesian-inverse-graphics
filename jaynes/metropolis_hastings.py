from functools import partial
from collections import namedtuple
import jax
import jax.numpy as jp
from jax.flatten_util import ravel_pytree


RandomWalkState = namedtuple('RandomWalkState', ['position', 'logdensity'])
RandomWalkInfo_fields = ['proposal', 'is_accepted', 'acceptance_rate']
RandomWalkInfo = namedtuple('RandomWalkInfo', RandomWalkInfo_fields)
proposal_fields = ['state', 'energy', 'weight', 'sum_log_p_accept']
Proposal = namedtuple('Proposal', proposal_fields)


@partial(jax.jit, static_argnames=('precision',), inline=True)
def linear_map(diag_or_dense_a, b, *, precision='highest'):
    """Perform a linear map of the form y = Ax.
    """
    dtype = jp.result_type(diag_or_dense_a.dtype, b.dtype)
    diag_or_dense_a = diag_or_dense_a.astype(dtype)
    b = b.astype(dtype)
    ndim = jp.ndim(diag_or_dense_a)

    if ndim <= 1:
        return jax.lax.mul(diag_or_dense_a, b)
    else:
        return jax.lax.dot(diag_or_dense_a, b, precision=precision)


def generate_gaussian_noise(key, position, mu=0.0, sigma=1.0):
    """Generate N(mu, sigma) noise with structure that matches a given PyTree.
    """
    flat_pytree, unravel = ravel_pytree(position)
    flat_shape = flat_pytree.shape
    flat_dtype = flat_pytree.dtype
    sample = jax.random.normal(key, shape=flat_shape, dtype=flat_dtype)
    return unravel(mu + linear_map(sigma, sample))


def binomial_sampling(key, now_proposal, new_proposal):
    p_accept = jp.clip(jp.exp(new_proposal.weight), a_max=1)
    do_accept = jax.random.bernoulli(key, p_accept)
    return jax.lax.cond(
        do_accept,
        lambda _: (new_proposal, do_accept, p_accept),
        lambda _: (now_proposal, do_accept, p_accept),
        operand=None)


def normal(sigma):
    if jp.ndim(sigma) > 2:
        raise ValueError("sigma must be a vector or a matrix.")

    def propose(key, position):
        return generate_gaussian_noise(key, position, sigma=sigma)

    return propose


def symmetric_proposal_log_density_fn(old_state, new_state):
    return -new_state.logdensity


def BuildProposals(proposal_log_density_fn):
    def build_now_proposal(state):
        return Proposal(state, 0.0, 0.0, -jp.inf)

    def build_new_proposal(old_state, new_state):
        new_energy = proposal_log_density_fn(old_state, new_state)
        old_energy = proposal_log_density_fn(new_state, old_state)
        delta_energy = old_energy - new_energy
        weight = delta_energy
        sum_log_p_accept = jp.minimum(delta_energy, 0.0)
        return Proposal(new_state, new_energy, weight, sum_log_p_accept)

    return build_now_proposal, build_new_proposal


def RMHKernel(log_density_fn, propose, proposal_log_density_fn):
    build_now_proposal, build_new_proposal = BuildProposals(
        proposal_log_density_fn)

    def build_trajectory(key, initial_state):
        position, logdensity = initial_state
        new_position = propose(key, position)
        return RandomWalkState(new_position, log_density_fn(new_position))

    def kernel_step(key, state):
        keys = jax.random.split(key)
        end_state = build_trajectory(keys[0], state)
        now_proposal = build_now_proposal(state)
        new_proposal = build_new_proposal(state, end_state)
        sample = binomial_sampling(keys[1], now_proposal, new_proposal)
        proposal, do_accept, p_accept = sample
        new_state = proposal.state
        return new_state, RandomWalkInfo(proposal, do_accept, p_accept)

    return kernel_step


def RMHInitialize(position, log_density_fn):
    return RandomWalkState(position, log_density_fn(position))


def AdditiveProposal(sigma):
    sample_move = normal(sigma)

    def propose(key, position):
        move_proposal = sample_move(key, position)
        new_position = jax.tree_util.tree_map(jp.add, position, move_proposal)
        return new_position
    return propose


def RMH(log_density_fn, propose, proposal_log_density_fn=None):
    if proposal_log_density_fn is None:
        proposal_log_density_fn = symmetric_proposal_log_density_fn
    kernel_step = RMHKernel(log_density_fn, propose, proposal_log_density_fn)
    initializer = partial(RMHInitialize, log_density_fn=log_density_fn)
    return kernel_step, initializer


@partial(jax.jit, static_argnums=(1, 3, 4))
def chain(key, kernel, initial_states, num_samples, num_chains):

    def one_step(step_state, sample_arg):
        now_key, states = step_state
        new_key, key = jax.random.split(now_key)
        keys = jax.random.split(key, num_chains)
        states, infos = jax.vmap(kernel)(keys, states)
        return (new_key, states), (states, infos)

    args = jp.arange(num_samples)
    _, (states, infos) = jax.lax.scan(one_step, (key, initial_states), args)

    return (states, infos)


def to_trace(variables, inverse_chain_samples, burn_in=0):
    trace = {}
    for variable in variables:
        inverse_samples = inverse_chain_samples[variable.name]
        forward_samples = variable.apply(inverse_samples).sample
        forward_samples = jp.swapaxes(forward_samples, 0, 1)[:, burn_in:]
        trace[variable.name] = forward_samples
    return trace
