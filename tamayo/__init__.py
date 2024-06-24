import pickle
import jax
import jax.numpy as jp

from .abstract import Material
from .abstract import PointLight
from .abstract import Shape
from .render import Render
from .camera import build_rays


def merge(*leafs):
    concatenated_leafs = []
    for leaf in leafs:
        concatenated_leafs.append(leaf)
    return jp.array(concatenated_leafs)


def merge_shapes(*shapes):
    return jax.tree_map(merge, *shapes), jp.ones(len(shapes), dtype=bool)


def write_pytree(x, filename):
    pickle.dump(x, open(f'{filename}.pkl', 'wb'))


def load_pytree(filename):
    # TODO remove the pkl glob takes in already
    return pickle.load(open(f'{filename}.pkl', 'rb'))
