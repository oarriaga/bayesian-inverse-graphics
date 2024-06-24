import jax.numpy as jp
from equinox import tree_deserialise_leaves


def extract_last_layer(branches):
    return jp.max(jp.array([branch['layer'] for branch in branches])).tolist()


def extract_invariant_features(features, branches):
    invariant_maps = []
    for branch in branches:
        layer_arg = branch['layer'] - 1
        invariant_maps.append(features[layer_arg][branch['featuremap']])
    return invariant_maps


def extract_features(x, model, last_layer_arg):
    features = []
    for layer in model.layers[:last_layer_arg]:
        x = layer(x)
        features.append(x)
    return features


def extract_featuremaps(features, branches):
    layers = ([branch['layer'] for branch in branches])
    featuremap_args = ([branch['featuremap'] for branch in branches])
    invariant_featuremaps = []
    for layer, featuremap in zip(layers, featuremap_args):
        invariant_featuremaps.append(features[layer][featuremap])
    return invariant_featuremaps


def build_branches(invariances, num_branches):
    branches = [invariances[str(arg)] for arg in range(num_branches)]
    return branches


def BRANCH_CNN(base_model, weights_path, invariances, num_branches):
    branches = build_branches(invariances, num_branches)
    last_layer_arg = extract_last_layer(branches)
    model = tree_deserialise_leaves(weights_path, base_model)

    def apply(x):
        features = extract_features(x, model, last_layer_arg)
        invariant_features = extract_invariant_features(features, branches)
        return invariant_features
    return apply
