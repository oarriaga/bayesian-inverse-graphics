import os
import glob
import json
import jax
import jax.numpy as jp
import numpy as np
from matplotlib.pyplot import imread


def load_image(filepath, shape):
    image = imread(filepath)[:, :, :3]
    image = jax.image.resize(image, (*shape, 3), 'bilinear')
    return np.array(image)


def load_depth(filepath, shape):
    depth = jp.expand_dims(jp.load(filepath), axis=2)
    depth = jax.image.resize(depth, (*shape, 1), 'bilinear')
    return np.array(depth)


def load_label(filepath):
    try:
        return json.load(open(filepath, 'r'))
    except json.decoder.JSONDecodeError as err:
        raise ValueError(f'loading {filepath} raised {err.__class__.__name__}')


def load_shot(directory, shape):
    label = load_label(os.path.join(directory, 'label.json'))
    image = load_image(os.path.join(directory, 'image.png'), shape)
    depth = load_depth(os.path.join(directory, 'depth.npy'), shape)
    return image, depth, label


def load_paths(root_path):
    wildacard = os.path.join(root_path, '*')
    directories = glob.glob(wildacard)
    directories = sorted(directories)
    return directories


def load(root_path, split, shape=(224, 224)):
    data_path = os.path.join(root_path, split)
    concepts = []
    for concept_path in load_paths(data_path):
        shots = [load_shot(path, shape) for path in load_paths(concept_path)]
        concepts.append(shots)
    return concepts


def flatten(dataset):
    images, depths, labels = [], [], []
    for concept in dataset:
        for shot in concept:
            image, depth, label = shot
            images.append(image)
            depths.append(depth)
            labels.append(label)
    return np.array(images), depths, labels


def sample(RNG, dataset, num_ways, num_shots, num_tests=1):
    random_concepts = RNG.choice(range(len(dataset)), num_ways, False)
    test_images, test_labels = [], []
    shot_images, shot_labels = [], []
    num_samples = num_shots + num_tests
    for label, concept_arg in enumerate(random_concepts):
        concept = dataset[concept_arg]
        random_shots = RNG.choice(range(len(concept)), num_samples, False)
        images = [concept[shot_arg][0] for shot_arg in random_shots]
        labels = [concept[shot_arg][2] for shot_arg in random_shots]
        labels = np.full(num_samples, label)
        shot_images.append(images[:num_shots])
        shot_labels.append(labels[:num_shots])
        test_images.append(images[num_shots:])
        test_labels.append(labels[num_shots:])
    shot_images = np.concatenate(shot_images, axis=0)
    shot_labels = np.concatenate(shot_labels, axis=0)
    test_images = np.concatenate(test_images, axis=0)
    test_labels = np.concatenate(test_labels, axis=0)
    return (shot_images, shot_labels), (test_images, test_labels)


def split_data(data, validation_split):
    num_train_samples = int(len(data) * (1 - validation_split))
    train_data = data[:num_train_samples]
    validation_data = data[num_train_samples:]
    return train_data, validation_data


def split(data, validation_split):
    num_train_samples = int(len(data) * (1 - validation_split))
    train_data = data[:num_train_samples]
    validation_data = data[num_train_samples:]
    return train_data, validation_data


def make_mosaic(images, shape, border=0):
    num_images, H, W, num_channels = images.shape
    num_rows, num_cols = shape
    if num_images > (num_rows * num_cols):
        raise ValueError('Number of images is bigger than shape')

    total_rows = (num_rows * H) + ((num_rows - 1) * border)
    total_cols = (num_cols * W) + ((num_cols - 1) * border)
    mosaic = np.ones((total_rows, total_cols, num_channels))

    padded_H = H + border
    padded_W = W + border

    for image_arg, image in enumerate(images):
        row = int(np.floor(image_arg / num_cols))
        col = image_arg % num_cols
        mosaic[row * padded_H:row * padded_H + H,
               col * padded_W:col * padded_W + W, :] = image
    return mosaic


def parse_shift(label):
    # y -> x,  -x -> z
    return jp.array([label['y'], -label['x']])


def parse_theta(label):
    return -label['theta_z']


def parse_scale(label):
    return jp.full(3, label['scale'])


def parse_class(label, name_to_arg):
    return jax.nn.one_hot(name_to_arg[label['mesh']], len(name_to_arg))


def parse_color(label):
    return jp.array(label['RGB']) / 255.0


def parse_label(label, name_to_class):
    shift = parse_shift(label)
    theta = parse_theta(label)
    scale = parse_scale(label)
    color = parse_color(label)
    shape = parse_class(label, name_to_class)
    return shift, theta, scale, color, shape


def parse_labels(labels, name_to_class, diffuse=0.9, ambient=0.1):
    samples = []
    for label in labels:
        label = list(label.values())[0]  # assumes single scenes
        shift, theta, scale, color, shape = parse_label(label, name_to_class)
        sample = [shift, theta, scale, color, ambient, diffuse, shape]
        samples.append(sample)
    return samples


def parse_metadata(root='datasets', dataset='PRIMITIVES'):
    metadata = os.path.join(root, dataset, 'metadata.json')
    metadata = json.load(open(metadata, 'r'))
    return metadata


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    RNG = np.random.default_rng(777)
    root = 'datasets'
    dataset = 'PRIMITIVES'
    split_name = 'train'
    num_ways = 30
    num_shot = 5
    num_test = 1

    dataset = load(os.path.join(root, dataset), split_name, (480, 640))
    shot_data, test_data = sample(RNG, dataset, num_ways, num_shot, num_test)
    shot_images, shot_labels = shot_data
    mosaic = make_mosaic(shot_images, (num_ways, num_shot), border=0)
    plt.imshow((mosaic * 255.0).astype('uint8'))
    plt.show()
