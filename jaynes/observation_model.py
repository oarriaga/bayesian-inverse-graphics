import jax.numpy as jp
from tamayo import SE3, Shape, Material, merge_shapes
from tamayo.abstract import Pattern
from tamayo.constants import NO_PATTERN

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors


def compute_camera_intrinsics(y_FOV, H, W):
    aspect_ratio = W / H
    y_focal_length = 1 / (jp.tan(y_FOV / 2.0))
    x_focal_length = y_focal_length / aspect_ratio

    x_translation = W / 2.0
    y_translation = H / 2.0
    return jp.array([[x_focal_length * (W / 2.0), 0.0, x_translation, 0.0],
                     [0.0, y_focal_length * (H / 2.0), y_translation, 0.0],
                     [0.0, 0.0, 1.0, 0.0]])


def compute_camera_matrix(camera_pose, y_FOV, H, W):
    camera_intrinsics = compute_camera_intrinsics(y_FOV, H, W)
    return jp.matmul(camera_intrinsics, camera_pose)


def homogenize_coordinates(point):
    return jp.concatenate([point, jp.ones(1)]).reshape(-1, 1)


def dehomogenize_coordinates(homogenous_point):
    homogenous_point = jp.squeeze(homogenous_point, axis=1)
    u, v, w = homogenous_point
    return jp.array([u / w, v / w])


def Projector(y_FOV, H, W, camera_pose):
    camera_matrix = compute_camera_matrix(camera_pose, y_FOV, H, W)

    def project(point3D):
        homogenous_point3D = homogenize_coordinates(point3D)
        homogenous_point2D = jp.matmul(camera_matrix, homogenous_point3D)
        point2D = dehomogenize_coordinates(homogenous_point2D)
        return point2D

    return project


def Attention(project, soften, H, W):
    x_range = jp.arange(0.0, 1.0, 1.0 / W)
    y_range = jp.arange(0.0, 1.0, 1.0 / H)
    x_grids, y_grids = jp.meshgrid(x_range, y_range)
    x_grids = jp.expand_dims(x_grids, axis=-1)
    y_grids = jp.expand_dims(y_grids, axis=-1)
    attention_map = jp.concatenate([x_grids, y_grids], axis=2)
    aspect_ratio = W / H

    def build_attention_map(keypoint, stdv=0.01):
        x_mean = keypoint[0] / W
        y_mean = keypoint[1] / H
        x_stdv = stdv
        y_stdv = stdv * aspect_ratio
        mean = jp.array([x_mean, y_mean])
        stdv = jp.array([x_stdv, y_stdv])
        distribution = tfd.MultivariateNormalDiag(mean, stdv)
        attention_density = distribution.prob(attention_map)
        soft_attention_map = soften(attention_density)
        return jp.expand_dims(soft_attention_map, axis=-1)

    def attent(sample):
        point3D = sample.transform[:3, 3]
        point2D = project(point3D)
        attention_map = build_attention_map(point2D, 0.05)
        return attention_map

    return attent


def denormalize_image(image):
    return 255.0 * image


def ObservationModel(render, floor, lights):
    zero_image = jp.zeros_like(floor.pattern.image)
    pattern = Pattern(jp.eye(4), NO_PATTERN, zero_image)

    def sample_to_shape(sample):
        x = sample['shift'][..., 0]
        z = sample['shift'][..., 1]
        y = sample['scale'][..., 1]
        translate = jp.array([x, y, z])
        translate = SE3.translation(translate)
        rotate = SE3.rotation_y(sample['theta'])
        scale = SE3.scaling(sample['scale'])
        transform = translate @ rotate @ scale
        material = Material(sample['color'],
                            sample['ambient'],
                            sample['diffuse'],
                            sample['specular'],
                            sample['shininess'])
        arg = jp.argmax(sample['classes'])
        return Shape(transform, arg, pattern, material)

    def apply(sample):
        shape = sample_to_shape(sample)
        scene, mask = merge_shapes(shape, floor)
        image, depth = render(scene, mask, lights)
        image = jp.clip(image, 0.0, 1.0)
        return image, depth

    return apply


def preprocess_input(image, mean=[103.939, 116.779, 123.68]):
    image = image.astype(jp.float32)
    image = image[..., ::-1]
    image = jp.moveaxis(image, 2, 0)
    mean = jp.array(mean)
    mean = jp.expand_dims(mean, axis=[1, 2])
    return image - mean


def compute_feature_loss(true_features, pred_features):
    losses = []
    for true_feature, pred_feature in zip(true_features, pred_features):
        loss = (true_feature - pred_feature)**2
        loss = jp.mean(loss, axis=(0, 1))
        losses.append(loss)
    return jp.array(losses)


def NeuroLikelihood(weight, branch_model):
    def apply(true_image, pred_image):
        true_image = denormalize_image(true_image)
        pred_image = denormalize_image(pred_image)
        true_image = preprocess_input(true_image)
        pred_image = preprocess_input(pred_image)
        true_features = branch_model(true_image)
        pred_features = branch_model(pred_image)
        losses = compute_feature_loss(true_features, pred_features)
        return -weight * (losses.sum())
    return apply


def Likelihood(observation_model, noise_model, neuro_model):

    def apply(forward_samples, true_image):
        pred_image, pred_depth = observation_model(forward_samples)
        color_log_prob = noise_model.log_prob(true_image - pred_image).sum()
        neuro_log_prob = neuro_model(true_image, pred_image)
        return color_log_prob + neuro_log_prob

    return apply


def parse_summary(summary, statistic='mean'):
    x_shift = summary[statistic]['shift[0]']
    y_shift = summary[statistic]['shift[1]']
    shift = jp.array([x_shift, y_shift])

    theta = jp.array(summary['mode']['theta'])

    x_scale = summary[statistic]['scale[0]']
    y_scale = summary[statistic]['scale[1]']
    z_scale = summary[statistic]['scale[2]']
    scale = jp.array([x_scale, y_scale, z_scale])

    r_color = summary[statistic]['color[0]']
    g_color = summary[statistic]['color[1]']
    b_color = summary[statistic]['color[2]']
    color = jp.array([r_color, g_color, b_color])

    ambient = jp.array(summary[statistic]['ambient'])
    diffuse = jp.array(summary[statistic]['diffuse'])
    specular = jp.array(summary[statistic]['specular'])
    shininess = jp.array(summary[statistic]['shininess'])

    class_0 = summary[statistic]['classes[0]']
    class_1 = summary[statistic]['classes[1]']
    class_2 = summary[statistic]['classes[2]']
    classes = jp.array([class_0, class_1, class_2])
    return {'shift': shift, 'theta': theta, 'scale': scale, 'color': color,
            'ambient': ambient, 'diffuse': diffuse, 'specular': specular,
            'shininess': shininess, 'classes': classes}


def estimate_point(summary, render, statistic='mean'):
    sample = parse_summary(summary, statistic)
    point_image, point_depth = render(sample)
    return point_image, point_depth
