from jax import numpy as jp


def rotation_z(angle):
    """Builds affine rotation matrix in Z axis.

    # Arguments
        angle: Float. Angle in radians.

    # Return
        Array (4, 4) rotation matrix in Z axis.
    """
    cos_angle = jp.cos(angle)
    sin_angle = jp.sin(angle)
    rotation_matrix_z = jp.array([[+cos_angle, -sin_angle, 0.0, 0.0],
                                  [+sin_angle, +cos_angle, 0.0, 0.0],
                                  [0.0, 0.0, 1.0, 0.0],
                                  [0.0, 0.0, 0.0, 1.0]])
    return rotation_matrix_z


def rotation_x(angle):
    """Builds affine rotation matrix in X axis.

    # Arguments
        angle: Float. Angle in radians.

    # Return
        Array (4, 4) rotation matrix in Z axis.
    """
    cos_angle = jp.cos(angle)
    sin_angle = jp.sin(angle)
    rotation_matrix_x = jp.array([[1.0, 0.0, 0.0, 0.0],
                                  [0.0, +cos_angle, -sin_angle, 0.0],
                                  [0.0, +sin_angle, +cos_angle, 0.0],
                                  [0.0, 0.0, 0.0, 1.0]])
    return rotation_matrix_x


def rotation_y(angle):
    """Builds affine rotation matrix in Y axis.

    # Arguments
        angle: Float. Angle in radians.

    # Return
        Array (4, 4) rotation matrix in Z axis.
    """
    cos_angle = jp.cos(angle)
    sin_angle = jp.sin(angle)
    rotation_matrix_y = jp.array([[+cos_angle, 0.0, +sin_angle, 0.0],
                                  [0.0, 1.0, 0.0, 0.0],
                                  [-sin_angle, 0.0, +cos_angle, 0.0],
                                  [0.0, 0.0, 0.0, 1.0]])
    return rotation_matrix_y


def translation(translation_vector):
    """Builds affine translation matrix

    # Arguments
        angle: Array (3) having [x, y, z] coordinates.

    # Return
        Array (4, 4) translation matrix.
    """
    x, y, z = translation_vector
    return jp.array([[1.0, 0.0, 0.0, x],
                     [0.0, 1.0, 0.0, y],
                     [0.0, 0.0, 1.0, z],
                     [0.0, 0.0, 0.0, 1.0]])


def scaling(scaling_vector):
    """Builds scaling translation matrix

    # Arguments
        angle: Array (3) having [x, y, z] scaling values.

    # Return
        Array (4, 4) scale matrix.
    """
    x, y, z = scaling_vector
    return jp.array([[x, 0.0, 0.0, 0.0],
                     [0.0, y, 0.0, 0.0],
                     [0.0, 0.0, z, 0.0],
                     [0.0, 0.0, 0.0, 1.0]])


def shearing(shearing_vector):
    x_y, x_z, y_x, y_z, z_x, z_y = shearing_vector
    return jp.array([[1.0, x_y, x_z, 0.0],
                     [y_x, 1.0, y_z, 0.0],
                     [z_x, z_y, 1.0, 0.0],
                     [0.0, 0.0, 0.0, 1.0]])


def view_transform(camera_origin, target_position, world_up):
    camera_forward = target_position - camera_origin
    camera_forward = camera_forward / jp.linalg.norm(camera_forward)
    world_up = world_up / jp.linalg.norm(world_up)
    camera_left = jp.cross(camera_forward, world_up)
    left_x, left_y, left_z = camera_left
    up_x, up_y, up_z = jp.cross(camera_left, camera_forward)
    x_forward, y_forward, z_forward = camera_forward
    orientation = jp.array([[left_x, left_y, left_z, 0.0],
                            [up_x, up_y, up_z, 0.0],
                            [-x_forward, -y_forward, -z_forward, 0.0],
                            [0.0, 0.0, 0.0, 1.0]])
    translate = translation(-camera_origin)
    view_transform = jp.matmul(orientation, translate)
    return view_transform
