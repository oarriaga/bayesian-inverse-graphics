from jax import numpy as jp


def rotation_z(angle):
    """Builds rotation matrix in Z axis.

    # Arguments
        angle: Float. Angle in radians.

    # Return
        Array (4, 4) rotation matrix in Z axis.
    """
    cos_angle = jp.cos(angle)
    sin_angle = jp.sin(angle)
    rotation_matrix_z = jp.array([[+cos_angle, -sin_angle, 0.0],
                                  [+sin_angle, +cos_angle, 0.0],
                                  [0.0, 0.0, 1.0]])
    return rotation_matrix_z


def rotation_x(angle):
    """Builds rotation matrix in X axis.

    # Arguments
        angle: Float. Angle in radians.

    # Return
        Array (4, 4) rotation matrix in Z axis.
    """
    cos_angle = jp.cos(angle)
    sin_angle = jp.sin(angle)
    rotation_matrix_x = jp.array([[1.0, 0.0, 0.0],
                                  [0.0, +cos_angle, -sin_angle],
                                  [0.0, +sin_angle, +cos_angle]])
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
    rotation_matrix_y = jp.array([[+cos_angle, 0.0, +sin_angle],
                                  [0.0, 1.0, 0.0],
                                  [-sin_angle, 0.0, +cos_angle]])
    return rotation_matrix_y
