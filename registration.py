import tensorflow as tf


def register(moving, reference, movmat=tf.eye(4), refmat=tf.eye(4),
             cost_fn=tf.losses.mean_squared_error,
             optimizer=tf.train.AdamOptimizer(), **kwargs):
    """
    take two images and compute a registration between them
    :param moving: moving image to be aligned
    :param reference: reference image to align with
    :param verbose:
    :return:
    """
    # compute difference
    m2r_affine = refmat @ tf.linalg.inv(movmat)

    # create a meshgrid for each (should be in millimeters or ratio to reference size...)
    xdim = reference.shape
    ydim = moving.shape

    Xijk = tf.stack(tf.meshgrid(
        tf.range(xdim[0]), tf.range(xdim[1]), tf.range(xdim[2])
    ))
    Yijk = tf.stack(tf.meshgrid(
        tf.range(ydim[0]), tf.range(ydim[1]), tf.range(ydim[2])
    ))

    # extract reference shape data
    Xdims = tf.shape(Xijk)[1:]
    Xnpoints = tf.reduce_prod(Xdims)
    Ydims = tf.shape(Yijk)[1:]
    Ynpoints = tf.reduce_prod(Ydims)

    # flatten coordinates
    ones = tf.ones(shape=[1, 1, Xnpoints], dtype=tf.int32)
    Xijk = tf.reshape(Xijk, [1, 3, Xnpoints])
    Xijk = tf.concat((Xijk, ones), axis=1)
    Xijk = tf.transpose(Xijk, [0, 2, 1])

    ones = tf.ones(shape=[1, 1, Ynpoints], dtype=tf.int32)
    Yijk = tf.reshape(Yijk, [1, 3, Ynpoints])
    Yijk = tf.concat((Yijk, ones), axis=1)
    Yijk = tf.transpose(Yijk, [0, 2, 1])

    Xijk = tf.cast(Xijk, tf.float64)
    Yijk = tf.cast(Yijk, tf.float64)

    moving = tf.reshape(moving, [1, Ynpoints, 1])

    # initialize a transformation
    # define free parameters (6 dof or 12 dof)
    xfm = translation_matrix()

    # compute linear transformation on indices
    Yijk_transformed = tf.einsum('ij,abj->abi', xfm, Yijk)

    # calculate cost function
    # this will probably be fairly slow by comparison
    Yval = tf.contrib.image.interpolate_spline(Yijk_transformed, moving, Xijk,
                                               order=1)
    # it does not work :( I think it is attempting to compute output using ALL
    # training points.

    Yval = tf.reshape(Yval, xdim)
    ## cost
    loss = cost_fn(reference, Yval)

    # optimizer
    graph = optimizer.minimize(loss)

    return graph


def nonlinear_register(moving, reference, **kwargs):
    pass


def rotation_matrix():
    quaternion = tf.get_variable(name='rotation_params', shape=(4,),
                                 initializer=tf.initializers.random_normal,
                                 constraint=tf.linalg.l2_normalize,
                                 dtype=tf.float64)


def translation_matrix():
    params = tf.get_variable(name='params', shape=(3,),
                             initializer=tf.initializers.random_normal,
                             dtype=tf.float64)
    one = tf.constant((1,), dtype=tf.float64)
    col = tf.expand_dims(tf.concat((params, one), axis=0), 1)
    mat = tf.concat((tf.eye(4, 3, dtype=tf.float64), col), axis=1)

    return mat


def scaling_matrix():
    pass


def skew_matrix():
    pass


def main():
    pass
