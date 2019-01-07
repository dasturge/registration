import tensorflow as tf


def register(moving, reference, cost_fn=tf.losses.mean_squared_error,
             optimizer=tf.train.AdamOptimizer(), **kwargs):
    """
    take two images and compute a registration between them
    :param moving: moving image to be aligned
    :param reference: reference image to align with
    :param verbose:
    :return:
    """
    # maybe load data, and convert to tensors

    # create a meshgrid for each (should be in millimeters or ratio to reference size...)
    Xijk = tf.stack(tf.meshgrid())
    Yijk = tf.stack(tf.meshgrid())

    # extract reference shape data
    Xdims = tf.shape(Xijk)[1:]
    Xnpoints = tf.reduce_prod(Xdims)
    Xijk_flat = tf.reshape(Xijk, [1, 4, Xnpoints])
    Xijk_flat = tf.transpose(Xijk_flat, [0, 2, 1])

    Ydims = tf.shape(Yijk)[1:]
    Ynpoints = tf.reduce_prod(Ydims)

    ## get boundaries for Xijk image

    # initialize a transformation
    # define free parameters (6 dof or 12 dof)
    theta1= tf.Variable()
    theta2= tf.Variable()
    xfm = tf.eye(4)

    # compute transformation on indices
    Yijk_new = tf.einsum('ij,jxyz->ixyz', xfm, Yijk)

    # calculate cost function
    # this will probably be fairly slow by comparison
    Yijk_flat = tf.reshape(Yijk_new, [1, 4, Ynpoints])
    Yijk_flat = tf.transpose(Yijk_flat, [0, 2, 1])
    Yval = tf.contrib.image.interpolate_spline(Yijk_flat, moving, Xijk_flat,
                                               order=1)
    ## cost
    loss = cost_fn(reference, Yval)

    # optimizer
    optimizer.minimize(loss)


def nonlinear_register(moving, reference, **kwargs):
    pass


def rotation_matrix():
    thetax = tf.get_variable()
    thetay = tf.get_variable()
    thetaz = tf.get_variable()




def translation_matrix():
    params = tf.get_variable(name='params',
                             initializer=tf.initializers.random_normal)
    one = tf.constant((1,))
    col = tf.stack((params, one))
    mat = tf.concat((tf.eye(4, 3), col), axis=1)

    return mat

def scaling_matrix():
    pass


def skew_matrix():
    pass


def main():
    pass
