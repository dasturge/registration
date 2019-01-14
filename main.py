import nibabel as nb
import tensorflow as tf

import registration


def main():
    image1 = './T1w.nii.gz'
    image2 = './T1w_acpc.nii.gz'

    image1 = nb.load(image1)
    image2 = nb.load(image2)

    im1hd = image1.affine
    im2hd = image2.affine

    imdata1 = image1.get_fdata()
    imdata2 = image2.get_fdata()

    imdata1 = tf.constant(imdata1)
    imdata2 = tf.constant(imdata2)
    im1hd = tf.constant(im1hd)
    im2hd = tf.constant(im2hd)


    graph = registration.register(imdata1, imdata2, im1hd, im2hd)

    init_op = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init_op)
        graph = sess.run(graph)

    x = 1


if __name__ == '__main__':
    main()
