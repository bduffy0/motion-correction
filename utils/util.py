
import tensorflow as tf


def create_image_summary(image,name):
    """
    given a 5d tensorflow tensor, create tensorboard image summary using the center slice after masking negative values

    :param image: tensorflow tensor
    :param name: string, summary image name
    """
    assert (image.get_shape().ndims == 5)
    # print('name = {} imshape = {}'.format(name,image.get_shape()))

    im_shape = image.get_shape().as_list()
    mid_pts = [int(x/2) for x in im_shape]
    image = tf.multiply(image,tf.cast(image>0,tf.float32))
    four_d_im = tf.expand_dims(image[mid_pts[0],:,:,mid_pts[3],:],0)
    tf.summary.image(name,four_d_im,max_outputs=1)
