from __future__ import division

import keras.backend as K
from keras.layers import Layer
from keras.layers import initializers


class xlayer(Layer):

    """
    Custom array that inputs the information on the x-grid.
    """

    def __init__(self, output_dim, xval, kernel_initializer='glorot_uniform', **kwargs):
        self.output_dim = output_dim
        self.xval = K.constant(xval)
        self.kernel_initializer = initializers.get(kernel_initializer)
        super(xlayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel', shape=(K.int_shape(self.xval)[0],
                input_shape[1], self.output_dim), initializer=self.kernel_initializer,
                trainable=True)
        super(xlayer, self).build(input_shape)

    def call(self, x):
        # xres outputs (None, input_shape[1], len(x_pdf))
        xres = K.tf.tensordot(x, self.xval, axes=0)
        # xfin outputs (None, output_dim)
        xfin = K.tf.tensordot(xres, self.kernel, axes=([1,2],[0,1]))
        return xfin

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
