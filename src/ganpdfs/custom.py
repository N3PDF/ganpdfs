import os
import numpy as np

# Silent tf for the time being
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from scipy import stats
from tensorflow.keras import initializers
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras.constraints import Constraint
from tensorflow.keras.losses import BinaryCrossentropy


def wasserstein_loss(y_true, y_pred):
    """
    Function that computes Wasserstein loss
    """
    return tf.reduce_mean(y_true * y_pred)


def disc_loss(real_output, fake_output):
    real_loss = BinaryCrossentropy(from_logits=True)(tf.ones_like(real_output), real_output)
    fake_loss = BinaryCrossentropy(from_logits=True)(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def genr_loss(fake_output):
    return BinaryCrossentropy(from_logits=True)(tf.ones_like(fake_output), fake_output)


class xlayer(Layer):

    """
    Custom array that inputs the information on the x-grid.
    """

    def __init__(self, output_dim, xval, kernel_initializer="glorot_uniform", **kwargs):
        self.output_dim = output_dim
        self.xval = K.constant(xval)
        self.kernel_initializer = initializers.get(kernel_initializer)
        super(xlayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(
            name="kernel",
            shape=(K.int_shape(self.xval)[0], input_shape[1], self.output_dim),
            initializer=self.kernel_initializer,
            trainable=True,
        )
        super(xlayer, self).build(input_shape)

    # @tf.function
    def call(self, x):
        # xres outputs (None, input_shape[1], len(x_pdf))
        xres = tf.tensordot(x, self.xval, axes=0)
        # xfin outputs (None, output_dim)
        xfin = tf.tensordot(xres, self.kernel, axes=([1, 2], [0, 1]))
        return xfin

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


class preprocessing_fit(Layer):

    """
    Custom array that does the preprocessing.
    Here, the parameters are fitted.
    """

    def __init__(self, xval, trainable=True, kernel_initializer="ones", **kwargs):
        self.xval = xval
        self.trainable = trainable
        self.kernel_initializer = initializers.get(kernel_initializer)
        super(preprocessing_fit, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(
            name="kernel",
            shape=(2,),
            initializer=self.kernel_initializer,
            trainable=self.trainable,
        )
        super(preprocessing_fit, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape

    # @tf.function
    def call(self, pdf):
        xres = self.xval ** self.kernel[0] * (1 - self.xval) ** self.kernel[1]
        return pdf * xres


class custom_losses(object):

    """
    The following is the implementation of the
    custom loss functions
    """

    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred

    def wasserstein_loss(self):
        return tf.reduce_mean(self.y_true * self.y_pred)


class ClipConstraint(Constraint):

    """
    The following constrains the weights
    of the critic model. This class is an
    extension of the "Constraint" class.
    """

    def __init__(self, clip_value):
        self.clip_value = clip_value

    # @tf.function
    def __call__(self, weights):
        return K.clip(weights, -self.clip_value, self.clip_value)

    def get_config(self):
        return {"clip_value": self.clip_value}
