import os
import numpy as np

import tensorflow as tf
from scipy import stats
from tensorflow.keras import initializers
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras.constraints import Constraint
from tensorflow.keras.losses import BinaryCrossentropy


def wasserstein_loss(y_true, y_pred):
    """Compute the wasserstein loss from the true and predictions. The loss function 
    is implemented by multiplying the expected label for each sample by the predicted 
    score (element wise), then calculating the mean. For more details, see
    https://arxiv.org/abs/1506.05439

    Parameters
    ----------
    y_true : tf.tensor(float)
        list of estimated probabilities for the true
        samples
    y_pred : tf.tensor(float)
        list of estimated probabilities for the fake
        samples
    """
    return K.mean(y_true * y_pred)


class ClipConstraint(Constraint):
    """Put constraints on the weights of a given
    Layer.
    """

    def __init__(self, clip_value):
        self.clip_value = clip_value

    def __call__(self, weights):
        return K.clip(weights, -self.clip_value, self.clip_value)

    def get_config(self):
        return {"clip_value": self.clip_value}


class ConvPDF(Layer):
    """Convolute the output of the previous layer with
    a subsample of the input/prior replica.

    Parameters
    ----------
    pdf: np.array
        Array of PDF grids
    """

    def __init__(self, pdf, trainable=True, **kwargs):
        index = np.random.randint(pdf.shape[0])
        self.pdf = K.constant(pdf[index])
        self.trainable = trainable
        super(ConvPDF, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return (None, self.pdf.shape[1], self.pdf.shape[2])

    def call(self, previous_layer):
        mult = previous_layer * self.pdf
        return mult


class ConvXgrid(Layer):
    """Convolute the output of the prvious layer with the 
    input x-grid."""

    def __init__(self, output_dim, xval, kernel_initializer="glorot_uniform", **kwargs):
        self.output_dim = output_dim
        self.xval = K.constant(xval)
        self.kernel_initializer = initializers.get(kernel_initializer)
        super(ConvXgrid, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(
            name="kernel",
            shape=(K.int_shape(self.xval)[0], input_shape[1], self.output_dim),
            initializer=self.kernel_initializer,
            trainable=True,
        )
        super(ConvXgrid, self).build(input_shape)

    def call(self, x):
        # xres outputs (None, input_shape[1], len(x_pdf))
        xres = tf.tensordot(x, self.xval, axes=0)
        # xfin outputs (None, output_dim)
        xfin = tf.tensordot(xres, self.kernel, axes=([1, 2], [0, 1]))
        return xfin

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


class PreprocessFit(Layer):

    """Add preprocessing to output of the previous layer. This probably assures
    the PDF-like behavior of the generated samples.

    Parameters
    ----------
    xval: np.array
        Array of x-grid
    """

    def __init__(self, xval, trainable=True, kernel_initializer="ones", **kwargs):
        self.xval = xval
        self.trainable = trainable
        self.kernel_initializer = initializers.get(kernel_initializer)
        super(PreprocessFit, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(
            name="kernel",
            shape=(2,),
            initializer=self.kernel_initializer,
            trainable=self.trainable,
        )
        super(PreprocessFit, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape

    # @tf.function
    def call(self, pdf):
        xres = self.xval ** self.kernel[0] * (1 - self.xval) ** self.kernel[1]
        return pdf * xres
