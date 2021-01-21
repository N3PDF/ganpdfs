import os
import numpy as np

import tensorflow as tf
from scipy import stats
from tensorflow.keras import optimizers
from tensorflow.keras import constraints
from tensorflow.keras import initializers
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras.constraints import Constraint
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.initializers import Zeros, Identity


def wasserstein_loss(y_true, y_pred):
    """Compute the wasserstein loss from the true and predictions.
    The loss function is implemented by multiplying the expected
    label for each sample by the predicted score (element wise),
    then calculating the mean. For more details, refer to:
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


def get_optimizer(optimizer):
    """Overwrite keras default parameters for optimizer.

    Parameters
    ----------
    optimizer : dict
        Dictionary containing information on the optimizer.

    Returns
    -------
    tf.keras.optimizers:
        Optimizer.
    """

    learning_rate = optimizer.get("learning_rate")
    optimizer_name = optimizer.get("optimizer_name")
    optimizer_class = optimizers.get(optimizer_name)
    optimizer_class.learning_rate = learning_rate
    return optimizer_class


def get_activation(model_params):
    """Extract activation functions from the input parameters.
    This is necessary for advanced activation functions.

    Parameters
    ----------
    model_params : dict
        Dictionary containing information on the Discriminator
        architecture.


    Returns
    -------
    tf.keras.activations:
        Activation function.
    """

    if model_params.get("activation" == "elu"):
        from tensorflow.keras.layers import ELU
        return ELU(alpha=1.0)
    elif model_params.get("activation") == "relu":
        from tensorflow.keras.layers import ReLU
        return ReLU()
    elif model_params.get("activation") == "leakyrelu":
        from tensorflow.keras.layers import LeakyReLU
        return LeakyReLU(alpha=0.2)
    else: raise ValueError("Activation not available.")


class WeightsClipConstraint(Constraint):
    """Put constraints on the weights of a given layer.

    Parameters
    ----------
    value : float
        Value to which the weights will be bounded on.
    """

    def __init__(self, value):
        self.value = value

    def __call__(self, weights):
        return K.clip(weights, -self.value, self.value)

    def get_config(self):
        return {"value": self.value}


class GenDense(Layer):

    def __init__(self, output_dim, use_bias, **kwargs):
        self.use_bias = use_bias
        self.units = output_dim
        self.binitializer = Zeros()
        self.kinitializer = Identity()
        super(GenDense, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(
            name='kernel',
            shape=(input_shape[-1], self.units),
            initializer=self.kinitializer,
            trainable=True
        )
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.units),
                initializer=self.binitializer,
                trainable=True
            )
        else: self.bias = None
        super(GenDense, self).build(input_shape)

    def call(self, inputs):
        output = K.dot(inputs, self.kernel)
        if self.use_bias: output = output + self.bias
        return output


class DiscDense(Layer):

    def __init__(self, output_dim, dicparams, **kwargs):
        wc = WeightsClipConstraint(dicparams.get("weights_constraints")) \
                if output_dim is not 1 else None
        self.units = output_dim
        self.kconstraint = constraints.get(wc)
        self.binitializer = initializers.get(dicparams["bias_initializer"])
        self.kinitializer = initializers.get(dicparams["kernel_initializer"])
        self.use_bias = dicparams.get("use_bias")
        self.activation = dicparams.get("activation")
        super(DiscDense, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(
            name='kernel',
            shape=(input_shape[-1], self.units),
            initializer=self.kinitializer,
            trainable=True,
            constraint=self.kconstraint
        )
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.units),
                initializer=self.binitializer,
                trainable=True
            )
        else: self.bias = None
        super(DiscDense, self).build(input_shape)

    def call(self, inputs):
        output = K.dot(inputs, self.kernel)
        if self.use_bias: output = output + self.bias
        return output


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
    """Convolute the output of the previous layer with the input x-grid."""

    def __init__(self, output_dim, xval, kinit="glorot_uniform", **kwargs):
        self.units = output_dim
        self.xval = K.constant(xval)
        self.kernel_initializer = initializers.get(kinit)
        super(ConvXgrid, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(
            name="kernel",
            shape=(K.int_shape(self.xval)[0], input_shape[1], self.units),
            initializer=self.kernel_initializer,
            trainable=True,
        )
        super(ConvXgrid, self).build(input_shape)

    def call(self, x):
        # xres outputs (None, input_shape[1], len(x_pdf))
        xres = tf.tensordot(x, self.xval, axes=0)
        # xfin outputs (None, units)
        xfin = tf.tensordot(xres, self.kernel, axes=([1, 2], [0, 1]))
        return xfin

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)


class PreprocessFit(Layer):
    """Add preprocessing to the output of the previous layer. This is
    expected to assure the PDF-like behavior of the generated samples.

    Parameters
    ----------
    xval: np.array
        Array of x-grid
    """

    def __init__(self, xval, trainable=True, kinit="ones", **kwargs):
        self.xval = xval
        self.trainable = trainable
        self.kernel_initializer = initializers.get(kinit)
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

    def call(self, pdf):
        xres = self.xval ** self.kernel[0]
        zres = (1 - self.xval) ** self.kernel[1]
        return pdf * xres * zres
