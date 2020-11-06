import numpy as np
import tensorflow as tf

from tensorflow.keras import initializers
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras.constraints import Constraint


def wasserstein_loss(y_true, y_pred):
    """Compute the wasserstein loss from the true and predictions. The
    loss function is implemented by multiplying the expected label for
    each sample by the predicted score (element wise), then calculating
    the mean. For more details, see https://arxiv.org/abs/1506.05439

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


def compute_weights(xgrid):
    """compute_weights.

    Parameters
    ----------
    xgrid :
        xgrid
    """

    # Compute space between neighboring points
    spacing = [0.0]
    for x in range(1, xgrid.shape[0]):
        spacing.append(np.abs(xgrid[x - 1] - xgrid[x]))
    spacing.append(0.0)
    # Compute Weights array
    weights = []
    for s in range(xgrid.shape[0]):
        weights.append((spacing[s] + spacing[s + 1]) / 2)
    weights = np.array(weights).reshape(1, xgrid.shape[0])
    return weights


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
        Grids of PDF
    """

    def __init__(self, pdf, **kwargs):
        self.pdf = K.constant(pdf)
        super(ConvPDF, self).__init__(**kwargs)

    def call(self, previous_layer):
        if previous_layer.shape[0] is not None:
            pdfshuffled = tf.random.shuffle(self.pdf)
            sampled_pdf = pdfshuffled[:previous_layer.shape[0]]
            return previous_layer * sampled_pdf
        return previous_layer


class ImposeSumRules(Layer):
    """Impose sum rules on PDF.

    Parameters
    ----------
    xgrid: np.array
        Array of x values
    """

    def __init__(self, xgrid, **kwargs):
        nb_basis_to_divide = 6
        xgrid = K.constant(xgrid)
        arr_ones = K.ones((2, xgrid.shape[0]))
        self.weight = K.constant(compute_weights(xgrid))
        # Compute 1/x
        xgrid = K.reshape(xgrid, shape=(1, xgrid.shape[0]))
        inverse_x = K.tile(1 / xgrid, (nb_basis_to_divide, 1))
        inverse_x = K.concatenate([arr_ones, inverse_x], axis=0)
        # Weight 1/x
        self.inverseXweighted = inverse_x * self.weight
        super(ImposeSumRules, self).__init__(**kwargs)

    def call(self, previous_layer):
        if previous_layer.shape[0] is not None:
            # Multiplication along the last two axes
            xsummed = K.sum(previous_layer * self.inverseXweighted, axis=2)     # Shape=(batches, evol)
            xtransposed = K.transpose(xsummed)                                  # shape=(evol, batches)
            gg_norm = (1 - xtransposed[0]) / xtransposed[1]
            vv_norm = 3 / xtransposed[2]
            v3_norm = 1 / xtransposed[3]
            v8_norm = 3 / xtransposed[4]
            # Define Normalization tensor
            ones = K.ones_like(gg_norm)
            normalization_constant = K.stack([
                        ones,
                        gg_norm,
                        vv_norm,
                        v3_norm,
                        v8_norm,
                        ones,
                        ones,
                        ones
                    ], axis=1)  # Shape=(batches, evol)
            final_normalization = K.expand_dims(normalization_constant, axis=-1)
            normalized_pdf = previous_layer * final_normalization
            return normalized_pdf
        return previous_layer


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

    def call(self, pdf):
        xres = self.xval ** self.kernel[0] * (1 - self.xval) ** self.kernel[1]
        return pdf * xres
