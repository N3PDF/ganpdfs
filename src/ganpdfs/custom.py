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


def compute_smr(pdf, inverse_x):
    """Impose sum rules on a bacth of PDF by performing the following:
    -   Compute the sum of the PDF over x while dividing the basis by x
        except for `sigma` and `g`.
    -   Take the x grid and expand the dimension such that the output
        shape is exactly the same as the batched PDF such that both can
        just be multiplied together.

    Parameters
    ----------
    pdf: tf
        Batches of PDF
    xgrid: tf
        Grid of x values

    Returns
    -------
    """
    sum_overx = K.sum(pdf * inverse_x, axis=2)
    # Compute Normalization Factor
    # ones = K.ones(2)
    # pdf_arr, sum_overx_arr = K.eval(pdf), K.eval(sum_overx)
    # vv_norm = [3 / v[2] for v in sum_overx]
    # v3_norm = [1 / v[3] for v in sum_overx]
    # v8_norm = [3 / v[4] for v in sum_overx]
    # gg_norm = [(1 - v[0]) / v[1] for v in sum_overx_arr]
    # norm_arr = np.array([gg_norm, vv_norm, v3_norm, v8_norm, ones, ones, ones, ones])
    # Perform Multiplication
    # pdf_norm = []
    # for p, n in zip(pdf_arr, norm_arr):
    #     normalized = (p.T * n.T).T
    #     pdf_norm.append(normalized)
    # pdf_norm = np.array(pdf_norm)
    # return K.constant(pdf_norm)
    return sum_overx


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


class ImposeSumRules(Layer):
    """Impose sum rules on PDF.

    Parameters
    ----------
    xgrid: np.array
        Array of x values
    """

    def __init__(self, xgrid, **kwargs):
        self.xgrid = K.constant(xgrid)
        super(ImposeSumRules, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return input_shape
    
    def build(self, input_shape):
        nb_basis_to_divide = 6
        arr_ones = K.ones((2, self.xgrid.shape[0]))
        xgrid = K.reshape(self.xgrid, shape=(1, self.xgrid.shape[0]))
        inverse_x = K.tile(1 / xgrid, (nb_basis_to_divide, 1))
        self.inverse_x = K.concatenate([arr_ones, inverse_x], axis=0)
        # super(ImposeSumRules, self).build(input_shape)

    def call(self, previous_layer):
        if previous_layer.shape[0] is not None:
            # Multiplication along the last two axes
            xsummed = K.sum(previous_layer * self.inverse_x, axis=2)        # Shape=(batches, evol)
            xtransposed = K.permute_dimensions(xsummed, pattern=(1, 0))     # Shape=(evol, batches)
            gg_norm = (1 - xtransposed[0]) / xtransposed[1]
            vv_norm = 3 / xtransposed[2]
            v3_norm = 1 / xtransposed[3]
            v8_norm = 3 / xtransposed[4]
            normalized_pdf = K.concatenate(
                    [
                        [xtransposed[0]],
                        [gg_norm],
                        [vv_norm],
                        [v3_norm],
                        [v8_norm],
                        [xtransposed[5]],
                        [xtransposed[6]],
                        [xtransposed[7]]
                    ],
                    axis=0
            )   # Shape=(evol, batches) 
            # Permute the dimensions
            pdfpermute = K.permute_dimensions(
                    previous_layer,
                    pattern=(2, 1, 0)
            )
            previous_layer = K.permute_dimensions(
                    pdfpermute * normalized_pdf,
                    pattern=(0, 1, 2)
            )
        return previous_layer


class ConvPDF(Layer):
    """Convolute the output of the previous layer with
    a subsample of the input/prior replica.

    Parameters
    ----------
    pdf: np.array
        Array of PDF grids
    """

    def __init__(self, pdf, **kwargs):
        index = np.random.randint(pdf.shape[0])
        self.pdf = K.constant(pdf[index])
        super(ConvPDF, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return input_shape

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

    def call(self, pdf):
        xres = self.xval ** self.kernel[0] * (1 - self.xval) ** self.kernel[1]
        return pdf * xres
