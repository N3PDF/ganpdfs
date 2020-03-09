import numpy as np
import tensorflow as tf
from tensorflow.keras import initializers
from tensorflow.keras import backend as K
from tensorflow.keras.constraints import Constraint
from tensorflow.keras.layers import Layer


"""
Custom functions
"""

def wasserstein_loss(y_true, y_pred):
    """
    Function that computes Wasserstein loss
    """
    return K.mean(y_true * y_pred)


def get_method(class_name, method):
    """
    Get the methods in a class
    """
    return getattr(class_name, method, None)


"""
Custom classes
"""

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
        return K.mean(y_true * y_pred)


class ClipConstraint(Constraint):

    """
    The following constrains the weights
    of the critic model. This class is an
    extension of the "Constraint" class.
    """

    def __init__(self, clip_value):
        self.clip_value = clip_value

    def __call__(self, weights):
        return K.clip(weights, -self.clip_value, self.clip_value)

    def get_config(self):
        return {"clip_value": self.clip_value}


class estimators(object):
    """
    Class that contains methods for the estimator computation
    """
    def __init__(self, true_distr, fake_distr):
        self.true_distr = true_distr
        self.fake_distr = fake_distr

    def mean(self):
        """
        Compute mean
        """
        return np.mean(self.true_distr), np.mean(self.fake_distr)

    def stdev(self):
        """
        Compute standard deviation
        """
        return np.std(self.true_distr), np.std(self.fake_distr)


class smm(object):

    """
    Similarity Metric Method (SMM)
    Custom metrics in order to assess the performance of the model.
    """

    def __init__(self, y_true, y_pred, params):
        epsilon = 1e-1
        self.y_true = y_true + epsilon
        self.y_pred = y_pred + epsilon
        self.estmtr = params['estimators']

    def kullback(self):
        """
        Kullback-Leibler divergence D(P || Q) for discrete distributions.
        For each value of p and q:
        \sum_{i}^{n} p_i*\log(p_i/q_i)
        """
        val = []
        for i in range(self.y_true.shape[0]):
            arr = np.where(
                self.y_true[i] != 0,
                self.y_true[i] * np.log(self.y_true[i] / self.y_pred[i]),
                0,
            )
            val.append(np.sum(arr))
        res = np.array(val)
        return res, np.mean(res)

    def euclidean(self):
        """
        Compute the Euclidean distance between two lists A and B.
        euc = \sum_{i}^{n} (A_i-B_i)^2.
        If A and B are the same, euc=0.
        """
        res = np.linalg.norm(self.y_true - self.y_pred)
        return res

    def ERF(self):
        # Normalization factor
        Nk = 1
        sum1, sum2 = 0, 0
        # Loop over the list of estimators
        for es in self.estmtr:
            for xi in range(self.y_true.shape[1]):
                # Initialize estimator class
                es_class = estimators(self.y_true[:,xi], self.y_pred[:,xi])
                # Evaluate obs for different estimator
                es_true, es_fake = get_method(es_class, es)()
                # Sum over the nb of grid
                sum1 += ((es_fake-es_true)/es_true)**2
            # Sum over the nb of estimators
            sum2 += Nk*sum1
        return sum2
