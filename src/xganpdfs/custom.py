from __future__ import division

import numpy as np
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


class preprocessing(Layer):

    """
    Custom array that does the preprocessing.
    """

    def __init__(self, xval, alpha, beta, **kwargs):
        self.xval  = xval
        self.alpha = alpha
        self.beta  = beta
        super().__init__(**kwargs)

    def compute_output_shape(self, x):
        return x

    def call(self, pdf):
        xres = self.xval**self.alpha * (1-self.xval)**self.beta
        return pdf*xres


class preprocessing_fit(Layer):

    """
    Custom array that does the preprocessing.
    Here, the parameters are fitted.
    """

    def __init__(self, xval, trainable=True, kernel_initializer='ones', **kwargs):
        self.xval = xval
        self.trainable = trainable
        self.kernel_initializer = initializers.get(kernel_initializer)
        super(preprocessing_fit, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel', shape=(2,),
                initializer=self.kernel_initializer,
                trainable=self.trainable)
        super(preprocessing_fit, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, pdf):
        xres = self.xval**self.kernel[0] * (1-self.xval)**self.kernel[1]
        return pdf*xres


class xmetrics(object):
    """
    Custom metrics in order to assess the performance of the model.
    """

    def __init__(self, y_true, y_pred):
        epsilon = 1e-3
        self.y_true = y_true + epsilon
        self.y_pred = y_pred + epsilon

    def kullback(self):
        """
        Kullback-Leibler divergence D(P || Q) for discrete distributions.
        For each value of p and q:
        \sum_{i}^{n} p_i*\log(p_i/q_i)
        """
        val = []
        for i in range(self.y_true.shape[0]):
            arr = np.where(self.y_true[i]!=0,
                    self.y_true[i]*np.log(self.y_true[i]/self.y_pred[i]), 0)
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
