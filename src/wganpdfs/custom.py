import os
import numpy as np
# Silent tf for the time being
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from scipy import stats
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
    return tf.reduce_mean(y_true * y_pred)


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

    # @tf.function
    def wasserstein_loss(self):
        return tf.reduce_mean(y_true*y_pred)


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


class estimators(object):
    """
    Class that contains methods for the estimator computation
    """
    def __init__(self, true_distr, fake_distr, Axs=None):
        self.Axs = Axs
        self.eps = 1e-8
        self.true_distr = true_distr
        self.fake_distr = fake_distr

    def mean(self):
        """
        Compute mean
        """
        Tmean = np.mean(self.true_distr,axis=self.Axs) + self.eps
        Fmean = np.mean(self.fake_distr,axis=self.Axs) + self.eps
        return Tmean, Fmean

    def stdev(self):
        """
        Compute standard deviation
        """
        Tstd = np.std(self.true_distr,axis=self.Axs) + self.eps
        Fstd = np.std(self.fake_distr,axis=self.Axs) + self.eps
        return Tstd, Fstd


class normalizationK(object):
    """
    Class that computes the normalization K for a given estimator
    """
    def __init__(self, true_distr, fake_distr, random_param):
        self.true_distr = true_distr
        self.fake_distr = fake_distr
        self.random_param = random_param

    def random_replicas(self, number):
        # Non-redundant choice
        index = np.random.choice(
            self.true_distr.shape[0],
            number,
            replace=False
        )
        return self.true_distr[index]

    def cfd68(self, name_est, rand_true):
        """
        Return arrays satisfying cfd from
        input arrays
        """
        eps = 1e-8
        estm = estimators(self.true_distr, rand_true, Axs=0)
        tr_mean, rd_mean = estm.mean()
        tr_stdv, rd_stdv = estm.stdev()
        # Shift std to avoid std=0
        tr_stdv += eps
        rd_stdv += eps
        # Compute 68% level (this return tuples)
        tr_cfd = stats.norm.interval(
            0.6827,
            loc=tr_mean,
            scale=tr_stdv
        )
        rd_cfd = stats.norm.interval(
            0.6827,
            loc=rd_mean,
            scale=rd_stdv
        )
        res_tr = np.zeros(self.true_distr.shape[1])
        res_rd = np.zeros(self.true_distr.shape[1])
        for z in range(self.true_distr.shape[1]):
            mask_rd = (rd_cfd[0][z]<=rand_true[:,z]) * (rand_true[:,z]<=rd_cfd[1][z])
            mask_tr = (tr_cfd[0][z]<=self.true_distr[:,z]) * (self.true_distr[:,z]<=tr_cfd[1][z])
            # Apply selection
            new_rd = rand_true[:,z][mask_rd]
            new_tr = self.true_distr[:,z][mask_tr]

            cfd_class = estimators(
                    new_tr,
                    new_rd
            )
            tr_res, rd_res = get_method(cfd_class, name_est)()
            res_tr[z] = tr_res
            res_rd[z] = rd_res
        fin_tr = res_tr + eps
        fin_rd = res_rd + eps

        return fin_tr, fin_rd

    def Nk_mean(self):
        """
        Normalization factor for mean estimator
        """
        sum2  = 0
        # Select fixed-sized subset from true
        Nsize = 50
        Nrand = 1000
        for r in range(1,Nrand):
            rand_distr = self.random_replicas(Nsize)
            xtr, xrd = self.cfd68('mean', rand_distr)
            sum1 = ((xrd - xtr) / xtr)**2
            sum2 += np.sum(sum1)
        return sum2/Nrand

    def Nk_stdev(self):
        """
        Normalization factor for the std estimator.
        Exactly the same as Nk_mean()
        """
        return self.Nk_mean()

class smm(object):

    """
    Similarity Metric Method (SMM)
    Custom metrics in order to assess the performance of the model.

    Input arguments:
        - y_true: multi-dimensional array of prior/true replicas
        - y_pred: multi-dimensional array of generated replicas
        - params: input runcard that contaains a list of estimators
    """

    def __init__(self, y_true, y_pred, params):
        epsilon = 0                 # This has to be ZERO
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
        sum2 = 0
        # Loop over the list of estimators
        for es in self.estmtr:
            # Call normalizations
            nk_class = normalizationK(
                self.y_true,
                self.y_pred,
                None
            )
            Nk = get_method(nk_class, 'Nk_'+es)()
            # Call estimators
            es_class = estimators(
                self.y_true,
                self.y_pred,
                Axs=0
            )
            es_true, es_fake = get_method(es_class, es)()
            sum1 = ((es_fake - es_true) / es_true)**2
            sum2 += np.sum(sum1) / Nk
        return sum2/len(self.estmtr)


# TODO do something like
# def ERF(y_true, y_pred, params):
#     sum2 = 0
#     estmtr = params['estimators']
#     # Loop over the list of estimators
#     for es in self.estmtr:
#         # Call normalizations
#         nk_class = normalizationK(
#             self.y_true,
#             self.y_pred,
#             None
#         )
#         Nk = get_method(nk_class, 'Nk_'+es)()
#         # Call estimators
#         es_class = estimators(
#             self.y_true,
#             self.y_pred,
#             Axs=0
#         )
#         es_true, es_fake = get_method(es_class, es)()
#         sum1 = ((es_fake - es_true) / es_true)**2
#         sum2 += np.sum(sum1) / Nk
#     return sum2/len(self.estmtr)
# 
