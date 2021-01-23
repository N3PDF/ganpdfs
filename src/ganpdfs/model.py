from tensorflow.keras import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Reshape
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import BatchNormalization

from ganpdfs.utils import do_nothing
from ganpdfs.custom import ExpLatent
from ganpdfs.custom import ExtraDense
from ganpdfs.custom import get_optimizer
from ganpdfs.custom import get_activation
from ganpdfs.custom import wasserstein_loss


class WGanModel:
    """WassersteinGanModel.
    """

    def __init__(self, pdf, params):
        self.pdf = pdf
        self.params = params
        self.ganparams = params.get("gan_parameters")
        self.discparams = params.get("disc_parameters")
        self.genparams = params.get("gen_parameters")
        _, self.fl_size, self.xg_size = pdf.shape

    def generator_model(self):
        """generator_model.
        """

        # Parameter calls
        g_shape = (self.fl_size, self.xg_size,)
        g_input = Input(shape=g_shape)
        # gnn_dim = self.genparams.get("number_nodes")
        # gnn_size = self.genparams.get("size_networks")
        # gs_activ = get_activation(self.discparams)
        # Output of Lambda layer has a shape
        # (None, nb_flavors, xgrid_size)
        g_lambd = Lambda(do_nothing)(g_input)
        g_dense = ExpLatent(self.xg_size, use_bias=False)(g_lambd)
        # for it in range(1, gnn_size + 1):
        #     g_dense = ExtraDense(
        #         gnn_dim // (2 ** it),
        #         self.discparams
        #     )(g_dense)
        #     g_dense = gs_activ(g_dense)
        return Model(g_input, g_dense, name="Generator")

    def critic_model(self):
        """critic_model.
        """

        # Parameter calls
        if self.discparams.get("loss") != "wasserstein":
            dloss = self.discparams.get("loss")
        else: dloss = wasserstein_loss
        dnn_dim = self.discparams.get("number_nodes")
        opt_name = self.discparams.get("optimizer")
        dnn_size = self.discparams.get("size_networks")
        ds_activ = get_activation(self.discparams)
        critic_opt = get_optimizer(opt_name)

        # Discriminator Architecture
        d_shape = (self.fl_size, self.xg_size,)
        d_input = Input(shape=d_shape)
        d_hidden = ExtraDense(dnn_dim, self.discparams)(d_input)
        d_hidden = BatchNormalization()(d_hidden)
        d_hidden = ds_activ(d_hidden)
        for it in range(1, dnn_size + 1):
            d_hidden = ExtraDense(
                dnn_dim // (2 ** it),
                self.discparams
            )(d_hidden)
            d_hidden = BatchNormalization()(d_hidden)
            d_hidden = ds_activ(d_hidden)
        d_flatten = Flatten()(d_hidden)
        d_output = ExtraDense(1, self.discparams)(d_flatten)
        d_model = Model(d_input, d_output, name="Discriminator")
        d_model.compile(loss=dloss, optimizer=critic_opt)
        return d_model

    def adversarial_model(self, generator, critic):
        """adversarial_model.

        Parameters
        ----------
        generator :
            generator
        critic :
            critic
        """

        # Call parameters
        if self.discparams.get("loss") != "wasserstein":
            advloss = self.discparams.get("loss")
        else: advloss = wasserstein_loss
        opt_name = self.ganparams.get("optimizer")
        adv_optimizer = get_optimizer(opt_name)
        model = Sequential(name="Adversarial")
        model.add(generator)       # Add Generator Model
        model.add(critic)          # Add Discriminator Model
        model.compile(loss=advloss, optimizer=adv_optimizer)
        return model


class DWGanModel:

    def __init__(self, pdf, params, activ):
        pass
