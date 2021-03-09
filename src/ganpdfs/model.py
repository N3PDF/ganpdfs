from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Reshape
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.constraints import max_norm

from ganpdfs.utils import construct_cnn
from ganpdfs.utils import do_nothing
from ganpdfs.custom import ConvPDF
from ganpdfs.custom import ExpLatent
from ganpdfs.custom import GenDense
from ganpdfs.custom import ExtraDense
from ganpdfs.custom import get_optimizer
from ganpdfs.custom import get_activation
from ganpdfs.custom import wasserstein_loss


class WGanModel:
    """WassersteinGanModel.
    """

    def __init__(self, pdf, params):
        self.pdf = pdf
        self.ganparams = params.get("gan_parameters")
        self.discparams = params.get("disc_parameters")
        self.genparams = params.get("gen_parameters")
        _, self.fl_size, self.xg_size = pdf.shape

    def generator_model(self):
        """generator_model.
        """

        # Parameter calls
        gnn_size = self.genparams.get("size_networks")

        # Generator Architecture
        g_shape = (self.fl_size, self.xg_size,)
        g_input = Input(shape=g_shape)
        g_lambd = Lambda(do_nothing)(g_input)
        g_dense = ExpLatent(self.xg_size, use_bias=False)(g_lambd)
        if self.genparams.get("custom_hidden", False):
            gnn_dim = self.genparams.get("number_nodes")
            gs_activ = get_activation(self.genparams)
            for it in range(1, gnn_size + 1):
                g_dense = GenDense(
                    gnn_dim * (2 ** it),
                    self.discparams
                )(g_dense)
                g_dense = BatchNormalization()(g_dense)
                g_dense = gs_activ(g_dense)
            g_dense = ExtraDense(self.xg_size, self.genparams)(g_dense)
        else:
            for it in range(1, gnn_size + 1):
                g_dense = GenDense(self.xg_size, self.genparams)(g_dense)
        g_model = Model(g_input, g_dense, name="Generator")
        assert g_model.output_shape == (None, self.fl_size, self.xg_size)
        return g_model

    def critic_model(self):
        """critic_model.
        """

        # Parameter calls
        inputloss = self.discparams.get("loss", "wasserstein")
        if inputloss != "wasserstein":
            dloss = self.discparams.get("loss")
        else: dloss = wasserstein_loss
        dnn_dim = self.discparams.get("number_nodes")
        opt_name = self.discparams.get("optimizer")
        dnn_size = self.discparams.get("size_networks")
        ds_activ = get_activation(self.discparams)
        ds_optmz = get_optimizer(opt_name)

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
        d_model.compile(loss=dloss, optimizer=ds_optmz)
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
        inputloss = self.ganparams.get("loss", "wasserstein")
        if inputloss != "wasserstein":
            advloss = self.ganparams.get("loss")
        else: advloss = wasserstein_loss
        opt_name = self.ganparams.get("optimizer")
        adv_optimizer = get_optimizer(opt_name)
        model = Sequential(name="Adversarial")
        model.add(generator)       # Add Generator Model
        model.add(critic)          # Add Discriminator Model
        model.compile(loss=advloss, optimizer=adv_optimizer)
        return model


class DWGanModel:

    def __init__(self, pdf, params):
        self.pdf = pdf
        self.params = params
        self.ganparams = params.get("gan_parameters")
        self.discparams = params.get("disc_parameters")
        self.genparams = params.get("gen_parameters")
        _, self.fl_size, self.xg_size, _ = pdf.shape

        # Compute DCNN structure
        self.gnn_size = self.genparams.get("size_networks") + 1
        self.dnn_size = self.discparams.get("size_networks") + 1
        self.cnnf = construct_cnn(pdf.shape[1], self.gnn_size)
        self.cnnx = construct_cnn(pdf.shape[2], self.gnn_size)

    def generator_model(self):
        """generator_model.
        """

        gnn_dim = self.genparams.get("number_nodes")
        n_nodes = self.cnnf[0] * self.cnnx[0] * gnn_dim
        gs_activ = get_activation(self.genparams)

        g_shape = (self.fl_size, self.xg_size,)
        g_input = Input(shape=g_shape)
        g_flaten = Flatten()(g_input)
        g_hidden = Dense(n_nodes)(g_flaten)
        g_hidden = BatchNormalization()(g_hidden)
        g_hidden = gs_activ(g_hidden)
        g_hidden = Reshape((self.cnnf[0], self.cnnx[0], gnn_dim))(g_hidden)
        # upSample at every iteration.
        for it in range(1, self.gnn_size):
            g_hidden = Conv2DTranspose(
                    gnn_dim // (2 ** it),
                    kernel_size=(4, 4),
                    strides=(self.cnnf[it], self.cnnx[it]),
                    padding="same"
                )(g_hidden)
            g_hidden = BatchNormalization()(g_hidden)
            g_hidden = gs_activ(g_hidden)
        # Upsample to (cnnf[2]*cnnf[3], cnnx[2]*cnnx[3])
        # Output shape=(None, fl_size, xg_size, 1)
        g_hidden = Conv2DTranspose(
                1,
                kernel_size=(7, 7),
                padding="same",
                activation="tanh"
            )(g_hidden)
        # Convolute input PDF
        if self.params.get("ConvoluteOutput", False):
            g_hidden = ConvPDF(self.pdf)(g_hidden)
        return Model(g_input, g_hidden, name="Generator")

    def critic_model(self):
        """critic_model.
        """

        if self.discparams.get("loss") != "wasserstein":
            dloss = self.discparams.get("loss")
        dcnn = self.discparams.get("number_nodes")
        ds_activ = get_activation(self.discparams)
        opt_name = self.discparams.get("optimizer")
        ds_optmz = get_optimizer(opt_name)

        d_shape = (self.fl_size, self.xg_size, 1)
        d_input = Input(shape=d_shape)
        d_hidden = Conv2D(
                dcnn,
                kernel_size=(4, 4),
                strides=(1, 2),
                padding="same",
                kernel_constraint=max_norm(1.)
            )(d_input)
        d_hidden = BatchNormalization()(d_hidden)
        d_hidden = ds_activ(d_hidden)
        # Downsampling at each iteration
        for it in range(1, self.dnn_size):
            d_hidden = Conv2D(
                    dcnn * (2 ** it),
                    kernel_size=(4, 4),
                    strides=(1, 1),
                    padding="same",
                    kernel_constraint=max_norm(1.),
                )(d_hidden)
            d_hidden = BatchNormalization()(d_hidden)
            d_hidden = ds_activ(d_hidden)
        # Flatten and output logits
        d_hidden = Flatten()(d_hidden)
        d_output = Dense(1)(d_hidden)

        d_model = Model(d_input, d_output, name="Discriminator")
        d_model.compile(loss=dloss, optimizer=ds_optmz)
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
        if self.ganparams.get("loss") != "wasserstein":
            advloss = self.ganparams.get("loss")
        else: advloss = wasserstein_loss
        opt_name = self.ganparams.get("optimizer")
        adv_optimizer = get_optimizer(opt_name)
        model = Sequential(name="Adversarial")
        model.add(generator)       # Add Generator Model
        model.add(critic)          # Add Discriminator Model
        model.compile(loss=advloss, optimizer=adv_optimizer)
        return model
