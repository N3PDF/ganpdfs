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

from ganpdfs.utils import do_nothing
from ganpdfs.utils import construct_cnn
from ganpdfs.custom import ExpLatent
from ganpdfs.custom import GenDense
from ganpdfs.custom import ConvolutePDF
from ganpdfs.custom import ExtraDense
from ganpdfs.custom import get_optimizer
from ganpdfs.custom import wasserstein_loss
from ganpdfs.custom import get_activation
from ganpdfs.custom import WeightsClipConstraint


class WGanModel:
    """GAN class that represents the models in terms of Deep Neural
    Networks. It includes 3 Models: the Genator, the Discriminator,
    and the Adversarial Model which controls the training of the
    Generator. Apart from the Adversarial Model, the Generator and
    the Discriminator are not compiled directly.

    Parameters
    ----------
    pdf: np.array(float)
        Input PDF grid
    params: dict
        dictionary containing the input parameters

    """

    def __init__(self, pdf, params):
        self.pdf = pdf
        self.latentvec = params.get("latent_space")
        self.ganparams = params.get("gan_parameters")
        self.discparams = params.get("disc_parameters")
        self.genparams = params.get("gen_parameters")
        _, self.fl_size, self.xg_size = pdf.shape

    def generator_model(self):
        """Generator Model based on a Deep Neural Network. It takes
        as input a latent space vector and gives as output a PDF grid
        sampled on a discrete x-grid.
        """

        gnn_size = self.genparams.get("size_networks")
        noise_dim = self.genparams.get("noise_dim", 100)
        gnn_dim = self.genparams.get("number_nodes")
        gs_activ = get_activation(self.genparams)
        structure = self.genparams.get("structure", "custom")

        # Loop over the Architecture
        if structure == "standard":
            g_input = Input(shape=(noise_dim,))
            g_dense = Lambda(do_nothing)(g_input)
            for it in range(1, gnn_size + 1):
                g_dense = ExtraDense(
                    gnn_dim * (2 ** it),
                    self.genparams
                )(g_dense)
                g_dense = BatchNormalization()(g_dense)
                g_dense = gs_activ(g_dense)
            g_dense = ExtraDense(
                    self.fl_size * self.xg_size,
                    self.genparams
            )(g_dense)
            g_prefl = Reshape((self.fl_size, self.xg_size))(g_dense)
            g_outpt = ConvolutePDF(self.pdf)(g_prefl)
            g_model = Model(g_input, g_outpt, name="Generator")
        elif structure == "custom":
            g_shape = (self.fl_size, self.xg_size,)
            g_input = Input(shape=g_shape)
            g_lambd = Lambda(do_nothing)(g_input)
            g_dense = ExpLatent(self.xg_size, use_bias=False)(g_lambd)
            for it in range(1, gnn_size + 1):
                g_dense = GenDense(self.xg_size, self.genparams)(g_dense)
            g_model = Model(g_input, g_dense, name="Generator")
        assert g_model.output_shape == (None, self.fl_size, self.xg_size)
        return g_model

    def critic_model(self):
        """Discriminator/Critic model based on Deep Neural Networks. It
        takes as input a PDF grid (which could be the real or the synthetic).
        It gives as output logit values.
        """

        dnn_dim = self.discparams.get("number_nodes")
        dnn_size = self.discparams.get("size_networks")
        ds_activ = get_activation(self.discparams)

        # Loop over the Architecture
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
        return d_model

    def adversarial_model(self, generator, critic):
        """Adversarial model based on Deep Neural Networks. It controls the
        training of the Generator through the output of the Discriminator.

        Parameters
        ----------
        generator : tf.Model
            generator model
        critic : tf.model
            critic model
        """

        # extract loss & optimizer
        inputloss = self.ganparams.get("loss", "wasserstein")
        if inputloss != "wasserstein":
            advloss = self.ganparams.get("loss")
        else: advloss = wasserstein_loss
        opt_name = self.ganparams.get("optimizer")
        adv_optimizer = get_optimizer(opt_name)

        # freeze Discriminator
        for layer in critic.layers:
            layer.trainable = False

        model = Sequential(name="Adversarial")
        model.add(generator)       # Add Generator Model
        model.add(critic)          # Add Discriminator Model
        model.compile(loss=advloss, optimizer=adv_optimizer)
        return model


class DWGanModel:
    """GAN class that represents the models in terms of Deep Convolu-
    tion Neural Networks. It includes 3 Models: the Generator, the
    Discriminator, and the Adversarial Model which controls the training
    of the Generator. Apart from the Adversarial Model, the Generator
    and the Discriminator are not compiled directly.

    Parameters
    ----------
    pdf: np.array(float)
        Input PDF grid
    params: dict
        dictionary containing the input parameters
    noise_dim: np.array(float)
        vector of random noise
    """

    def __init__(self, pdf, params, noise_dim):
        self.pdf = pdf
        self.params = params
        self.noise_size = noise_dim
        _, self.fl_size, self.xg_size, _ = pdf.shape
        self.ganparams = params.get("gan_parameters")
        self.discparams = params.get("disc_parameters")
        self.genparams = params.get("gen_parameters")

        # Compute DCNN structure
        self.gnn_size = self.genparams.get("size_networks") + 1
        self.dnn_size = self.discparams.get("size_networks") + 1
        self.cnnf = construct_cnn(pdf.shape[1], self.gnn_size)
        self.cnnx = construct_cnn(pdf.shape[2], self.gnn_size)

    def generator_model(self):
        """Generator Model based on Deep Concolutional Neural Networks. It
        takes as input a random noise vector and gives as output a PDF grid
        sampled on a discrete x-grid.
        """

        gnn_dim = self.genparams.get("number_nodes")
        n_nodes = self.cnnf[0] * self.cnnx[0] * gnn_dim
        gs_activ = get_activation(self.genparams)

        g_model = Sequential(name="Generator")
        g_model.add(Dense(n_nodes, input_dim=self.noise_size))
        if self.discparams.get("batchnorm", True):
            g_model.add(BatchNormalization())
        g_model.add(gs_activ)
        g_model.add(Reshape((self.cnnf[0], self.cnnx[0], gnn_dim)))
        # Loop over the number of hidden layers and
        # upSample at every iteration.
        for it in range(1, self.gnn_size):
            g_model.add(
                Conv2DTranspose(
                    gnn_dim // (2 ** it),
                    kernel_size=(4, 4),
                    strides=(self.cnnf[it], self.cnnx[it]),
                    padding="same"
                )
            )
            if self.discparams.get("batchnorm", True):
                g_model.add(BatchNormalization())
            g_model.add(gs_activ)
        g_model.add(
            Conv2DTranspose(
                1,
                kernel_size=(7, 7),
                padding="same"
            )
        )
        return g_model

    def critic_model(self):
        """Discriminator/Critic model based on Deep Convolutional Neural Networks.
        It takes as input a PDF grid (which could be the real or the synthetic).
        It gives as output logit values.
        """

        const = WeightsClipConstraint(0.01)
        dcnn = self.discparams.get("number_nodes")
        ds_activ = get_activation(self.discparams)

        d_input = (self.fl_size, self.xg_size, 1)
        d_model = Sequential(name="Discriminator")
        d_model.add(
            Conv2D(
                dcnn,
                kernel_size=(4, 4),
                strides=(1, 2),
                padding="same",
                kernel_constraint=const,
                input_shape=d_input
            )
        )
        if self.discparams.get("batchnorm", False):
            d_model.add(BatchNormalization())
        d_model.add(ds_activ)
        # Loop over the number of Layers
        # by Downsampling at each iteration
        for it in range(1, self.dnn_size):
            d_model.add(
                Conv2D(
                    dcnn * (2 ** it),
                    kernel_size=(4, 4),
                    strides=(1, 1),
                    padding="same",
                    kernel_constraint=const,
                )
            )
            if self.discparams.get("batchnorm", False):
                d_model.add(BatchNormalization())
            d_model.add(ds_activ)
        # Flatten and output logits
        d_model.add(Flatten())
        d_model.add(Dense(1))
        return d_model

    def adversarial_model(self, generator, critic):
        """Adversarial model based on Deep Convolutional Neural Networks. It
        controls the training of the Generator through the Discriminator.

        Parameters
        ----------
        generator : tf.Model
            generator model
        critic : tf.model
            critic model
        """

        # Call parameters
        if self.ganparams.get("loss") != "wasserstein":
            advloss = self.ganparams.get("loss")
        else: advloss = wasserstein_loss
        opt_name = self.ganparams.get("optimizer")
        adv_optimizer = get_optimizer(opt_name)

        # freeze Discriminator
        for layer in critic.layers:
            layer.trainable = False

        model = Sequential(name="Adversarial")
        model.add(generator)       # Add Generator Model
        model.add(critic)          # Add Discriminator Model
        model.compile(loss=advloss, optimizer=adv_optimizer)
        return model
