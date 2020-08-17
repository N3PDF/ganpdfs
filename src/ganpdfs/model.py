import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.initializers import RandomNormal
from keras.layers.advanced_activations import LeakyReLU

from ganpdfs.custom import ConvPDF
from ganpdfs.custom import ConvXgrid
from ganpdfs.custom import PreprocessFit
from ganpdfs.custom import ClipConstraint
from ganpdfs.custom import wasserstein_loss


class WassersteinGanModel:
    """WassersteinGanModel.
    """

    def __init__(self, pdf, params, noise_size, activ, optmz):
        self.pdf = pdf
        self.activ = activ
        self.optmz = optmz
        self.params = params
        self.noise_size = noise_size

    def generator_model(self):
        """generator_model.
        """

        n_nodes = 7 * 7 * 256
        fl_size = self.pdf.shape[1]
        xg_size = self.pdf.shape[2]
        dnn_dim = self.params["g_nodes"]
        # Generator Input
        g_input = Input(shape=self.noise_size)
        # 1st Layer
        g1l = Dense(n_nodes)(g_input)
        g1a = LeakyReLU(alpha=0.2)(g1l)
        # 2nd Layer
        g2l = Dense(dnn_dim)(g1a)
        g2b = BatchNormalization()(g2l)
        g2a = LeakyReLU(alpha=0.2)(g2b)
        # 3rd Layer
        g3l = Dense(dnn_dim * 2)(g2a)
        g3b = BatchNormalization()(g3l)
        g3a = LeakyReLU(alpha=0.2)(g3b)
        # 4th Layer
        g4l = Dense(fl_size * xg_size)(g3a)
        g4r = Reshape((fl_size, xg_size))(g4l)
        # Output
        g_output = ConvPDF(self.pdf)(g4r)
        return Model(g_input, g_output, name="Generator")
        
    def critic_model(self):
        """critic_model.
        """

        fl_size = self.pdf.shape[1]
        xg_size = self.pdf.shape[2]
        dnn_dim = self.params["d_nodes"]
        # Weight Constraints
        const = ClipConstraint(1)
        # Discriminator Input
        d_input = Input(shape=(fl_size, xg_size))
        # 1st Layer
        d1l = Dense(dnn_dim, kernel_constraint=const)(d_input)
        d1b = BatchNormalization()(d1l)
        d1a = LeakyReLU(alpha=0.2)(d1b)
        # 2nd Layer
        d2l = Dense(dnn_dim // 2, kernel_constraint=const)(d1a)
        d2b = BatchNormalization()(d2l)
        d2a = LeakyReLU(alpha=0.2)(d2b)
        # Flatten and Output Logit
        d3l = Flatten()(d2a)
        d_output = Dense(1)(d3l)
        # Compile Critic Model
        critic_opt = self.optmz[self.params["d_opt"]]
        cr_model = Model(d_input, d_output, name="Critic")
        cr_model.compile(loss=wasserstein_loss, optimizer=critic_opt)
        return cr_model

    def adversarial_model(self, generator, critic):
        """adversarial_model.

        Parameters
        ----------
        generator :
            generator
        critic :
            critic
        """

        model = Sequential(name="Adversarial")
        # Add Generator Model
        model.add(generator)
        # Add Critic Model
        model.add(critic)
        # Compile Adversarial Model
        adv_opt = self.optmz[self.params["gan_opt"]]
        model.compile(loss=wasserstein_loss, optimizer=adv_opt)
        return model


class DCNNWassersteinGanModel:
    """DCNNWassersteinGanModel.
    """

    def __init__(self, noise_size, x_grid, params, activ, optmz):
        self.activ = activ
        self.optmz = optmz
        self.x_grid = x_grid
        self.params = params
        self.noise_size = noise_size

        # Generator
        self.generator = self.generator_model()
        if not params["scan"]:
            self.generator.summary()

        # Critic/Discriminator
        self.critic = self.critic_model()
        self.critic.compile(
                loss=wasserstein_loss,
                optimizer=self.optmz[params["d_opt"]]
        )
        if not params["scan"]:
            self.critic.summary()

        # Adversarial
        self.adversarial = self.adversarial_model()
        self.adversarial.compile(
                loss=wasserstein_loss,
                optimizer=self.optmz[params["gan_opt"]]
        )
        if not params["scan"]:
            self.adversarial.summary()

    def generator_model(self):
        """generator_model.
        """

        G_input = Input(shape=(self.noise_size,))

        G1l = Dense(7 * 7 * 256, use_bias=False)(G_input)
        G1b = BatchNormalization()(G1l)
        G1a = self.activ[self.params["g_act"]](G1b)
        G1r = Reshape((7, 7, 256))(G1a)

        G2l = Conv2DTranspose(
                128,
                kernel_size=(5, 5),
                strides=(1, 1),
                padding="same",
                use_bias=False
        )(G1r)
        G2b = BatchNormalization()(G2l)
        G2a = self.activ[self.params["g_act"]](G2b)

        G3l = Conv2DTranspose(
                64,
                kernel_size=(5, 5),
                strides=(1, 5),
                padding="same",
                use_bias=False
        )(G2a)
        G3b = BatchNormalization()(G3l)
        G3a = self.activ[self.params["g_act"]](G3b)

        G_output = Conv2DTranspose(
                1,
                kernel_size=(5, 5),
                strides=(1, 2),
                padding="same",
                use_bias=False,
                activation="tanh"
        )(G3a)

        return Model(G_input, G_output, name="Generator")

    def critic_model(self):
        """critic_model.
        """

        D_input = Input(shape=(7, 70, 1))
        D1l = Conv2D(
                64,
                kernel_size=(5, 5),
                strides=(1, 5),
                padding="same"
        )(D_input)
        D1a = self.activ[self.params["d_act"]](D1l)
        D1d = Dropout(0.3)(D1a)

        D2l = Conv2D(
                128,
                kernel_size=(5, 5),
                strides=(1, 2)
        )(D1d)
        D2a = self.activ[self.params["d_act"]](D2l)
        D2d = Dropout(0.3)(D2a)

        D3r = Flatten()(D2d)

        D_output = Dense(1)(D3r)
        
        return Model(D_input, D_output, name="Critic")

    def adversarial_model(self):
        """
        This method implements the Adversarial Training.
        The Critic is Frozen by default and only will be
        turned on when training the Critic.
        """

        # Set the critic training to False
        self.critic.trainable = False

        # Adversarial Training Architecture
        G_input = Input(shape=(self.noise_size,))
        fake_pdf = self.generator(G_input)
        validity = self.critic(fake_pdf)

        return Model(G_input, validity, name="Adversarial")
