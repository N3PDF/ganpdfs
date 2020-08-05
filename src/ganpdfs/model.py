import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.initializers import RandomNormal
from keras.layers.advanced_activations import LeakyReLU

from ganpdfs.custom import xlayer
from ganpdfs.custom import disc_loss
from ganpdfs.custom import genr_loss
from ganpdfs.custom import ClipConstraint
from ganpdfs.custom import wasserstein_loss
from ganpdfs.custom import preprocessing_fit


class WassersteinGanModel:
    """WassersteinGanModel.
    """

    def __init__(self, noise_size, output_size, x_grid, params, activ, optmz):
        self.noise_size = noise_size
        self.output_size = output_size
        self.x_grid = x_grid
        self.params = params
        self.activ = activ
        self.optmz = optmz
        self.scan = params["scan"]
        self.g_nodes = params["g_nodes"]
        self.d_nodes = params["d_nodes"]

        # ---------------------------#
        #   CRITIC/DISCRIMINATOR    #
        # ---------------------------#
        crit_optimizer = self.optmz[params["d_opt"]]
        self.critic = self.critic_model()
        self.critic.compile(
            loss=wasserstein_loss,
            optimizer=crit_optimizer,
            metrics=["accuracy"]
        )
        if not self.scan:
            self.critic.summary()

        # ---------------------------#
        #         GENERATOR         #
        # ---------------------------#
        self.generator = self.generator_model()
        if not self.scan:
            self.generator.summary()

        # ---------------------------#
        #     ADVERSARIAL MODEL     #
        # ---------------------------#
        gan_optimizer = self.optmz[params["gan_opt"]]
        self.adversarial = self.adversarial_model()
        self.adversarial.compile(
                loss=wasserstein_loss,
                optimizer=gan_optimizer
        )
        if not self.scan:
            self.adversarial.summary()

    def generator_model(self):
        """
        This constructs the architercture of the Generator.
        """
        
        return tf.keras.Sequential()

    def critic_model(self):
        """
        This construct the architecture of the Critic.
        """

        """
        Input of D/Output of G. This is a 1 dim vector of the
        size of the x-gird.
        """

        return tf.keras.Sequential()

    def adversarial_model(self):
        """
        This method implements the Adversarial Training.
        The Critic is Frozen by default and only will be
        turned on when training the Critic.
        """

        # Set teh critic training to False
        self.critic.trainable = False

        # Adversarial Training Architecture ##
        G_input = Input(shape=(self.noise_size,))
        fake_pdf = self.generator(G_input)
        validity = self.critic(fake_pdf)

        ad_model = Model(G_input, validity)
        return ad_model


class DCNNWassersteinGanModel:
    """DCNNWassersteinGanModel.
    """

    def __init__(self, noise_size, output_size, x_grid, params, activ, optmz):
        self.activ = activ
        self.optmz = optmz
        self.x_grid = x_grid
        self.params = params
        self.noise_size = noise_size
        self.output_size = output_size
        self.scan = params["scan"]

        # ---------------------------#
        #         GENERATOR         #
        # ---------------------------#
        self.generator = self.generator_model()
        if not self.scan:
            self.generator.summary()

        # ---------------------------#
        #          CRITIC           #
        # ---------------------------#
        crit_optimizer = self.optmz[params["d_opt"]]
        self.critic = self.critic_model()
        self.critic.compile(
                loss=disc_loss, optimizer=crit_optimizer
        )
        if not self.scan:
            self.critic.summary()

        # ---------------------------#
        #     ADVERSARIAL MODEL     #
        # ---------------------------#
        gan_optimizer = self.optmz[params["gan_opt"]]
        self.adversarial = self.adversarial_model()
        self.adversarial.compile(
                loss=disc_loss, optimizer=gan_optimizer
        )
        if not self.scan:
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
