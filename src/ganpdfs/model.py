import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Reshape
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import BatchNormalization

from ganpdfs.custom import ConvPDF
from ganpdfs.custom import ConvXgrid
from ganpdfs.utils import construct_cnn
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
        # Get PDF grid info
        self.fl_size = pdf.shape[1]
        self.xg_size = pdf.shape[2]
        # Define Activations & Optimizers
        self.g_activ = params.get("g_act")
        self.c_activ = params.get("d_act")
        self.c_optmz = params.get("d_opt")
        self.adv_opmtz = params.get("gan_opt")

    def generator_model(self):
        """generator_model.
        """

        n_nodes = 7 * 7 * 256
        dnn_dim = self.params.get("g_nodes")
        # Generator Input
        g_input = Input(shape=self.noise_size)
        # 1st Layer
        g1l = Dense(n_nodes)(g_input)
        g1a = self.activ.get(self.g_activ)(g1l)
        # 2nd Layer
        g2l = Dense(dnn_dim)(g1a)
        g2b = BatchNormalization()(g2l)
        g2a = self.activ.get(self.g_activ)(g2b)
        # 3rd Layer
        g3l = Dense(dnn_dim * 2)(g2a)
        g3b = BatchNormalization()(g3l)
        g3a = self.activ.get(self.g_activ)(g3b)
        # 4th Layer
        g4l = Dense(self.fl_size * self.xg_size)(g3a)
        # g4r = Reshape((self.fl_size, self.xg_size))(g4l)
        g_output = Reshape((self.fl_size, self.xg_size))(g4l)
        # Output
        # g_output = ConvPDF(self.pdf)(g4r)
        return Model(g_input, g_output, name="Generator")
        
    def critic_model(self):
        """critic_model.
        """

        dnn_dim = self.params["d_nodes"]
        # Weight Constraints
        const = ClipConstraint(1)
        # Discriminator Input
        d_input = Input(shape=(self.fl_size, self.xg_size))
        # 1st Layer
        d1l = Dense(dnn_dim, kernel_constraint=const)(d_input)
        d1b = BatchNormalization()(d1l)
        d1a = self.activ.get(self.c_activ)(d1b)
        # 2nd Layer
        d2l = Dense(dnn_dim // 2, kernel_constraint=const)(d1a)
        d2b = BatchNormalization()(d2l)
        d2a = self.activ.get(self.c_activ)(d2b)
        # Flatten and Output Logit
        d3l = Flatten()(d2a)
        d_output = Dense(1)(d3l)
        # Compile Critic Model
        critic_opt = self.optmz.get(self.c_optmz)
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
        adv_opt = self.optmz.get(self.adv_opmtz)
        model.compile(loss=wasserstein_loss, optimizer=adv_opt)
        return model


class DCNNWassersteinGanModel:
    """DCNNWassersteinGanModel.
    """

    def __init__(self, pdf, params, noise_size, activ, optmz):
        self.pdf = pdf
        self.activ = activ
        self.optmz = optmz
        self.params = params
        self.noise_size = noise_size
        # Get PDF grid info
        self.fl_size = pdf.shape[1]
        self.xg_size = pdf.shape[2]
        # Define Activations & Optimizers
        self.g_activ = params.get("g_act")
        self.c_activ = params.get("d_act")
        self.c_optmz = params.get("d_opt")
        self.adv_opmtz = params.get("gan_opt")
        # Compute DCNN structure
        self.cnnf = construct_cnn(pdf.shape[1], 3)
        self.cnnx = construct_cnn(pdf.shape[2], 3)

    def generator_model(self):
        """generator_model.
        """

        gcnn = self.params.get("g_nodes")
        n_nodes = self.cnnf[0] * self.cnnx[0] * gcnn
        g_input = Input(shape=(self.noise_size,))
        # 1st DCNN Layer
        G1l = Dense(n_nodes)(g_input)
        G1b = BatchNormalization()(G1l)
        G1a = self.activ.get(self.g_activ)(G1b)
        G1r = Reshape((self.cnnf[0], self.cnnx[0], gcnn))(G1a)
        # 2nd Layer:
        # Upsample to (cnnf[0]*cnnf[1], cnnx[0]*cnnx[1])
        G2l = Conv2DTranspose(
                gcnn // 2,
                kernel_size=(4, 4),
                strides=(self.cnnf[1], self.cnnx[1]),
                padding="same",
        )(G1r)
        G2b = BatchNormalization()(G2l)
        G2a = self.activ.get(self.g_activ)(G2b)
        # 3rd Layer:
        # Upsample to (cnnf[1]*cnnf[2], cnnx[1]*cnnx[2])
        G3l = Conv2DTranspose(
                gcnn // 4,
                kernel_size=(4, 4),
                strides=(self.cnnf[2], self.cnnx[2]),
                padding="same",
        )(G2a)
        G3b = BatchNormalization()(G3l)
        G3a = self.activ.get(self.g_activ)(G3b)
        # 4th Layer:
        # Upsample to (cnnf[2]*cnnf[3], cnnx[2]*cnnx[3])
        # 4th output shape=(None, fl_size, xg_size, 1)
        G4l = Conv2DTranspose(
                1,
                kernel_size=(7, 7),
                padding="same",
                activation="tanh"
        )(G3a)
        # Output Layer
        g_output = ConvPDF(self.pdf)(G4l)
        return Model(g_input, g_output, name="Generator")

    def critic_model(self):
        """critic_model.
        """

        dcnn = self.params.get("d_nodes")
        const = ClipConstraint(1)
        d_input = Input(shape=(self.fl_size, self.xg_size, 1))
        # Downsample to (7, 35)
        D1l = Conv2D(
                dcnn,
                kernel_size=(4, 4),
                strides=(1, 2),
                padding="same",
                kernel_constraint=const,
        )(d_input)
        D1b = BatchNormalization()(D1l)
        D1a = self.activ.get(self.c_activ)(D1b)
        # Downsample to (7, 7)
        D2l = Conv2D(
                dcnn * 2,
                kernel_size=(4, 4),
                strides=(1, 5),
                padding="same",
                kernel_constraint=const,
        )(D1a)
        D2b = BatchNormalization()(D2l)
        D2a = self.activ.get(self.c_activ)(D2b)
        # Flatten and output logits
        D3r = Flatten()(D2a)
        d_output = Dense(1)(D3r)
        # Compile Model
        critic_opt = self.optmz.get(self.c_optmz)
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

        adv_model = Sequential(name="Adversarial")
        # Add Generator
        adv_model.add(generator)
        # Add Critic
        adv_model.add(critic)
        # Compile Adversarial Model
        adv_opt = self.optmz.get(self.adv_opmtz)
        adv_model.compile(loss=wasserstein_loss, optimizer=adv_opt)
        return adv_model
