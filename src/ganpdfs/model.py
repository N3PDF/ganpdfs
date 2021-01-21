from tensorflow.keras import Model
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Reshape
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import BatchNormalization

from ganpdfs.custom import ConvPDF
from ganpdfs.custom import GenDense
from ganpdfs.custom import ConvXgrid
from ganpdfs.utils import do_nothing
from ganpdfs.utils import construct_cnn
from ganpdfs.custom import ClipConstraint
from ganpdfs.custom import wasserstein_loss


class WGanModel:
    """WassersteinGanModel.
    """

    def __init__(self, pdf, params, activ, optmz):
        self.pdf = pdf
        self.activ = activ
        self.optmz = optmz
        self.params = params
        self.g_activ = params.get("g_act")
        self.c_activ = params.get("d_act")
        self.c_optmz = params.get("d_opt")
        self.adv_opmtz = params.get("gan_opt")
        self.d_size = params.get("ddnn_size", 1)
        self.g_size = params.get("gdnn_size", 2)
        _, self.fl_size, self.xg_size = pdf.shape

    def generator_model(self):
        """generator_model.
        """
        # Generator Input
        g_shape = (self.fl_size, self.xg_size)
        g_input = Input(shape=g_shape)
        # Output of Lambda layer has a shape
        # (None, nb_flavors, xgrid_size)
        g_lambd = Lambda(do_nothing)(g_input)
        g_dense = GenDense(self.xg_size, use_bias=False)(g_lambd)
        # g_dense = Dense(
        #     self.xg_size,
        #     use_bias=False,
        #     trainable=True,
        #     kernel_initializer=initializer
        # )(g_lambd)
        return Model(g_input, g_dense)

    # def critic_model(self):
    #     """critic_model.
    #     """

    #     dnn_dim = self.params["d_nodes"]
    #     # Weight Constraints
    #     const = ClipConstraint(1)
    #     # Discriminator Input
    #     d_input = (self.fl_size, self.xg_size)
    #     d_model = Sequential(name="Critic")
    #     # 1st Layer
    #     d_model.add(Dense(dnn_dim, kernel_constraint=const, input_shape=d_input))
    #     d_model.add(BatchNormalization())
    #     d_model.add(self.activ.get(self.c_activ))
    #     # Loop over the number of Layers
    #     # by decreasing the size at each iterations
    #     for it in range(1, self.d_size + 1):
    #         d_model.add(Dense(dnn_dim // (2 ** it), kernel_constraint=const))
    #         d_model.add(BatchNormalization())
    #         d_model.add(self.activ.get(self.c_activ))
    #     # Flatten and Output Logit
    #     d_model.add(Flatten())
    #     d_model.add(Dense(1))
    #     # Compile Critic Model
    #     critic_opt = self.optmz.get(self.c_optmz)
    #     d_model.compile(loss="binary_crossentropy", optimizer=critic_opt)
    #     return d_model

    def critic_model(self):
        """critic_model.
        """
        # Weight Constraints
        const = ClipConstraint(1)
        closs = "binary_crossentropy"
        dnn_dim = self.params["d_nodes"]
        critic_opt = self.optmz.get(self.c_optmz)
        # Discriminator Input
        d_input = Input(shape=(self.fl_size, self.xg_size))
        d_hidden = Dense(
            dnn_dim,
            kernel_constraint=const
        )(d_input)
        d_hidden = BatchNormalization()(d_hidden)
        d_hidden = self.activ.get(self.c_activ)(d_hidden)
        for it in range(1, self.d_size + 1):
            d_hidden = Dense(
                dnn_dim // (2 ** it),
                kernel_constraint=const
            )(d_hidden)
            d_hidden = BatchNormalization()(d_hidden)
            d_hidden = self.activ.get(self.c_activ)(d_hidden)
        d_flatten = Flatten()(d_hidden)
        d_output = Dense(1)(d_flatten)
        d_model = Model(d_input, d_output, name="Critic")
        d_model.compile(loss=closs, optimizer=critic_opt)
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
        advloss = "binary_crossentropy"
        adv_opt = self.optmz.get(self.adv_opmtz)
        model = Sequential(name="Adversarial")
        model.add(generator)  # Add Generator Model
        model.add(critic)     # Add Critic Model
        model.compile(loss=advloss, optimizer=adv_opt)
        return model


class DWGanModel:
    """DCNNWassersteinGanModel.
    """

    def __init__(self, pdf, params, activ, optmz):
        self.pdf = pdf
        self.activ = activ
        self.optmz = optmz
        self.params = params
        self.noise_size = None
        self.g_activ = params.get("g_act")
        self.c_activ = params.get("d_act")
        self.c_optmz = params.get("d_opt")
        self.adv_opmtz = params.get("gan_opt")
        _, self.fl_size, self.xg_size = pdf.shape
        self.d_size = params.get("dcnn_size", 1) + 1
        self.g_size = params.get("gcnn_size", 2) + 1
        self.cnnf = construct_cnn(pdf.shape[1], self.g_size)
        self.cnnx = construct_cnn(pdf.shape[2], self.g_size)

    def generator_model(self):
        """generator_model.
        """
        gcnn = self.params.get("g_nodes")
        n_nodes = self.cnnf[0] * self.cnnx[0] * gcnn
        g_model = Sequential(name="Generator")
        # 1st DCNN Layer
        g_model.add(Dense(n_nodes, input_dim=self.noise_size))
        g_model.add(BatchNormalization())
        g_model.add(self.activ.get(self.g_activ))
        g_model.add(Reshape((self.cnnf[0], self.cnnx[0], gcnn)))
        # Loop over the number of hidden layers and
        # upSample at every iteration.
        for it in range(1, self.g_size):
            g_model.add(
                Conv2DTranspose(
                    gcnn // (2 ** it),
                    kernel_size=(4, 4),
                    strides=(self.cnnf[it], self.cnnx[it]),
                    padding="same",
                )
            )
            g_model.add(BatchNormalization())
            g_model.add(self.activ.get(self.g_activ))
        # 4th Layer:
        # Upsample to (cnnf[2]*cnnf[3], cnnx[2]*cnnx[3])
        # 4th output shape=(None, fl_size, xg_size, 1)
        g_model.add(
            Conv2DTranspose(
                1,
                kernel_size=(7, 7),
                padding="same",
                activation="tanh"
            )
        )
        # Convolute input PDF
        if self.params.get("ConvoluteOutput", True):
            g_model.add(ConvPDF(self.pdf))
        return g_model

    def critic_model(self):
        """critic_model.
        """
        dcnn = self.params.get("d_nodes")
        const = ClipConstraint(1)
        d_input = (self.fl_size, self.xg_size, 1)
        d_model = Sequential(name="Discriminator/Critic")
        # Downsample to (7, 35)
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
        d_model.add(BatchNormalization())
        d_model.add(self.activ.get(self.c_activ))
        # Loop over the number of Layers
        # by Downsampling at each iteration
        for it in range(1, self.d_size):
            d_model.add(
                Conv2D(
                    dcnn * (2 ** it),
                    kernel_size=(4, 4),
                    strides=(1, 1),
                    padding="same",
                    kernel_constraint=const,
                )
            )
            d_model.add(BatchNormalization())
            d_model.add(self.activ.get(self.c_activ))
        # Flatten and output logits
        d_model.add(Flatten())
        d_model.add(Dense(1))
        # Compile Model
        critic_opt = self.optmz.get(self.c_optmz)
        d_model.compile(loss=wasserstein_loss, optimizer=critic_opt)
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
        adv_model = Sequential(name="Adversarial")
        adv_opt = self.optmz.get(self.adv_opmtz)
        adv_model.add(generator)    # Add Generator
        adv_model.add(critic)       # Add Critic
        adv_model.compile(loss=wasserstein_loss, optimizer=adv_opt)
        return adv_model
