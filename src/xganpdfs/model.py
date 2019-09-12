import tensorflow as  tf
from keras import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
from xganpdfs.custom import xlayer, xmetrics, preprocessing_fit
from keras.layers import Reshape, Conv2D, Flatten, BatchNormalization, LSTM, Activation
from keras.layers.advanced_activations import LeakyReLU, ELU, ReLU
from keras.optimizers import Adam, RMSprop, SGD, Adadelta


class vanilla_xgan_model(object):

    """
    Generative Adversarial Neural Network Model.
    * Discriminator::Learn the features of the true data and
      and check whether or not the generated data ressembles to the true.
    * Generator::Learn to generate fake data by tricking the Discriminator.
    * GAN::Manage the adversarial training.
    """

    def __init__(self, noise_size, output_size, x_pdf, params, activ, optmz):
        self.noise_size  = noise_size
        self.output_size = output_size
        self.x_pdf = x_pdf
        self.params  = params
        self.activ   = activ
        self.optmz   = optmz
        self.g_nodes = params['g_nodes']
        self.d_nodes = params['d_nodes']

        #---------------------------#
        #       DISCRIMINATOR       #
        #---------------------------#
        disc_optimizer = self.optmz[params['d_opt']]
        self.discriminator = self.discriminator_model()
        self.discriminator.compile(loss=params['d_loss'],
                                   optimizer=disc_optimizer,
                                   metrics=['accuracy'])
        self.discriminator.name = 'Discriminator'
        self.discriminator.summary()

        #---------------------------#
        #         GENERATOR         #
        #---------------------------#
        G_input = Input(shape=(noise_size,))
        self.generator = self.generator_model()
        self.generator.name = 'Generator'
        self.generator.summary()

        #---------------------------#
        #     ADVERSARIAL MODEL     #
        #---------------------------#
        """
        Set Discriminator training to be fasle by default.
        Only set to true when the discriminator is trained.
        """
        self.discriminator.trainable = False
        fake_pdf = self.generator(G_input)
        validity = self.discriminator(fake_pdf)
        gan_optimizer = self.optmz[params['gan_opt']]

        self.gan = Model(G_input, validity)
        self.gan.compile(loss=params['gan_loss'],
                         optimizer=gan_optimizer,
                         metrics=['accuracy'])
        self.gan.name = 'GAN'
        self.gan.summary()

    def generator_model(self):
        """
        This construct the architercture
        of the Generator. 
        """
        # Input of G/Random noise of 1 dim vector
        G_input = Input(shape=(self.noise_size,))

        # 1st hidden dense layer
        G_1l = Dense(self.g_nodes//4)(G_input)
        G_1a = self.activ[self.params['g_act']](G_1l)
        # 2nd hidden custom layer with x-grid
        G_2l = xlayer(self.g_nodes//2, self.x_pdf)(G_1a)
        G_2a = self.activ[self.params['g_act']](G_2l)
        # 3rd hidden dense layer
        G_3l = Dense(self.g_nodes)(G_2a)
        G_3a = self.activ[self.params['g_act']](G_3l)
        # 4th hidden dense layer
        G_4l = Dense(self.g_nodes*2)(G_3a)
        G_4a = self.activ[self.params['g_act']](G_4l)
        # Compute the actual PDF
        G_5l = Dense(self.output_size, activation='sigmoid')(G_4a)
        # Output layer with preprocessing
        G_output = preprocessing_fit(self.x_pdf)(G_5l)

        return Model(G_input, G_output)

    def discriminator_model(self):
        """
        This construct the architecture
        of the Discriminator.
        """

        """
        Input of D/Output of G.
        This is a 1 dim vector of the
        size of the x-gird.
        """
        D_input = Input(shape=(self.output_size,))

        # 1st hidden dense layer
        D_1l = Dense(self.d_nodes)(D_input)
        D_1a = self.activ[self.params['d_act']](D_1l)
        # 2nd hidden dense layer
        D_2l = Dense(self.d_nodes//2)(D_1a)
        D_2a = self.activ[self.params['d_act']](D_2l)
        # # 3rd hidden dense layer
        # D_in = Dense(self.d_nodes//8)(D_in)
        # D_in = self.activ[self.params['d_act']](D_in)

        # Output 1 dimensional probability
        D_output = Dense(1, activation='sigmoid')(D_2a)

        return Model(D_input, D_output)
