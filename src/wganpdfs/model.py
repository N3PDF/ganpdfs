import tensorflow as  tf
from keras import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
from wganpdfs.custom import ClipConstraint, wasserstein_loss
from wganpdfs.custom import xlayer, xmetrics, preprocessing_fit
from keras.layers import Reshape, Conv2D, Flatten, BatchNormalization, LSTM, Activation
from keras.layers.advanced_activations import LeakyReLU, ELU, ReLU
from keras.optimizers import Adam, RMSprop, SGD, Adadelta
from keras.initializers import RandomNormal


class wasserstein_xgan_model(object):

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
        self.x_pdf   = x_pdf
        self.params  = params
        self.activ   = activ
        self.optmz   = optmz
        self.g_nodes = params['g_nodes']
        self.d_nodes = params['d_nodes']

        #---------------------------#
        #          CRITIC           #
        #---------------------------#
        crit_optimizer = self.optmz[params['d_opt']]
        self.critic    = self.critic_model()
        self.critic.compile(loss=wasserstein_loss, optimizer=crit_optimizer)
        self.critic.name = 'Critic_Architecture'
        self.critic.summary()

        #---------------------------#
        #         GENERATOR         #
        #---------------------------#
        self.generator = self.generator_model()
        self.generator.name = 'Generator_Architecture'
        self.generator.summary()

        #---------------------------#
        #     ADVERSARIAL MODEL     #
        #---------------------------#
        gan_optimizer = self.optmz[params['gan_opt']]
        self.adversarial = self.adversarial_model()
        self.adversarial.compile(loss=wasserstein_loss, optimizer=gan_optimizer)
        self.adversarial.name = 'Adversarial_Architecture'
        self.adversarial.summary()

    def generator_model(self):
        """
        This construct the architercture of the Generator. 
        """

        # Weights initialization
        init = RandomNormal(stddev=0.02)

        ## Generator Architecture ##
        # Input of G/Random noise of 1 dim vector
        G_input = Input(shape=(self.noise_size,))

        # 1st hidden dense layer
        G_1l = Dense(self.g_nodes//4, kernel_initializer=init)(G_input)
        # G_1b = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(G_1l)
        G_1a = self.activ[self.params['g_act']](G_1l)
        # 2nd hidden custom layer with x-grid
        G_2l = xlayer(self.g_nodes//2, self.x_pdf, kernel_initializer=init)(G_1a)
        # G_2l = Dense(self.g_nodes//2, kernel_initializer=init)(G_1a)
        # G_2b = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(G_2l)
        G_2a = self.activ[self.params['g_act']](G_2l)
        # 3rd hidden dense layer
        G_3l = Dense(self.g_nodes, kernel_initializer=init)(G_2a)
        # G_3b = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(G_3l)
        G_3a = self.activ[self.params['g_act']](G_3l)
        # 4th hidden dense layer
        G_4l = Dense(self.g_nodes*2, kernel_initializer=init)(G_3a)
        # G_4b = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(G_4l)
        G_4a = self.activ[self.params['g_act']](G_4l)
        # Compute the actual PDF
        # G_output = Dense(self.output_size, activation='tanh', kernel_initializer=init)(G_4a)
        G_5l = Dense(self.output_size, activation='tanh', kernel_initializer=init)(G_4a)
        # Output layer with preprocessing
        # G_output = preprocessing_fit(self.x_pdf)(G_5l)

        return Model(G_input, G_5l)

    def critic_model(self):
        """
        This construct the architecture of the Critic.
        """

        """
        Input of D/Output of G. This is a 1 dim vector of the
        size of the x-gird.
        """

        # Weights initialization
        init  = RandomNormal(stddev=0.02)
        # Weights constraint
        const = ClipConstraint(0.01)

        ## Critic Architecture ##
        D_input = Input(shape=(self.output_size,))

        # 1st hidden dense layer
        D_1l = Dense(self.d_nodes, kernel_initializer=init, kernel_constraint=const)(D_input)
        # D_1b = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(D_1l)
        D_1a = self.activ[self.params['d_act']](D_1l)
        # 2nd hidden dense layer
        D_2l = Dense(self.d_nodes//2, kernel_initializer=init, kernel_constraint=const)(D_1a)
        # D_2b = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(D_2l)
        D_2a = self.activ[self.params['d_act']](D_2l)
        # # 3rd hidden dense layer
        # D_in = Dense(self.d_nodes//8, kernel_initializer=init, kernel_constraint=const)(D_in)
        # D_in = self.activ[self.params['d_act']](D_in)

        # Output 1 dimensional probability
        D_output = Dense(1, activation='linear')(D_2a)

        return Model(D_input, D_output)

    def adversarial_model(self):
        """
        This method implements the Adversarial Training.
        The Critic is Frozen by default and only will be
        turned on when training the Critic.
        """

        # Set teh critic training to False
        self.critic.trainable = False

        ## Adversarial Training Architecture ##
        G_input  = Input(shape=(self.noise_size,))
        fake_pdf = self.generator(G_input)
        validity = self.critic(fake_pdf)

        return Model(G_input, validity)
