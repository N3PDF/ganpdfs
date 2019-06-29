from keras import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
from keras.layers import Reshape, Conv2D, Flatten, BatchNormalization, LSTM, Activation
from keras.layers.advanced_activations import LeakyReLU, ELU, ReLU
from keras.optimizers import Adam, RMSprop, SGD, Adadelta


class vanilla_xgan_model(object):

    """
    Generative Adversarial Neural Network Model.
    * Discriminator::Learn the features of the true data and
    and check whether or not the generated data ressembles to the true.
    * Generator::Learn to generate fake data by tricking the Discriminator.
    * GAN::Do the adversarial training.
    """

    def __init__(self, noise_size, output_size, x_pdf, params, activ, optmz):
        self.noise_size  = noise_size
        self.output_size = output_size
        self.G   = None
        self.D   = None
        self.G_M = None
        self.D_M = None
        self.GAN_M = None
        self.x_pdf = x_pdf

        self.params  = params
        self.activ   = activ
        self.optmz   = optmz
        self.g_nodes = params['g_nodes']
        self.d_nodes = params['d_nodes']

    def generator(self):

        # Input of G/Random noises
        G_input = Input(shape=(self.noise_size,))

        # 1st hidden dense layer
        G_in = Dense(self.g_nodes//4)(G_input)
        G_in = self.activ[self.params['g_act']](G_in)
        # 2nd hidden custom layer with x-grid
        G_in = xlayer(self.g_nodes//2, self.x_pdf)(G_in)
        G_in = self.activ[self.params['g_act']](G_in)
        # 3rd hidden dense layer
        G_in = Dense(self.g_nodes)(G_in)
        G_in = self.activ[self.params['g_act']](G_in)
        # Output layer
        G_output = Dense(self.output_size, activation='sigmoid')(G_in)

        self.G = Model(G_input, G_output)

        return self.G

    def discriminator(self):

        # Input of D/Output of G
        D_input = Input(shape=(self.output_size,))

        # 1st hidden dense layer
        D_in = Dense(self.d_nodes)(D_input)
        D_in = self.activ[self.params['d_act']](D_in)
        # 2nd hidden dense layer
        D_in = Dense(self.d_nodes//2)(D_in)
        D_in = self.activ[self.params['d_act']](D_in)
        # 3rd hidden dense layer
        D_in = Dense(self.d_nodes//8)(D_in)
        D_in = self.activ[self.params['d_act']](D_in)

        # Output 1 dimensional probability
        D_output = Dense(1, activation='sigmoid')(D_in)

        self.D = Model(D_input, D_output)

        return self.D

    def generator_model(self):

        gen_optimizer = self.optmz[self.params['g_opt']]

        self.G_M = Sequential()
        self.G_M.add(self.generator())
        self.G_M.compile(loss=self.params['g_loss'], optimizer=gen_optimizer,
                        metrics=['accuracy'])
        self.G_M.name = 'Generator'
        self.G.summary()

        return self.G_M

    def discriminator_model(self):

        disc_optimizer = self.optmz[self.params['d_opt']]

        self.D_M = Sequential()
        self.D_M.add(self.discriminator())
        self.D_M.compile(loss=self.params['d_loss'], optimizer=disc_optimizer,
                        metrics=['accuracy'])
        self.D_M.name = 'Discriminator'
        self.D.summary()

        return self.D_M

    def gan_model(self):

        gan_optimizer = self.optmz[self.params['gan_opt']]

        Gan_input  = Input(shape=(self.noise_size,))
        Gan_latent = self.G_M(Gan_input)
        Gan_output = self.D_M(Gan_latent)
        self.GAN_M = Model(Gan_input, Gan_output)

        self.D_M.trainable = False

        self.GAN_M.name = 'GAN'
        self.GAN_M.compile(loss=self.params['gan_loss'], optimizer=gan_optimizer,
                        metrics=['accuracy'])
        self.GAN_M.summary()

        return self.GAN_M


class dc_xgan_model(object):

    """
    Generative Adversarial Neural Network Model.
    * Discriminator::Learn the features of the true data and
    and check whether or not the generated data ressembles to the true.
    * Generator::Learn to generate fake data by tricking the Discriminator.
    * GAN::Do the adversarial training.
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

        disc_optimizer = self.optmz[self.params['d_opt']]
        G_input = Input(shape=(noise_size,))

        # Construct the models
        self.discriminator = self.discriminator_model()
        self.discriminator.compile(loss=self.params['d_loss'], optimizer=disc_optimizer,
                             metrics=['accuracy'])
        self.generator = self.generator_model()

        # Generate fake PDFs
        fake = self.generator(G_input)

        """
        Set Discriminator training to be false.
        Only set to true when the discriminator is trained.
        """
        self.discriminator.trainable = False

        # Discriminator takes the fake PDFs and determines its validity
        validity = self.discriminator(fake)

        # GAN/Combined model
        gan_optimizer = self.optmz[self.params['gan_opt']]
        self.gan = Model(G_input, validity)
        self.gan.compile(loss=params['gan_loss'], optimizer=gan_optimizer)

    def generator_model(self):

        model = Sequential()

        model.add(Dense(self.g_nodes//4, input_dim=self.noise_size))
        model.add(self.activ[self.params['g_act']])
        # model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(self.g_nodes//2))
        model.add(self.activ[self.params['g_act']])
        # model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(self.g_nodes))
        model.add(self.activ[self.params['g_act']])
        # model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(self.g_nodes*2))
        model.add(self.activ[self.params['g_act']])
        model.add(Dense(self.output_size))

        model.summary()

        noise = Input(shape=(self.noise_size,))
        img = model(noise)

        return Model(noise, img)

    def discriminator_model(self):

        model = Sequential()

        # model.add(Flatten(input_shape=self.output_size))
        model.add(Dense(self.d_nodes, input_dim=self.output_size))
        model.add(self.activ[self.params['d_act']])
        model.add(Dense(self.d_nodes//2))
        model.add(self.activ[self.params['g_act']])
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        img = Input(shape=(self.output_size,))
        validity = model(img)

        return Model(img, validity)
