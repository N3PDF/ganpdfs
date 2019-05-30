import os, sys
import keras.backend as K
from keras import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
from keras.layers import Reshape, Conv2D, Flatten, BatchNormalization, LSTM, Activation, Layer
from keras.layers.advanced_activations import LeakyReLU, ELU, ReLU
from keras.layers import initializers
from keras.optimizers import Adam, RMSprop, SGD, Adadelta

import pdb
import lhapdf
import hyperopt
import numpy as np
from random import sample
import matplotlib.pyplot as plt

# Format the input PDFs
class input_pdfs(object):

    def __init__(self, pdf_name, x_pdf, nb_replicas, Q_value, flavors):
        self.x_pdf    = x_pdf
        self.Q_value  = Q_value
        self.pdf_name = pdf_name
        self.flavors  = flavors
        self.nb_replicas = nb_replicas

    def build_pdf(self):

        # Get the PDF4LHC15 for test purpose and print some description
        if self.nb_replicas == 1:
            pdf_central = [lhapdf.mkPDF(self.pdf_name, 0)]
        else:
            pdf_init = lhapdf.mkPDFs(self.pdf_name)
            pdf_central = sample(pdf_init, self.nb_replicas)

        # Format the input PDFs
        data  = []
        for pdf in pdf_central:
            row = []
            for x in self.x_pdf:
                row.append(pdf.xfxQ2(self.flavors,x,self.Q_value)-pdf.xfxQ2(-self.flavors,x,self.Q_value))
            data.append(row)
        return np.array(data)

# Define a custom layer containing information about x
class xlayer(Layer):

    def __init__(self, output_dim, xval, kernel_initializer='glorot_uniform', **kwargs):
        self.output_dim = output_dim
        self.xval = K.constant(xval)
        self.kernel_initializer = initializers.get(kernel_initializer)
        super(xlayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel', shape=(K.int_shape(self.xval)[0],
                input_shape[1], self.output_dim), initializer=self.kernel_initializer,
                trainable=True)
        super(xlayer, self).build(input_shape)

    def call(self, x):
        # xres outputs (None, input_shape[1], len(x_pdf))
        xres = K.tf.tensordot(x, self.xval, axes=0)
        # xfin outputs (None, output_dim)
        xfin = K.tf.tensordot(xres, self.kernel, axes=([1,2],[0,1]))
        return xfin

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


# Construct the Model
class xgan_model(object):

    def __init__(self, noise_size, output_size, x_pdf, params, activ, optmz):
        self.noise_size  = noise_size
        self.output_size = output_size
        self.G = None
        self.D = None
        self.G_M = None
        self.D_M = None
        self.GAN_M = None
        self.x_pdf = x_pdf

        self.params = params
        self.activ  = activ
        self.optmz  = optmz
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


# Do the training
class xgan_train(object):

    def __init__(self, x_pdf, pdf_name, noise_size, params, activ, optmz, nb_replicas=1, Q_value=1.7874388, flavors=2):
        self.sampled_pdf = input_pdfs(pdf_name, x_pdf, nb_replicas, Q_value, flavors).build_pdf()
        self.x_pdf = x_pdf
        self.nb_replicas = nb_replicas
        self.output_size = len(x_pdf)
        self.noise_size = noise_size

        self.xgan_model = xgan_model(noise_size, self.output_size, x_pdf, params, activ, optmz)
        self.generator = self.xgan_model.generator_model()
        self.discriminator = self.xgan_model.discriminator_model()
        self.gan = self.xgan_model.gan_model()

    def plot_generated_pdf(self, nth_training, nrep):
        noise = np.random.normal(0, 1, size=[nrep, self.noise_size])
        generated_pdf = self.generator.predict(noise)

        plt.figure()
        for i in range(generated_pdf.shape[0]):
            plt.plot(self.x_pdf, self.sampled_pdf[i], color='blue', alpha=0.75)
            plt.plot(self.x_pdf, generated_pdf[i], color='red', alpha=0.75)
        plt.title('Samples at Iteration %d'%nth_training)
        plt.tight_layout()
        plt.savefig('iterations/gan_generated_pdf_at_training_%d.png' % nth_training, dpi=250)
        plt.close()

    def train(self, nb_training=20000, batch_size=4, nd_steps=6, ng_steps=1, verbose=False):

        for k in range(1, nb_training+1):
            for _ in range(int(self.sampled_pdf.shape[0]/batch_size)):

                pdf_batch = self.sampled_pdf[np.random.randint(0, self.sampled_pdf.shape[0],
                                            size=batch_size)]

                # Train the Discriminator
                for _ in range(nd_steps):
                    noise = np.random.normal(0,1,size=[batch_size, self.noise_size])
                    pdf_fake = self.generator.predict(noise)
                    xinput = np.concatenate([pdf_batch, pdf_fake])

                    y_disc = np.zeros(2*batch_size)
                    y_disc[:batch_size] = 1.0

                    # Make sure to train the Discriminator
                    self.discriminator.trainable = True
                    dloss = self.discriminator.train_on_batch(xinput, y_disc)

                # Train the GAN
                for _ in range(ng_steps):
                    noise = np.random.normal(0,1,size=[batch_size,self.noise_size])
                    y_gen = np.ones(batch_size)

                    # Make sure that the Discriminator is not trained
                    self.discriminator.trainable = False
                    gloss = self.gan.train_on_batch(noise, y_gen)

                if verbose:
                    loss_info = "Iteration %d: \t .D loss: %f \t D acc: %f" % (k, dloss[0], dloss[1])
                    loss_info = "%s  \t .G loss: %f" % (loss_info, gloss[0])

                    if k % 100 == 0:
                        print(loss_info)

                    if k % 1000 == 0:
                        self.plot_generated_pdf(k, self.nb_replicas)

        return gloss[0]



# Define global Variable
X_PDF = np.load('x_grid.npy')
NB_INPUT_REP = 1

# Dictionary for activation funtions
activ = {'leakyrelu': LeakyReLU(alpha=0.2), 'elu': ELU(alpha=1.0), 'relu': ReLU()}

# Dictionary for optimization functions
optmz = {'sgd': SGD(lr=0.01), 'rms': RMSprop(lr=0.001), 'adadelta': Adadelta(lr=1.0)}

# Define the hyper parameter optimization function
def hyper_train(params):
    xgan_pdfs = xgan_train(X_PDF, "NNPDF31_nnlo_as_0118", 100, params, activ, optmz, nb_replicas=NB_INPUT_REP)
    g_loss = xgan_pdfs.train(nb_training=14000, batch_size=1, nd_steps=2, ng_steps=3, verbose=False)
    return {'loss': g_loss, 'status': 'ok'}

# Define the hyper parameters
hparams = {'g_nodes'  : hyperopt.hp.choice('g_nodes', [128,256,512]),
           'd_nodes'  : hyperopt.hp.choice('d_nodes', [128,256,512,1024]),
           'g_act'    : hyperopt.hp.choice('g_act', ['leakyrelu', 'elu', 'relu']),
           'd_act'    : hyperopt.hp.choice('d_act', ['leakyrelu', 'elu', 'relu']),
           'g_opt'    : hyperopt.hp.choice('g_opt', ['sgd', 'rms', 'adadelta']),
           'd_opt'    : hyperopt.hp.choice('d_opt', ['sgd', 'rms', 'adadelta']),
           'gan_opt'  : hyperopt.hp.choice('gan_opt', ['sgd', 'rms', 'adadelta']),
           'g_loss'   : hyperopt.hp.choice('g_loss', ['binary_crossentropy','mean_squared_error']),
           'd_loss'   : hyperopt.hp.choice('d_loss', ['binary_crossentropy','mean_squared_error']),
           'gan_loss' : hyperopt.hp.choice('gan_loss', ['binary_crossentropy','mean_squared_error'])}

if __name__ == '__main__':

    # Hyper Scan
    trials = hyperopt.Trials()

    hyper_result = hyperopt.fmin(
                fn        = hyper_train,
                space     = hparams,
                algo      = hyperopt.tpe.suggest,
                trials    = trials,
                max_evals = 4)

    print(hyper_result)
