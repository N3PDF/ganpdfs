from keras import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
from keras.layers import Reshape, Conv2D, Flatten, BatchNormalization, LSTM, Activation, Layer
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import initializers
import keras.backend as K
from pdformat import x_pdf
import pdb

# Define a custom layer containing information about x
class xlayer(Layer):

    def __init__(self, output_dim, xval=x_pdf, kernel_initializer='glorot_uniform', **kwargs):
        self.output_dim = output_dim
        self.xval = K.constant(xval)
        self.kernel_initializer = initializers.get(kernel_initializer)
        super(xlayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(K.int_shape(self.xval)[0], input_shape[1], self.output_dim),
                                      initializer=self.kernel_initializer,
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

# Construct the Generator
def generator_model(noise_size, output_size, params):

    # Take the input noise
    G_input = Input(shape=(noise_size,))

    # Construct the Model
    G = Dense(params['g_nodes']//4)(G_input)
    G = LeakyReLU(0.2)(G)

    G = xlayer(params['g_nodes']//2)(G)
    G = LeakyReLU(0.2)(G)

    G = Dense(params['g_nodes'])(G)
    G = LeakyReLU(0.2)(G)

    G_output = Dense(output_size, activation='sigmoid')(G)
    # G_output = Activation("tanh")(G_output)

    generator = Model(G_input, G_output)

    return generator

# Construct the Discriminator
def discriminator_model(GAN_size, params):

    # Take the generated output
    D_input = Input(shape=(GAN_size,))

    # Construct the Model
    D = Dense(params['d_nodes'])(D_input)
    D = LeakyReLU(0.2)(D)

    D = Dense(params['d_nodes']//2)(D)
    D = LeakyReLU(0.2)(D)
    
    D = Dense(params['d_nodes']//4)(D)
    D = LeakyReLU(0.2)(D)
    # D = Dropout(0.2)(D)

    D_output = Dense(1, activation='sigmoid')(D)

    discriminator = Model(D_input, D_output)

    return discriminator

## CNN MODEL ## 

def generator_model_cnn(noise_size, output_size):

    G_input = Input(shape=(noise_size,))

    G = Dense(128, kernel_initializer='glorot_normal')(G_input)
    G = Activation('tanh')(G)
    G = BatchNormalization(momentum=0.99)(G)
    G = Reshape((32, 4))(G)
    G = LSTM(32, return_sequences=True)(G)
    G = LSTM(16, return_sequences=False)(G)
    G = Activation('tanh')(G)

    G_output = Dense(output_size, activation="tanh")(G)

    generator = Model(G_input, G_output)

    return generator

def discriminator_model_cnn(GAN_size):

    D_input = Input(shape=(GAN_size,))

    D = Dense(128)(D_input)
    D = Reshape((8, 8, 2))(D)
    D = Conv2D(64, kernel_size=3, strides=1, padding="same")(D)
    D = LeakyReLU(alpha=0.2)(D)
    D = Conv2D(32, kernel_size=3, strides=1, padding="same")(D)
    #D = BatchNormalization()(D)
    D = LeakyReLU(alpha=0.2)(D)
    D = Conv2D(16, kernel_size=3, strides=1, padding="same")(D)
    #D = BatchNormalization()(D)
    D = LeakyReLU(alpha=0.2)(D)
    D = Flatten()(D)
    #D = BatchNormalization()(D)
    D = LeakyReLU(alpha=0.2)(D)
    D = Dropout(0.2)(D)

    D_output = Dense(1, activation="sigmoid")(D)
    # D_output = Dense(1)(D)

    discriminator = Model(D_input, D_output)
    return discriminator
