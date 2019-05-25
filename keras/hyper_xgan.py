import numpy as np
from random import sample

import hyperopt
import keras.backend as K
from keras import Model
from keras.layers import Input
from keras.optimizers import Adam, RMSprop, Adadelta, SGD

from model import generator_model, discriminator_model
from model import generator_model_cnn, discriminator_model_cnn
from ploting import plot_generated_pdf, plot_generated_repl
from pdformat import x_pdf, sample_pdf, nb_input_rep

# Generate the data
pdf_dataX = sample_pdf()
length    = pdf_dataX.shape[1]

# Define parameters
random_noise_dim = 100

# Define the batch size relative to the number of input replicas
if nb_input_rep == 1:
    batch_size = 1
else:
    batch_size = nb_input_rep//4
batch_count = int(pdf_dataX.shape[0]/batch_size)


def hyper_params(params):

    K.clear_session()

    # Call the Generator Model
    def make_generator():
        return generator_model(random_noise_dim, length, params)
    
    # Call the Discriminator Model
    def make_discriminator():
        return discriminator_model(length, params)
    
    # Compile the Gen
    Generator = make_generator()
    Generator.name = "generator"
    # Generator.compile(loss='mean_squared_error', optimizer = optimizer1)
    Generator.compile(loss=params['g_loss'], optimizer=params['g_opt'])
    Generator.summary()
    
    # Compile the Dis
    Discriminator = make_discriminator()
    Discriminator.name = 'discriminator'
    Discriminator.compile(loss=params['d_loss'], optimizer=params['d_opt'], metrics=['accuracy'])
    Discriminator.summary()
    
    # Choose to train the Discriminator or Not
    always_train_Discriminator = False
    
    Gan_input  = Input(shape = (random_noise_dim,))
    Gan_latent = Generator(Gan_input)
    Gan_output = Discriminator(Gan_latent)
    
    if not always_train_Discriminator:
        Discriminator.trainable = False
    GAN = Model(inputs = Gan_input, outputs = Gan_output)
    GAN.name   = "gan"
    GAN.compile(loss=params['gan_loss'], optimizer=params['gan_opt'])
    GAN.summary()
    
    # Set the number of training
    number_training = 14000
    
    # Number of steps to train G&D
    nd_steps = 1
    ng_steps = 1
    
    # f = open('loss.csv','w')
    # f.write('Iteration,Discriminator Loss,Discriminator accuracy,Generator Loss\n')
    
    # Train the Model
    for k in range(1,number_training+1):
        for _ in range(batch_count):
            pdf_batch = pdf_dataX[np.random.randint(0, pdf_dataX.shape[0], size=batch_size)]
    
            for m in range(nd_steps):
                noise = np.random.normal(0,1,size=[batch_size,random_noise_dim])
                pdf_fake  = Generator.predict(noise)
                xinput = np.concatenate([pdf_batch,pdf_fake])
    
                # Train the Discriminator with trues and fakes
                y_Discriminator = np.zeros(2*batch_size)
                y_Discriminator[:batch_size] = 1.0
                if not always_train_Discriminator:
                    Discriminator.trainable = True  # Ensure the Discriminator is trainable
                dloss = Discriminator.train_on_batch(xinput, y_Discriminator)
    
            for n in range(ng_steps):
                # Train the generator by generating fakes and lying to the Discriminator
                noise = np.random.normal(0, 1, size = [batch_size, random_noise_dim])
                y_gen = np.ones(batch_size)
                if not always_train_Discriminator:
                    Discriminator.trainable = False
                gloss = GAN.train_on_batch(noise, y_gen)
    
        # loss_info = "Iteration %d: \t .D loss: %f \t D acc: %f" % (k, dloss[0], dloss[1])
        # loss_info = "%s  \t .G loss: %f" % (loss_info, gloss)
    
        # if k % 100 == 0:
        #     print(loss_info)
        #     f.write("%d,%f,%f,%f\n"%(k,dloss[0],dloss[1],gloss))
    
        # if k % 1000==0:
        #     plot_generated_pdf(x_pdf, pdf_dataX, random_noise_dim, k, Generator, always_train_Discriminator, pdf_repl=batch_size)

    return {'loss': gloss, 'status': 'ok'}
    
    # f.close()

params = {'g_nodes' : hyperopt.hp.choice('g_nodes', [128,256,512]),
          'd_nodes' : hyperopt.hp.choice('d_nodes', [128,256,512,1024]),
          'g_loss'  : hyperopt.hp.choice('g_loss', ['mean_squared_error', 'binary_crossentropy']),
          'd_loss'  : hyperopt.hp.choice('d_loss', ['mean_squared_error', 'binary_crossentropy']),
          'gan_loss': hyperopt.hp.choice('gan_loss', ['mean_squared_error', 'binary_crossentropy']),
          'g_opt'   : hyperopt.hp.choice('g_opt', ['Adadelta', 'Adam', 'RMSprop']),
          'd_opt'   : hyperopt.hp.choice('d_opt', ['Adadelta', 'Adam', 'RMSprop']),
          'gan_opt' : hyperopt.hp.choice('gan_opt', ['Adadelta', 'Adam', 'RMSprop'])}

trials = hyperopt.Trials()

hyper_result = hyperopt.fmin(
         fn     = hyper_params,
         space  = params,
         algo   = hyperopt.tpe.suggest,
         trials = trials,
         max_evals = 300)

print(hyper_result)

# ## GENERATE AS MANY REPLICAS AS ONE WANTS ##
# nb_repl = 10000
# x_index = [60,61,62,63,64] # X values for the histogram plots
# plot_generated_repl(x_pdf, Generator, nb_repl, random_noise_dim, x_index)
