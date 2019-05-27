import numpy as np
from random import sample
import argparse, shutil
import yaml, os, pprint, time, pickle
import hyperopt
from hyperopt import fmin, tpe, hp, Trials, space_eval, STATUS_OK
from hyperopt.mongoexp import MongoTrials

import keras.backend as K
from keras import Model
from keras.layers import Input
from keras.optimizers import Adam, RMSprop, Adadelta, SGD

from ganpdfs.model import generator_model, discriminator_model
from ganpdfs.model import generator_model_cnn, discriminator_model_cnn
from ganpdfs.ploting import plot_generated_pdf, plot_generated_repl
from ganpdfs.pdformat import x_pdf, sample_pdf, nb_input_rep


def run_hyperparameter_scan(search_space, max_evals, cluster, folder):
    """Running hyperparameter scan using hyperopt"""

    print('[+] Performing hyperparameter scan...')
    if cluster:
        trials = MongoTrials(cluster, exp_key='exp1')
    else:
        trials = Trials()
    best = fmin(build_and_train_model, search_space, algo=tpe.suggest, 
                max_evals=max_evals, trials=trials)
    best_setup = space_eval(search_space, best)
    print('\n[+] Best scan setup:')
    pprint.pprint(best_setup)
    with open('%s/best-model.yaml' % folder, 'w') as wfp:
        yaml.dump(best_setup, wfp, default_flow_style=False)
    log = '%s/hyperopt_log_{}.pickle'.format(time()) % folder
    with open(log, 'wb') as wfp:
        print(f'[+] Saving trials in {log}')
        pickle.dump(trials.trials, wfp)
    return best_setup


def build_and_train_model(params, plot=False):
    """Training model"""
    K.clear_session()

        # Generate the data
    pdf_dataX = sample_pdf(params['pdf'])
    length    = pdf_dataX.shape[1]

    # Define parameters
    random_noise_dim = 100

    # Define the batch size relative to the number of input replicas
    if nb_input_rep == 1:
        batch_size = 1
    else:
        batch_size = nb_input_rep//4
    batch_count = int(pdf_dataX.shape[0]/batch_size)

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
    number_training = params['epochs']
    
    # Number of steps to train each G&D
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
        
        if k % 100 == 0:
            loss_info = "Iteration %d: \t .D loss: %f \t D acc: %f" % (k, dloss[0], dloss[1])
            loss_info = "%s  \t .G loss: %f" % (loss_info, gloss)
            print(loss_info)
        
        if plot:    
            if k % 1000==0:
                plot_generated_pdf(x_pdf, pdf_dataX, random_noise_dim, k, Generator, always_train_Discriminator, pdf_repl=batch_size)

    return {'loss': gloss, 'status': STATUS_OK}
    

#----------------------------------------------------------------------
def load_yaml(runcard_file):
    """Loads yaml runcard"""
    with open(runcard_file, 'r') as stream:
        runcard = yaml.load(stream)
    for key, value in runcard.items():
        if 'hp.' in str(value):
            runcard[key] = eval(value)
    return runcard


#----------------------------------------------------------------------
def main():
    """Main controller"""
    # read command line arguments
    parser = argparse.ArgumentParser(description='Train a PDF GAN.')
    parser.add_argument('runcard', action='store', default=None,
                        help='A json file with the setup.')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='The output folder.', required=True)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--hyperopt', default=None, type=int,
                        help='Enable hyperopt scan.')
    parser.add_argument('--cluster', default=None, type=str, 
                        help='Enable cluster scan.')
    args = parser.parse_args()

    # check input is coherent
    if not os.path.isfile(args.runcard):
        raise ValueError('Invalid runcard: not a file.')
    if args.force:
        print('WARNING: Running with --force option will overwrite existing model.')

    # prepare the output folder
    if not os.path.exists(args.output):
        os.mkdir(args.output)
    elif args.force:
        shutil.rmtree(args.output)
        os.mkdir(args.output)
    else:
        raise Exception(f'{args.output} already exists, use "--force" to overwrite.')
    out = args.output.strip('/')

    # copy runcard to output folder
    shutil.copyfile(args.runcard, f'{out}/input-runcard.json')

    print('[+] Loading runcard')
    hps = load_yaml(args.runcard)

    if args.hyperopt:
        hps['scan'] = True
        hps = run_hyperparameter_scan(hps, args.hyperopt, args.cluster, out)
    hps['scan'] = False

    loss = build_and_train_model(hps)

    # ## GENERATE AS MANY REPLICAS AS ONE WANTS ##
    # nb_repl = 10000
    # x_index = [60,61,62,63,64] # X values for the histogram plots
    # plot_generated_repl(x_pdf, Generator, nb_repl, random_noise_dim, x_index)
