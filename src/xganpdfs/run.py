import hyperopt
import numpy as np
import argparse, shutil
import keras.backend as K
import yaml, os, pprint, time, pickle
from xganpdfs.xgrid import xnodes
from xganpdfs.hyper_xgan import xgan_train
from xganpdfs.filetrials import FileTrials
from hyperopt import fmin, tpe, hp, Trials, space_eval, STATUS_OK
from hyperopt.mongoexp import MongoTrials
from keras.layers.advanced_activations import LeakyReLU, ELU, ReLU
from keras.optimizers import Adam, RMSprop, SGD, Adadelta


#----------------------------------------------------------------------
def run_hyperparameter_scan(search_space, max_evals, cluster, folder):
    """Running hyperparameter scan using hyperopt"""

    print('[+] Performing hyperparameter scan...')
    if cluster:
        trials = MongoTrials(cluster, exp_key='exp1')
    else:
        """
        Use constum trials in order to save the model after each trials.
        The following will generate a .json file containing the trials.
        """
        trials = FileTrials(folder, parameters=search_space)
    best = fmin(hyper_train, search_space, algo=tpe.suggest, 
                max_evals=max_evals, trials=trials)
    # Save the overall best model
    best_setup = space_eval(search_space, best)
    print('\n[+] Best scan setup:')
    pprint.pprint(best_setup)
    with open('%s/best-model.yaml' % folder, 'w') as wfp:
        yaml.dump(best_setup, wfp, default_flow_style=False)
    log = '%s/hyperopt_log_{}.pickle'.format(time.time()) % folder
    with open(log, 'wb') as wfp:
        print(f'[+] Saving trials in {log}')
        pickle.dump(trials.trials, wfp)
    return best_setup


#----------------------------------------------------------------------
# Define the hyper parameter optimization function
def hyper_train(params):
    # Load the x_grid
    X_PDF = xnodes().build_xgrid()
    # Define the number of input replicas
    NB_INPUT_REP = params['input_replicas']
    # Define the number of batches
    if NB_INPUT_REP < 5:
        BATCH_SIZE = 1
    else:
        BATCH_SIZE = int(NB_INPUT_REP/5)

    # Clear Keras session
    K.clear_session()
    # Dictionary for activation funtions
    activ = {'leakyrelu': LeakyReLU(alpha=0.2), 'elu': ELU(alpha=1.0), 'relu': ReLU()}
    # Dictionary for optimization functions
    optmz = {'sgd': SGD(lr=0.01), 'rms': RMSprop(lr=0.001), 'adadelta': Adadelta(lr=1.0)}
    xgan_pdfs = xgan_train(X_PDF, params['pdf'], 100, params, activ, optmz, nb_replicas=NB_INPUT_REP, flavors=params['fl'])
    # In case one needs to pretrain the Discriminator
    # xgan_pdfs.pretrain_disc(BATCH_SIZE)
    g_loss = xgan_pdfs.train(nb_training=params['epochs'], batch_size=BATCH_SIZE, verbose=params['verbose'])
    return {'loss': g_loss, 'status': STATUS_OK} 


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
    parser.add_argument('--nreplicas', default=None, type=int,
                        help='Define the number of input replicas.')
    parser.add_argument('--pplot', default=None, type=int,
                        help='Define the number of output replicas.')
    parser.add_argument('--flavors', default=None, type=int,
                        help='Choose the falvours.')
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
    hps['verbose'] = False
    hps['save_output'] = out

    # Define the number of input replicas
    if args.nreplicas == None:
        hps['input_replicas'] = 1
    elif args.nreplicas > 0 :
        hps['input_replicas'] = args.nreplicas
    else:
        raise Exception(f'{args.nreplicas} must be a positive value!!!')

    # Define the number of output replicas
    if args.pplot == None:
        hps['out_replicas'] = hps['input_replicas']
    elif args.pplot > 0:
        hps['out_replicas'] = args.pplot
    else:
        raise Exception(f'{args.pplot} must be a positive value!!!')

    # Check the input flavor
    if args.flavors == None:
        hps['fl'] = 2       # Take the u quark as a default
    elif args.flavors in [1,2,3,21]:
        hps['fl'] = args.flavors
    else:
        raise Exception(f'{args.falvors} must be one of the particle IDs!!!')

    # # Print the Summary
    # print("""[*] Summary of the parameters: \n 
    #         \t - Number of input replicas : %d \n
    #         \t - Number of ouput replicas : %d \n
    #         \t - Chosen flavor            : %d """
    #         %(hps['input_replicas'], hps['out_replicas'], hps['fl']))

    # If hyperscan is set true
    if args.hyperopt:
        hps['scan'] = True
        hps = run_hyperparameter_scan(hps, args.hyperopt, args.cluster, out)

    # Run the best Model and output log
    hps['scan'] = False
    hps['verbose'] = True

    loss = hyper_train(hps)
