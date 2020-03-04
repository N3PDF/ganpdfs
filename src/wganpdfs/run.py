import os, argparse, shutil
from wganpdfs.hyperscan import hyper_train, load_yaml, run_hyperparameter_scan

def main():
    """
    This is the Main controller.
    It controlls the input parameters.
    """

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
    # parser.add_argument('--timeline', action='store_true')
    args = parser.parse_args()

    # check the runcard
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
        raise Exception(f'{args.nreplicas} not valid. Value must be positive!!!')

    # Define the number of output replicas
    if args.pplot == None:
        hps['out_replicas'] = hps['input_replicas']
    elif args.pplot > 0:
        hps['out_replicas'] = args.pplot
    else:
        raise Exception(f'{args.pplot} not valid. Value must be positive!!!')

    # Check the input flavor
    if args.flavors == None:
        hps['fl'] = 1       # Take the u quark as a default
    elif args.flavors in [1,2,3,21]:
        hps['fl'] = args.flavors
    else:
        raise Exception(f'{args.falvors} not valid. Must be one of the particle IDs!!!')

    # If hyperscan is set true
    if args.hyperopt:
        hps['scan'] = True
        hps = run_hyperparameter_scan(hps, args.hyperopt, args.cluster, out)

    # Run the best Model and output log
    hps['scan'] = False
    hps['verbose'] = True

    loss = hyper_train(hps)
