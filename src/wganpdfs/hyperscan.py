import re
import ast
import yaml
import time
import pickle

from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import SGD, RMSprop, Adadelta
from tensorflow.keras.layers import ELU, ReLU, LeakyReLU

from hyperopt import fmin, tpe, hp
from hyperopt.mongoexp import MongoTrials
from hyperopt import space_eval, STATUS_OK

from wganpdfs.pdformat import xnodes
from wganpdfs.train import xgan_train
from wganpdfs.filetrials import FileTrials

re_function = re.compile("(?<=hp\.)\w*(?=\()")
re_args = re.compile("\(.*\)$")

# ----------------------------------------------------------------------
def load_yaml(runcard_file):
    """Loads yaml runcard"""
    with open(runcard_file, "r") as stream:
        runcard = yaml.load(stream)
    hyperdict = runcard.get("hyperopt", [])
    for key, value in hyperdict.items():
        fname = re_function.search(value)
        if fname is None:
            raise ValueError(f"No hp.function found in ${key}:{value}")
        fname = fname[0]
        args = re_args.search(value)
        if args is None:
            raise ValueError(f"No arguments found in ${key}:{value}")
        vals = ast.literal_eval(args[0])
        runcard[key] = getattr(hp, fname)(*vals)
    return runcard


# ----------------------------------------------------------------------
def run_hyperparameter_scan(search_space, max_evals, cluster, folder):

    """
    Run the hyperparameter scan using hyperopt.
    """

    print("[+] Performing hyperparameter scan...")
    if cluster:
        trials = MongoTrials(cluster, exp_key="exp1")
    else:
        """
        Use constum trials in order to save the model after each trials.
        The following will generate a .json file containing the trials.
        """
        trials = FileTrials(folder, parameters=search_space)
    best = fmin(
        hyper_train, search_space, algo=tpe.suggest, max_evals=max_evals, trials=trials
    )
    # Save the overall best model
    best_setup = space_eval(search_space, best)
    print("\n[+] Best scan setup:")
    pprint.pprint(best_setup)
    with open("%s/best-model.yaml" % folder, "w") as wfp:
        yaml.dump(best_setup, wfp, default_flow_style=False)
    log = "%s/hyperopt_log_{}.pickle".format(time.time()) % folder
    with open(log, "wb") as wfp:
        print(f"[+] Saving trials in {log}")
        pickle.dump(trials.trials, wfp)
    return best_setup


# ----------------------------------------------------------------------
def hyper_train(params):

    """
    Define the hyper parameters optimization function.
    """
    # Load the x_grid
    X_PDF = xnodes().build_xgrid()
    # Define the number of input replicas
    NB_INPUT_REP = params["input_replicas"]
    # Define the number of batches
    if NB_INPUT_REP < 10:
        BATCH_SIZE = 1
    else:
        BATCH_SIZE = int(NB_INPUT_REP / 10)

    # Clear Keras session
    K.clear_session()

    # Model Parameters
    # List of activation funtions
    activ = {"leakyrelu": LeakyReLU(alpha=0.2), "elu": ELU(alpha=1.0), "relu": ReLU()}
    # List of optimization functions
    optmz = {
        "sgd": SGD(lr=0.0075),
        "rms": RMSprop(lr=0.00005),
        "adadelta": Adadelta(lr=1.0),
    }

    xgan_pdfs = xgan_train(
        X_PDF,
        params["pdf"],
        100,
        params,
        activ,
        optmz,
        nb_replicas=NB_INPUT_REP,
        flavors=params["fl"],
    )

    # In case one needs to pretrain the Discriminator
    # xgan_pdfs.pretrain_disc(BATCH_SIZE, epochs=4)

    g_loss = xgan_pdfs.train(
        nb_epochs=params["epochs"], batch_size=BATCH_SIZE, verbose=params["verbose"]
    )
    return {"loss": g_loss, "status": STATUS_OK}
