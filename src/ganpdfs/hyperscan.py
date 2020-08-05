import re
import ast
import yaml
import time
import pickle

from tensorflow.keras import backend as K
from tensorflow.keras.layers import ELU
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import Adadelta

from hyperopt import fmin, tpe, hp
from hyperopt.mongoexp import MongoTrials
from hyperopt import space_eval, STATUS_OK

from ganpdfs.train import GanTrain
from ganpdfs.filetrials import FileTrials

RE_FUNCTION = re.compile("(?<=hp\.)\w*(?=\()")
RE_ARGS = re.compile("\(.*\)$")

# ----------------------------------------------------------------------
def load_yaml(runcard_file):
    """load_yaml.

    Parameters
    ----------
    runcard_file :
        runcard_file
    """
    with open(runcard_file, "r") as stream:
        runcard = yaml.load(stream, Loader=yaml.FullLoader)
    hyperdict = runcard.get("hyperopt", {})
    for key, value in hyperdict.items():
        fname = RE_FUNCTION.search(value)
        if fname is None:
            raise ValueError(f"No hp.function found in ${key}:{value}")
        fname = fname[0]
        args = RE_ARGS.search(value)
        if args is None:
            raise ValueError(f"No arguments found in ${key}:{value}")
        vals = ast.literal_eval(args[0])
        runcard[key] = getattr(hp, fname)(*vals)
    return runcard


# ----------------------------------------------------------------------
def run_hyperparameter_scan(func_train, search_space, max_evals, cluster, folder):
    """run_hyperparameter_scan.

    Parameters
    ----------
    func_train :
        func_train
    search_space :
        search_space
    max_evals :
        max_evals
    cluster :
        cluster
    folder :
        folder
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
        func_train, search_space, algo=tpe.suggest, max_evals=max_evals, trials=trials
    )
    # Save the overall best model
    best_setup = space_eval(search_space, best)
    print("\n[+] Best scan setup:")
    #     pprint.pprint(best_setup)
    with open("%s/best-model.yaml" % folder, "w") as wfp:
        yaml.dump(best_setup, wfp, default_flow_style=False)
    log = "%s/hyperopt_log_{}.pickle".format(time.time()) % folder
    with open(log, "wb") as wfp:
        print(f"[+] Saving trials in {log}")
        pickle.dump(trials.trials, wfp)
    return best_setup


# ----------------------------------------------------------------------
def hyper_train(params, xpdf, pdf):
    """hyper_train.

        Parameters
        ----------
        params :
            params
        xpdf :
            xpdf
        pdf :
            pdf
    """
    # Define the number of input replicas
    NB_INPUT_REP = params["input_replicas"]
    # Define the number of batches
    if NB_INPUT_REP < 10:
        BATCH_SIZE = 1
    else:
        BATCH_SIZE = int(NB_INPUT_REP / 100)

    # Model Parameters
    # TODO do this depending on the parameters instead of generating all of them
    # List of activation funtions
    activ = {"leakyrelu": LeakyReLU(alpha=0.2), "elu": ELU(alpha=1.0), "relu": ReLU()}
    # List of optimization functions
    optmz = {
        "sgd": SGD(lr=0.0075),
        "adam": Adam(1e-4),
        "rms": RMSprop(lr=0.00005),
        "adadelta": Adadelta(lr=1.0),
    }

    # Train on Input/True pdf
    xgan_pdfs = GanTrain(
        xpdf, pdf, 100, params, activ, optmz, nb_replicas=NB_INPUT_REP,
    )

    # In case one needs to pretrain the Discriminator
    # xgan_pdfs.pretrain_disc(BATCH_SIZE, epochs=4)

    smm_result = xgan_pdfs.train(
        nb_epochs=params["epochs"], batch_size=BATCH_SIZE, verbose=params["verbose"]
    )
    return {"loss": smm_result, "status": STATUS_OK}
