import re
import ast
import yaml
import time
import pickle
import logging

from tensorflow.keras import backend as K

from hyperopt import fmin, tpe, hp
from hyperopt.mongoexp import MongoTrials
from hyperopt import space_eval, STATUS_OK

from ganpdfs.train import GanTrain
from ganpdfs.filetrials import FileTrials

logger = logging.getLogger(__name__)

RE_FUNCTION = re.compile("(?<=hp\.)\w*(?=\()")
RE_ARGS = re.compile("\(.*\)$")


def recursive_dic(hyperdict):
    """Parse netsed dictionary for hyperopt.

    Parameters
    ----------
    hyperdict: dict
        Dictionary containing information on the hyper-parameter tune

    Returns
    -------
    dict:
        Dictionary where the values of the keys are hp.functions
    """

    newhyperdict = {}
    for key, value in hyperdict.items():
        if isinstance(value, dict):
            newhyperdict[key] = recursive_dic(value)
        else:
            fname = RE_FUNCTION.search(value)
            if fname is None:
                raise ValueError(f"No hp.function found in ${key}:{value}")
            fname = fname[0]
            args = RE_ARGS.search(value)
            if args is None:
                raise ValueError(f"No arguments found in ${key}:{value}")
            vals = ast.literal_eval(args[0])
            newhyperdict[key] = getattr(hp, fname)(*vals)
    return newhyperdict


def load_yaml(runcard_file):
    """Load YAML file.

    Parameters
    ----------
    runcard_file : str
        input runcard file
    """

    with open(runcard_file, "r") as stream:
        runcard = yaml.load(stream, Loader=yaml.FullLoader)
    hyperdict = runcard.get("hyperopt", {})
    newhyperdict = recursive_dic(hyperdict)
    runcard.update(newhyperdict)
    return runcard


def run_hyperparameter_scan(func_train, search_space, max_evals, cluster, folder):
    """Take care of fetching all the necessary information from a runcard file
    in order to perfrom the hyperparameter scan.

    Parameters
    ----------
    func_train : ganpdfs.train.GanTrain
        GANs function that takes care of the training
    search_space :
        search_space
    max_evals : int
        total number of evalutions
    cluster : str
        cluster adresses
    folder : str
        folder to store the results
    """

    logger.info("Performing hyperparameter scan.")
    if cluster:
        trials = MongoTrials(cluster, exp_key="exp1")
    else:
        # Use constum trials in order to save the model after each trials.
        # The following will generate a .json file containing the trials.
        trials = FileTrials(folder, parameters=search_space)
    best = fmin(
        func_train,
        search_space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials
    )
    # Save the overall best model
    best_setup = space_eval(search_space, best)
    logger.info("Best scan setup:")
    #     pprint.pprint(best_setup)
    with open("%s/best-model.yaml" % folder, "w") as wfp:
        yaml.dump(best_setup, wfp, default_flow_style=False)
    log = "%s/hyperopt_log_{}.pickle".format(time.time()) % folder
    with open(log, "wb") as wfp:
        logger.info(f"Saving trials in {log}")
        pickle.dump(trials.trials, wfp)
    return best_setup


def hyper_train(params, xpdf, pdf):
    """Take care of running the hyperparameter scan by designing a
    search space through which the model tries to find the best
    architecture that maximizes the figure of merit.

    Parameters
    ----------
    params : dict
        dictionary that contains all the input parameters on which the
        search space is based
    xpdf : np.array(float)
        array of x-grid
    pdf : np.array(float)
        input/prior pdf
    """

    # Train on Input/True pdf
    xgan_pdfs = GanTrain(xpdf, pdf, params)

    smm_result = xgan_pdfs.train(nb_epochs=params.get("epochs"))
    # TODO: rename "loss" throughout hyperopt
    return {"loss": smm_result, "status": STATUS_OK}
