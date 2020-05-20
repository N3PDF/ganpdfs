"""
    This file contains the main driver of the wganpdfs code
"""
import os
import shutil
import logging
import argparse

from wganpdfs.pdformat import xnodes
from wganpdfs.pdformat import input_pdfs
from wganpdfs.hyperscan import hyper_train
from wganpdfs.hyperscan import load_yaml
from wganpdfs.hyperscan import run_hyperparameter_scan

logger = logging.getLogger(__name__)

ALLOWED_FLAVORS = [1, 2, 3, 21]


def positive_int(value):
    """ Checks that a given number is positive """
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(
            f"Negative values are not allowed, received: {value}"
        )
    return ivalue


def flavour_int(value):
    """ Check whether the given flavour is among the allowed values """
    ivalue = int(value)
    if ivalue not in ALLOWED_FLAVORS:
        raise argparse.ArgumentTypeError(
            f"{ivalue} not allowed, allowed values: {ALLOWED_FLAVORS}"
        )
    return ivalue


def argument_parser():
    """
        Parse the input arguments for wganpdfs
    """
    # read command line arguments
    parser = argparse.ArgumentParser(description="Train a PDF GAN.")
    parser.add_argument("runcard", help="A json file with the setup.")
    parser.add_argument("-o", "--output", help="The output folder.", required=True)
    parser.add_argument("-f", "--force", action="store_true")
    parser.add_argument("--hyperopt", type=int, help="Enable hyperopt scan.")
    parser.add_argument("--cluster", help="Enable cluster scan.")
    parser.add_argument(
        "-n",
        "--nreplicas",
        type=positive_int,
        help="Define the number of input replicas.",
    )
    parser.add_argument(
        "--pplot", type=positive_int, help="Define the number of output replicas."
    )
    parser.add_argument("--flavors", type=flavour_int, help="Choose the flavours.")
    # parser.add_argument('--timeline', action='store_true')
    args = parser.parse_args()

    # Sanitize the arguments
    # Check that the flavours are allowed

    # check the runcard
    if not os.path.isfile(args.runcard):
        raise ValueError("Invalid runcard: not a file.")
    if args.force:
        logger.warning("Running with --force option will overwrite existing model.\n\n")

    # prepare the output folder
    if not os.path.exists(args.output):
        os.mkdir(args.output)
    elif args.force:
        shutil.rmtree(args.output)
        os.mkdir(args.output)
    else:
        raise Exception(f'{args.output} already exists, use "--force" to overwrite.')

    return args


def main():
    """
    This is the Main controller.
    It controlls the input parameters.
    """

    args = argument_parser()

    out = args.output.strip("/")

    # copy runcard to output folder
    shutil.copyfile(args.runcard, f"{out}/input-runcard.json")

    logger.info("[+] Loading runcard")
    hps = load_yaml(args.runcard)
    hps["verbose"] = False
    hps["save_output"] = out

    # Define the number of input replicas
    if args.nreplicas is None:
        hps["input_replicas"] = 1
    else:
        hps["input_replicas"] = args.nreplicas

    # Define the number of output replicas
    if args.pplot is None:
        hps["out_replicas"] = hps["input_replicas"]
    else:
        hps["out_replicas"] = args.pplot

    # Check the input flavor
    if args.flavors is None:
        hps["fl"] = 1  # Take the u quark as a default
    else:
        hps["fl"] = args.flavors

    ## One-time generation of input PDF ##
    # Load the x_grid
    x_pdf = xnodes().build_xgrid()
    # Choose Q^2 value
    Q_value = 1.7874388
    # Generate PDF
    logger.info("[+] Loading input PDFs...")
    pdf = input_pdfs(
        hps["pdf"],
        x_pdf,
        hps["input_replicas"],
        Q_value,
        hps["fl"]
    ).build_pdf()


    # If hyperscan is set true
    if args.hyperopt:
        hps["scan"] = True
        if args.nreplicas < 90:
            raise Exception('Choose number of replicas to be greater than 90.')
        def fn_hyper_train(params):
            return hyper_train(params, x_pdf, pdf)
        # Run hyper scan
        hps = run_hyperparameter_scan(
            fn_hyper_train,
            hps,
            args.hyperopt,
            args.cluster,
            out
        )

    # Run the best Model and output log
    hps["scan"] = False
    hps["verbose"] = True

    loss = hyper_train(hps, x_pdf, pdf)
