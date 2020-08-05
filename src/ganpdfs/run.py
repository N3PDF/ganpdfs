"""
    This file contains the main driver of the wganpdfs code
"""
import os
import shutil
import logging
import argparse

from ganpdfs.pdformat import XNodes
from ganpdfs.pdformat import InputPDFs
from ganpdfs.hyperscan import load_yaml
from ganpdfs.hyperscan import hyper_train
from ganpdfs.hyperscan import run_hyperparameter_scan

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def positive_int(value):
    """ Checks that a given number is positive """
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(
            f"Negative values are not allowed, received: {value}"
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
    # parser.add_argument('--timeline', action='store_true')
    args = parser.parse_args()

    # check the runcard
    if not os.path.isfile(args.runcard):
        raise ValueError("Invalid runcard: not a file.")
    if args.force:
        logger.warning("Running with --force option will overwrite existing model.")

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

    logger.info("Loading runcard")
    hps = load_yaml(args.runcard)
    # Turn off verbose during hyperopt
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

    # One-time generation of input PDF
    # Choose number of flavours
    nf = 3
    # Choose Q^2 value
    q_value = 1.7874388
    # Load the x_grid
    x_grid = XNodes().build_xgrid()
    # Generate PDF
    logger.info("Loading input PDFs...")
    pdf = InputPDFs(
        hps["pdf"], x_grid, q_value, nf
    ).build_pdf()
    # Size flavours
    hps["flsize"] = pdf.shape[1]
    # Size x-grid
    hps["ngsize"] = pdf.shape[1]

    # If hyperscan is set true
    if args.hyperopt:
        hps["scan"] = True
        if args.nreplicas < 90:
            raise Exception("Choose number of replicas to be greater than 90.")

        def fn_hyper_train(params):
            return hyper_train(params, x_grid, pdf)

        # Run hyper scan
        hps = run_hyperparameter_scan(
            fn_hyper_train, hps, args.hyperopt, args.cluster, out
        )

    # Run the best Model and output log
    hps["scan"] = False
    hps["verbose"] = True

    loss = hyper_train(hps, x_grid, pdf)
