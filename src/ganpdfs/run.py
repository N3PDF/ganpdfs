# This file contains the main driver of the wganpdfs code

import os
import shutil
import logging
import argparse
import numpy as np

# Silent tf for the time being
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from ganpdfs.pdformat import XNodes
from ganpdfs.pdformat import InputPDFs
from ganpdfs.hyperscan import load_yaml
from ganpdfs.hyperscan import hyper_train
from ganpdfs.hyperscan import run_hyperparameter_scan

logging.basicConfig(
        level=logging.INFO,
        format="\033[0;32m[%(levelname)s]\033[97m %(message)s"
    )
logger = logging.getLogger(__name__)

# Random Seeds
np.random.seed(0)
tf.random.set_seed(0)


def splash():
    info = """\033[34m
+-------------------------------------------------------------------------+
|𝖌𝖆𝖓𝖕𝖉𝖋𝖘:                                                                 |
|-------                                                                  |
|Generative Adversarial Neural Networks (GANs) for PDF replicas.          |
|https://n3pdf.github.io/ganpdfs/                                         |
|© N3PDF                                                                  |
+-------------------------------------------------------------------------+ 
           """
    print(info + '\033[0m \033[97m')


def positive_int(value):
    """Checks that a given number is positive.
    """

    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(
            f"Negative values are not allowed, received: {value}"
        )
    return ivalue


def argument_parser():
    """Parse the input arguments for wganpdfs.
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
        "-k",
        "--fake",
        type=positive_int,
        help="Define the number of output replicas."
    )
    # parser.add_argument('--timeline', action='store_true')
    args = parser.parse_args()

    # check the runcard
    if not os.path.isfile(args.runcard):
        raise ValueError("Invalid runcard: not a file.")
    if args.force:
        logger.warning("Running with --force will overwrite existing model.")

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
    """Main controller from which the main parameters are set and defined.
    """
    splash()
    args = argument_parser()
    out = args.output.strip("/")

    # Copy runcard to output folder
    shutil.copyfile(args.runcard, f"{out}/input-runcard.json")

    logger.info("Loading runcard.")
    hps = load_yaml(args.runcard)
    hps["save_output"] = out

    # Prepare Grids
    # One-time Generation
    nf = hps.get("nf", 6)            # Choose Number of flavours
    qvalue = hps.get("q", 1.65)      # Choose value of Initial

    # Generate PDF grids
    logger.info("Loading input PDFs.")
    init_pdf = InputPDFs(hps["pdf"], qvalue, nf)
    # Load the x-Grid
    # Choose the LHAPDF x-grid by default
    hps["pdfgrid"] = init_pdf.extract_xgrid()
    if hps["x_grid"] == "custom":
        # x_grid = XNodes().build_xgrid()
        mn = hps["pdfgrid"][0]
        mx = hps["pdfgrid"][-1]
        x_grid = init_pdf.custom_xgrid(mn, mx, 60)
    elif hps["x_grid"] == "lhapdf":
        x_grid = hps["pdfgrid"]
    else:
        raise ValueError("{} is not a valid grid".format(hps["x_grid"]))
    pdf = init_pdf.build_pdf(x_grid)

    # Define the number of input replicas
    if args.nreplicas is None:
        hps["input_replicas"] = pdf.shape[0]
    else:
        hps["input_replicas"] = args.nreplicas

    # Define the number of output replicas
    if args.fake is None:
        hps["out_replicas"] = hps["input_replicas"]
    else:
        # Generate the missing replicas
        hps["out_replicas"] = args.fake - hps["input_replicas"]

    # If hyperscan is set true
    if args.hyperopt:
        hps["scan"] = True

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
