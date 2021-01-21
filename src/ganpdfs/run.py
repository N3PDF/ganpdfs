# This file contains the main driver of the ganpdfs code

import os
import shutil
import logging
import argparse
import numpy as np

# Set tensorflow log level to error only
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

import tensorflow as tf
from rich.table import Table
from rich.style import Style
from rich.console import Console
from ganpdfs.pdformat import XNodes
from ganpdfs.pdformat import InputPDFs
from ganpdfs.hyperscan import load_yaml
from ganpdfs.hyperscan import hyper_train
from ganpdfs.hyperscan import run_hyperparameter_scan

console = Console()
logging.basicConfig(
        level=logging.INFO,
        format="\033[0;32m[%(levelname)s]\033[97m %(message)s",
    )
logger = logging.getLogger(__name__)

# Random Seeds
tf.random.set_seed(0)


NF = 6      # Number of flavours
Q0 = 1.65   # Initial energy scale


def splash():
    """Splash information."""

    style = Style(color="blue")
    logo = Table(show_header=True, header_style="bold blue", style=style)
    logo.add_column("𝖌𝖆𝖓𝖕𝖉𝖋𝖘", justify="center", width=80)
    logo.add_row("[bold blue]Generative Adversarial Neural Networks (GANs) for PDF replicas.")
    logo.add_row("[bold blue]https://n3pdf.github.io/ganpdfs/")
    logo.add_row("[bold blue]© N3PDF 2021")
    logo.add_row("[bold blue]Authors: Stefano Carrazza, Juan E. Cruz-Martinez, Tanjona R. Rabemananjara")
    console.print(logo)


def posint(value):
    """Checks that a given number is positive."""

    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"Negative values are not allowed: {value}")
    return ivalue


def argument_parser():
    """Parse the input arguments for wganpdfs.
    """
    # read command line arguments
    parser = argparse.ArgumentParser(description="Train a PDF GAN.")
    parser.add_argument("-f", "--force", action="store_true")
    parser.add_argument("runcard", help="A json file with the setup.")
    parser.add_argument("-c", "--cluster", help="Enable cluster scan.")
    parser.add_argument("-o", "--output", help="The output folder.", required=True)
    parser.add_argument("-s", "--hyperopt", type=int, help="Enable hyperopt scan.")
    parser.add_argument("-k", "--fake", type=posint, help="Number of output replicas.")
    parser.add_argument("-n", "--nreplicas", type=posint, help="Number of input replicas.")
    args = parser.parse_args()

    # check the runcard
    if not os.path.isfile(args.runcard):
        raise ValueError("Invalid runcard: not a file.")
    if args.force:
        logger.warning("Running with --force will overwrite existing results.")

    # prepare the output folder
    if not os.path.exists(args.output):
        os.mkdir(args.output)
    elif args.force:
        shutil.rmtree(args.output)
        os.mkdir(args.output)
    else:
        raise Exception(f'{args.output} exists, use "--force" to overwrite.')
    return args


def main():
    """Main controller from which the main parameters are set and defined.
    """
    splash()
    args = argument_parser()
    out = args.output.strip("/")

    # Copy runcard to output folder
    shutil.copyfile(args.runcard, f"{out}/input-runcard.json")

    hps = load_yaml(args.runcard)
    hps["save_output"] = out
    hps["tot_replicas"] = args.fake
    nf = hps.get("nf", NF)
    qvalue = hps.get("q", Q0)

    # Generate PDF grids (one time generation)
    console.print("\n• Computing PDF grids with the following parameters:", style="bold blue")
    init_pdf = InputPDFs(hps["pdf"], qvalue, nf)
    # Load the x-Grid
    # Choose the LHAPDF x-grid by default
    hps["pdfgrid"] = init_pdf.extract_xgrid()
    if hps["x_grid"] == "lhapdf":
        xgrid = hps["pdfgrid"]
    elif hps["x_grid"] == "custom":
        xgrid = XNodes().build_xgrid()
    elif hps["x_grid"] == "standard":
        xgrid = init_pdf.custom_xgrid()
    else:
        raise ValueError("{} is not a valid grid".format(hps["x_grid"]))

    # Print summary table of PDF grids
    summary = Table(show_header=True, header_style="bold blue")
    summary.add_column("Parameters", justify="left", width=30)
    summary.add_column("Description", justify="center", width=40)
    summary.add_row("Prior PDF set", f"{hps['pdf']}")
    summary.add_row("Input energy Q0", f"{Q0} GeV")
    summary.add_row(
        "x-grid size",
        f"{xgrid.shape[0]} points, x=({xgrid[0]:.4e}, {xgrid[-1]:.4e})"
    )
    console.print(summary)

    pdf = init_pdf.build_pdf(xgrid)
    pdf_lhapdf = init_pdf.lhaPDF_grids()
    pdfs = (pdf, pdf_lhapdf)

    # Define the number of input replicas
    hps["input_replicas"] = pdf.shape[0] if args.nreplicas is None else args.nreplicas
    hps["out_replicas"] = hps["input_replicas"] if args.fake is None else \
            args.fake - hps["input_replicas"]

    # If hyperscan is True
    if args.hyperopt:
        hps["scan"] = True  # Enable hyperscan

        def fn_hyper_train(params):
            return hyper_train(params, xgrid, pdfs)

        # Run hyper scan
        hps = run_hyperparameter_scan(
            fn_hyper_train, hps, args.hyperopt, args.cluster, out
        )

    # Run the best Model and output logs
    hps["scan"] = False
    hps["verbose"] = True
    loss = hyper_train(hps, xgrid, pdfs)
