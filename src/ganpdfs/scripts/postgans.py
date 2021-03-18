"""
PostFit for GANs replicas
"""

import os
import re
import shutil
import lhapdf
import logging
import pathlib
import argparse

from subprocess import PIPE
from subprocess import Popen
from rich.table import Table
from rich.console import Console

from validphys import lhio
from validphys.core import PDF


console = Console()
logger = logging.getLogger(__name__)


def get_lhapdf_dir():
    """`get_lhapdf_dir` retrieves the path where the LHAPDF
    data are located.
    """
    lhapdf_dir = Popen(["lhapdf-config", "--datadir"], stdout=PIPE)
    lhapdf_pathdir, _ = lhapdf_dir.communicate()
    lhapdf_pathdir = lhapdf_pathdir.decode("utf-8")
    lhapdf_pathdir = lhapdf_pathdir.replace("\n", "")
    return lhapdf_pathdir


def create_symlink(source, destintation):
    """`create_symlink` creates a symbolic link bewteen the data
    genereted in the run folder and the LHAPDF `datadir` folder.

    Parameters
    ----------
    source : str
        Path indicating the source file to be linked to.
    destintation : str
        Paht indicating where the linked file is located.
    """
    return os.symlink(source, destintation)


def replace_num_members(info_file, nbprior, totrep):
    """Replace the value of the `NumMembers` key in PDF set .info file
    by the new total number of MC replicas.

    Parameters
    ----------
    info_file : str
        file containing the information of the PDF
    nbprior :
        number of prior MC replicas
    totrep :
        total number of the new MC of replicas
    """
    subst = f"NumMembers: {totrep+1}"
    pattern = f"NumMembers: {nbprior}"
    file_handle = open(info_file, 'r')
    file_string = file_handle.read()
    file_handle.close()
    file_string = (re.sub(pattern, subst, file_string))
    file_handle = open(info_file, 'w')
    file_handle.write(file_string)
    file_handle.close()


def postgans(pdf_name, ntotal_rep, check=False):
    """`postgans` Links the generated PDF grids in the run directory
    to the correct directory where the LHAPDF data are located. This
    is done by first requesting the path to the LHAPDF `datadir` using
    `get_lhapdf_dir` and then creates a folder where the enhanced PDFs
    replicas will be placed. If the latter already exists, then removes
    it. The linking is done in two steps:

        (1) first, link all the prior replicas
        (2) and then link all the generated replicas

    The `.info` file of the new enhanced PDF is just a copy of the prior.
    However, it has to be checked and assured beforehand that this is
    exactly the same as the `.info` file in the generated replicas.

    Parameters
    ----------
    pdf_name : str
        Prior PDF name
    ntotal_rep : int
        Total number of replicas
    check : bool
        Choose to whether or not make a check by importing the
        enhanced PDF.
    """

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("POSTGANS", justify="center", width=40)
    table.add_row("Link the generated pdfs to LHAPDF datadir.")
    console.print(table)

    # Access paths
    lhapdf_dir = get_lhapdf_dir()
    lhapdf_pth = pathlib.Path(lhapdf_dir).absolute()
    prior_path = lhapdf_pth / f"{pdf_name}"
    gnpdf_path = lhapdf_pth / f"{pdf_name}_enhanced"

    # Create pdf folder if non-existing and remove it if
    # already existing and replace by a new one.
    if gnpdf_path.is_dir():
        logger.warning(f"{gnpdf_path} already exists and will be removed.")
        shutil.rmtree(gnpdf_path)
    gnpdf_path.mkdir(exist_ok=True)

    # Get GANs output grid
    gan_folder = f"{pdf_name}_enhanced"
    gans_grids = pathlib.Path().absolute() / f"{gan_folder}" / "nnfit"

    # Count the number of replicas in the prior folder
    nbfiles_prior = os.listdir(prior_path)
    nbreplicas_prior = len(nbfiles_prior) - 1

    pdf_info = f"{pdf_name}.info"
    gdf_info = f"{pdf_name}_enhanced.info"
    prior_info = os.path.join(prior_path, pdf_info)
    gnpdf_info = os.path.join(gnpdf_path, gdf_info)
    # Copy file to LHAPDF datadir
    shutil.copy(prior_info, gnpdf_info)
    # Replace NumMembers entry
    replace_num_members(gnpdf_info, nbreplicas_prior, ntotal_rep)

    # Loop over the replicas
    for rep in range(1, ntotal_rep + 1):
        gnpdf = f"{pdf_name}_enhanced_{rep:04}.dat"
        gnpdf_file = os.path.join(gnpdf_path, gnpdf)
        gans_file = gans_grids / f"replica_{rep}"
        gans_file = os.path.join(gans_file, f"{gan_folder}.dat")
        create_symlink(gans_file, gnpdf_file)

    # Compute Replica 0
    lhapdf.pathsPrepend(str(lhapdf_pth))
    generated_pdf = PDF(pdf_name + "_enhanced")
    lhio.generate_replica0(generated_pdf)

    if check:
        try:
            console.print("\n• Try loading enhanced Replica 0.", style="bold blue")
            gen_name = pdf_name + "_enhanced"
            lhapdf.mkPDF(gen_name, 0)
        except RuntimeError as exp:
            console.print(
                    f"\n• {pdf_name} might be corrupted according to {exp}.",
                    style="bold blue"
            )
    console.print("\n• Symbolink link added successfully.", style="bold blue")


def arg_parser():
    """Paree input PDF"""
    parser = argparse.ArgumentParser(description="Create set into LHAPDF")
    parser.add_argument("--pdf", help="PDF name", required=True)
    parser.add_argument("--nenhanced", help="Size of enhanced", required=True)
    argument = parser.parse_args()
    return argument


def main():
    args = arg_parser()
    pdf_name = args.pdf
    enhanced_size = int(args.nenhanced)

    postgans(pdf_name, enhanced_size, check=True)


if __name__ == "__main__":
    main()
