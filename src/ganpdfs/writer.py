"""
Module conataning functions that write down the grid from the outputs of the
GANs. This generates the same output file as n3fit.

For futher details, refer to https://github.com/NNPDF/nnpdf/blob/
8eb094f04c73b994502c1cf0f2592f5541e9c284/n3fit/src/n3fit/io/writer.py
"""

import os
import numpy as np
from reportengine.compat import yaml


class WriterWrapper:
    def __init__(self, outputname, fake_pdf, xgrid, replica_ind, qscale):
        """__init__.

        Parameters
        ----------
        replica_number :
            replica_number
        qscale :
            qscale
        """

        self.xgrid = xgrid
        self.qscale = qscale
        self.fake_pdf = fake_pdf
        self.outputname = outputname
        self.replica_index = replica_ind

    def write_data(self, replica_path):
        """write_data.

        Parameters
        ----------
        replica_path_set :
            replica_path_set
        outputname :
            outputname
        """

        os.makedirs(replica_path, exist_ok=True)

        # export PDF grid to file
        storegrid(
            self.fake_pdf,
            self.xgrid,
            self.qscale,
            self.outputname,
            self.replica_index,
            replica_path
        )


def evln2lha(evln):
    """evln2lha.

    Parameters
    ----------
    evln :
        evln
    """

    # evln Basis:
    # ----------
    # {"PHT","SNG","GLU","VAL","V03","V08","V15",
    # "V24","V35","T03","T08","T15","T24","T35"};
    # lha Basis:
    # ---------
    # {"TBAR","BBAR","CBAR","SBAR","UBAR","DBAR",
    # "GLUON","D","U","S","C","B","T","PHT"}
    lha = np.zeros(evln.shape)
    lha[13] = evln[0]

    lha[6] = evln[2]

    lha[8] = (
        10 * evln[1]
        + 30 * evln[9]
        + 10 * evln[10]
        + 5 * evln[11]
        + 3 * evln[12]
        + 2 * evln[13]
        + 10 * evln[3]
        + 30 * evln[4]
        + 10 * evln[5]
        + 5 * evln[6]
        + 3 * evln[7]
        + 2 * evln[8]
    ) / 120

    lha[4] = (
        10 * evln[1]
        + 30 * evln[9]
        + 10 * evln[10]
        + 5 * evln[11]
        + 3 * evln[12]
        + 2 * evln[13]
        - 10 * evln[3]
        - 30 * evln[4]
        - 10 * evln[5]
        - 5 * evln[6]
        - 3 * evln[7]
        - 2 * evln[8]
    ) / 120

    lha[7] = (
        10 * evln[1]
        - 30 * evln[9]
        + 10 * evln[10]
        + 5 * evln[11]
        + 3 * evln[12]
        + 2 * evln[13]
        + 10 * evln[3]
        - 30 * evln[4]
        + 10 * evln[5]
        + 5 * evln[6]
        + 3 * evln[7]
        + 2 * evln[8]
    ) / 120

    lha[5] = (
        10 * evln[1]
        - 30 * evln[9]
        + 10 * evln[10]
        + 5 * evln[11]
        + 3 * evln[12]
        + 2 * evln[13]
        - 10 * evln[3]
        + 30 * evln[4]
        - 10 * evln[5]
        - 5 * evln[6]
        - 3 * evln[7]
        - 2 * evln[8]
    ) / 120

    lha[9] = (
        10 * evln[1]
        - 20 * evln[10]
        + 5 * evln[11]
        + 3 * evln[12]
        + 2 * evln[13]
        + 10 * evln[3]
        - 20 * evln[5]
        + 5 * evln[6]
        + 3 * evln[7]
        + 2 * evln[8]
    ) / 120

    lha[3] = (
        10 * evln[1]
        - 20 * evln[10]
        + 5 * evln[11]
        + 3 * evln[12]
        + 2 * evln[13]
        - 10 * evln[3]
        + 20 * evln[5]
        - 5 * evln[6]
        - 3 * evln[7]
        - 2 * evln[8]
    ) / 120

    lha[10] = (
        10 * evln[1]
        - 15 * evln[11]
        + 3 * evln[12]
        + 2 * evln[13]
        + 10 * evln[3]
        - 15 * evln[6]
        + 3 * evln[7]
        + 2 * evln[8]
    ) / 120

    lha[2] = (
        10 * evln[1]
        - 15 * evln[11]
        + 3 * evln[12]
        + 2 * evln[13]
        - 10 * evln[3]
        + 15 * evln[6]
        - 3 * evln[7]
        - 2 * evln[8]
    ) / 120

    lha[11] = (
        5 * evln[1]
        - 6 * evln[12]
        + evln[13]
        + 5 * evln[3]
        - 6 * evln[7]
        + evln[8]
    ) / 60

    lha[1] = (
        5 * evln[1]
        - 6 * evln[12]
        + evln[13]
        - 5 * evln[3]
        + 6 * evln[7]
        - evln[8]
    ) / 60

    lha[12] = (evln[1] - evln[13] + evln[3] - evln[8]) / 12

    lha[0] = (evln[1] - evln[13] - evln[3] + evln[8]) / 12
    return lha


def storegrid(fake_replica, xgrid, qscale, outputname, replica_ind, replica_path):
    """storegrid.

    Parameters
    ----------
    fake_replica :
        fake_replica
    xgrid :
        xgrid
    qscale :
        qscale
    outputname :
        outputname
    replica_ind :
        replica_ind
    replica_path :
        replica_path
    """

    # TODO: Check why q^2
    lha = evln2lha(fake_replica).T

    data = {
        "replica": replica_ind,
        "q20": qscale,
        "xgrid": xgrid.T.tolist(),
        "labels": [
            "TBAR",
            "BBAR",
            "CBAR",
            "SBAR",
            "UBAR",
            "DBAR",
            "GLUON",
            "D",
            "U",
            "S",
            "C",
            "B",
            "T",
            "PHT",
        ],
        "pdfgrid": lha.tolist(),
    }

    with open(f"{replica_path}/{outputname}.exportgrid", "w") as fs:
        yaml.dump(data, fs)
