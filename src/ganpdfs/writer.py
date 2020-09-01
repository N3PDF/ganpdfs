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
    lha = fake_replica.T

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
