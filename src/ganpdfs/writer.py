"""
Module conataning functions that write down the grid from the outputs of
the GANs. This generates the same output file as n3fit.

For futher details, refer to https://github.com/NNPDF/nnpdf/blob/
8eb094f04c73b994502c1cf0f2592f5541e9c284/n3fit/src/n3fit/io/writer.py
"""

import os
import numpy as np
from reportengine.compat import yaml


class WriterWrapper:
    """Class that writes the generated grid into a file. This has exactly
    the same format as the N3FIT output ins such a way that the results can
    be evolved using evolven3fit.

    Parameters
    ----------
    outputname : str
        name of the output folder
    fake_pdf : np.array(float)
        array of fake datased
    xgrid : np.array(float)
        array of x-grid
    replica_ind : int
        index of the given replica
    qscale : float
        value of the initial scale
    """

    def __init__(self, outputname, fake_pdf, xgrid, replica_ind, qscale):

        self.xgrid = xgrid
        self.qscale = qscale
        self.fake_pdf = fake_pdf
        self.outputname = outputname
        self.replica_index = replica_ind

    def write_data(self, replica_path):
        """Write the data into an `.exportgrid` file using the `storegrid`
        method.

        Parameters
        ----------
        replica_path_set : str
            path where the replicas are stored
        outputname : str
            name of the output folder
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
    """Store the results into a grid in the same format as the output of N3FIT.

    Parameters
    ----------
    fake_replica : np.array
        array of fake dataset
    xgrid : np.array(float)
        array of points defining the x-grid
    qscale : float
        initial scale at which the computation was done
    outputname : str
        name of the output folder
    replica_ind : int
        index of the given replica
    replica_path : str
        path where the replicas are stored
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
