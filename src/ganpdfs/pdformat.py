import os
import math
import numpy as np
import pdfflow

from subprocess import PIPE
from subprocess import Popen

from validphys.pdfbases import fitbasis_to_NN31IC


class XNodes:

    """Custom x-grid. This might be useful in case there are some
    x-grid format that maximizes the training of the GANs and get
    better performances."""

    def __init__(self):
        self.x_nodes = [
            1.0974987654930569e-05,
            1.5922827933410941e-05,
            2.3101297000831580e-05,
            3.3516026509388410e-05,
            4.8626015800653536e-05,
            7.0548023107186455e-05,
            1.0235310218990269e-04,
            1.4849682622544667e-04,
            2.1544346900318823e-04,
            3.1257158496882353e-04,
            4.5348785081285824e-04,
            6.5793322465756835e-04,
            9.5454845666183481e-04,
            1.3848863713938717e-03,
            2.0092330025650459e-03,
            2.9150530628251760e-03,
            4.2292428743894986e-03,
            6.1359072734131761e-03,
            8.9021508544503934e-03,
            1.2915496650148829e-02,
            1.8738174228603830e-02,
            2.7185882427329403e-02,
            3.9442060594376556e-02,
            5.7223676593502207e-02,
            8.3021756813197525e-02,
            0.10000000000000001,
            0.11836734693877551,
            0.13673469387755102,
            0.15510204081632653,
            0.17346938775510204,
            0.19183673469387758,
            0.21020408163265308,
            0.22857142857142856,
            0.24693877551020407,
            0.26530612244897961,
            0.28367346938775512,
            0.30204081632653063,
            0.32040816326530613,
            0.33877551020408170,
            0.35714285714285710,
            0.37551020408163271,
            0.39387755102040811,
            0.41224489795918373,
            0.43061224489795924,
            0.44897959183673475,
            0.46734693877551026,
            0.48571428571428565,
            0.50408163265306127,
            0.52244897959183678,
            0.54081632653061229,
            0.55918367346938780,
            0.57755102040816331,
            0.59591836734693870,
            0.61428571428571421,
            0.63265306122448983,
            0.65102040816326534,
            0.66938775510204085,
            0.68775510204081625,
            0.70612244897959175,
            0.72448979591836737,
            0.74285714285714288,
            0.76122448979591839,
            0.77959183673469379,
            0.79795918367346941,
            0.81632653061224492,
            0.83469387755102042,
            0.85306122448979593,
            0.87142857142857133,
            0.88979591836734695,
            0.90816326530612246,
        ]

    def build_xgrid(self):
        """build_xgrid.
        """

        x_grid = np.array(self.x_nodes)
        return x_grid


class InputPDFs:

    """Instantiate the computation of the input/prior PDF grid.

    Parameters
    ---------
    pdf: str
        name of the input/prior PDF set
    q_value : float
        initiale value of the scale at which the PDF grid is
        constructed
    """

    def __init__(self, pdf_name, q_value):

        self.nf = 3
        self.q_value = q_value
        self.pdf_name = pdf_name

    def extract_xgrid(self):
        """Extract the x-grid format from the input PDF file. The nice
        thing about this that there will not be a need for interpolation 
        later on.

        Returns
        -------
        np.array of shape (size,)
            containing x-grid points
        """

        lhapdf_dir = Popen(["lhapdf-config", "--datadir"], stdout=PIPE)
        pdf_pathdir, _ = lhapdf_dir.communicate()
        pdf_pathdir = pdf_pathdir.decode("utf-8")
        pdf_pathdir = pdf_pathdir.replace("\n", "")
        replica_zero = self.pdf_name + "_0000.dat"
        file_path = os.path.join(pdf_pathdir, self.pdf_name, replica_zero)
        w = open(file_path, "r")
        # Skip the head
        for _ in range(0, 10):
            if "--" in w.readline():
                break
        # Fetch x-grid
        lhapdf_info = w.readline()
        lhapdf_grid = lhapdf_info.replace("\n", "")
        lhapdf_grid = [float(i) for i in lhapdf_grid.split()]
        return np.array(lhapdf_grid)

    def custom_xgrid(self, minval, maxval, nbpoints, grid_type="linear"):
        """Construct a custom xgrid by taking the smallest and largest
        value of the LHAPDF grid and sample the points equally spaced.

        Parameters
        ----------
        minval: float
            Minimum x value
        maxval: float
            Maximum x value
        nbpoints: int
            Number of xgrid points

        Returns
        -------
        np.array(float)
            x-grid array
        """

        if grid_type == "linear":
            return np.linspace(minval, maxval, num=nbpoints, endpoint=False)
        else:
            xgrid = np.logspace(
                        math.log(minval),
                        math.log(maxval),
                        num=nbpoints,
                        base=math.exp(1)
                    )
            return xgrid

    def build_pdf(self, xgrid):
        """Construct the input PDFs grid based on the number of input
        replicas. The PDF grid is represented in the evolution basis
        following the structure.
        evolbasis = {"sigma","g","v","v3","v8","t3","t8","t15"}

        The  following returns a multi-dimensional array that has
        the following shape (nb_replicas, nb_flavours, size_xgrid)

        Parameters
        ----------
        xgrid : np.array
            array of x-grid

        Returns
        -------
        np.array(float) of shape (nb_replicas, nb_flavours, size_xgrid)
        """

        # Sample pdf with all the flavors
        # nb total flavors = 2 * nf + 2
        # Evolution basis
        flav_info = [
            {"fl": "u"},
            {"fl": "ubar"},
            {"fl": "d"},
            {"fl": "dbar"},
            {"fl": "s"},
            {"fl": "sbar"},
            {"fl": "c"},
            {"fl": "g"},
        ]
        # Flavours PIDs
        flav_list = {
             "u": 2,
             "ubar": -2,
             "d": 1,
             "dbar": -1,
             "s": 3,
             "sbar": -3,
             "c": 4,
             "g": 0,
        }

        # Since we need to call the grid just once, it doesn't make sense to compile it
        pdfflow.run_eager(True)
        pflow = pdfflow.mkPDFs(self.pdf_name)
        # Read the pdffgrid with pflow
        pids = list(flav_list.values())
        qs = np.ones_like(xgrid)*self.q_value
        inpdf_full = pflow.py_xfxQ2(pids, xgrid, qs)
        pdfflow.run_eager(False)

        # Remove member 0
        inpdf = inpdf_full[1:]

        # Compute Rotation Matrix from Flavour basis to
        rotmat = fitbasis_to_NN31IC(flav_info, 'FLAVOUR')
        evolbasis = np.tensordot(inpdf, rotmat, (2, 0))
        evolbasis = np.transpose(evolbasis, axes=[0, 2, 1])
        return evolbasis
