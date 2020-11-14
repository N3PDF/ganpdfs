import os
import math
import lhapdf
import numpy as np

from subprocess import PIPE
from subprocess import Popen


class XNodes:

    """Custom x-grid. This might be useful in case there are some
    x-grid format that maximizes the training of the GANs and get
    better performances."""

    def __init__(self):
        self.x_nodes = [
            1.0000000000000001e-09,
            1.4508287784959398e-09,
            2.1049041445120207e-09,
            3.0538555088334157e-09,
            4.4306214575838816e-09,
            6.4280731172843209e-09,
            9.3260334688321995e-09,
            1.3530477745798068e-08,
            1.9630406500402714e-08,
            2.8480358684358022e-08,
            4.1320124001153370e-08,
            5.9948425031894094e-08,
            8.6974900261778356e-08,
            1.2618568830660210e-07,
            1.8307382802953678e-07,
            2.6560877829466870e-07,
            3.8535285937105315e-07,
            5.5908101825122239e-07,
            8.1113083078968731e-07,
            1.1768119524349981e-06,
            1.7073526474706905e-06,
            2.4770763559917115e-06,
            3.5938136638046262e-06,
            5.2140082879996849e-06,
            7.5646332755462914e-06,
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
            0.92653061224489797,
            0.94489795918367347,
            0.96326530612244898,
            0.98163265306122449,
            1.0000000000000000
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
    nf: int
        total number of flavors
    """

    def __init__(self, pdf_name, q_value, nf):

        self.nf = nf
        self.q_value = q_value
        self.pdf_name = pdf_name
        self.lhpdf = lhapdf.mkPDFs(pdf_name)

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

    def custom_xgrid(self, nbpoints):
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
        #TODO: Move this function!

        logx = int(nbpoints / 2)
        linx = int(nbpoints - logx)
        xgrid_log = np.logspace(-9, -1, logx + 1)
        xgrid_lin = np.linspace(0.1, 1, linx)
        xgrid = np.concatenate([xgrid_log[:-1], xgrid_lin]).reshape(nbpoints, 1)
        return xgrid

    def build_pdf(self, xgrid):
        """Construct the input PDFs based on the number of input
        replicas and the number of flavors.

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
        # nb total flavors = 2 * nf + 1
        xgrid_size = xgrid.shape[0]
        pdf_size = len(self.lhpdf) - 1
        # Construct a grid of zeros to store the results
        inpdf = np.zeros((pdf_size, 2 * self.nf + 2, xgrid_size))

        for p in range(pdf_size):
            for f in range(-self.nf, self.nf + 2):
                for x in range(xgrid_size):
                    inpdf[p][f + self.nf][x] = self.lhpdf[p + 1].xfxQ(
                        f, xgrid[x], self.q_value
                    )
        return inpdf

    def lhaPDF_grids(self):
        """lhaPDF_grids.
        """
        xgrid = self.extract_xgrid()
        return self.build_pdf(xgrid)
