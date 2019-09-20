import lhapdf
import numpy as np
from random import sample

class xnodes(object):

    """
    Construct the x-grid as in NNPDF
    """

    def __init__(self):
        self.x_nodes = [1.0053959e-04, 1.0915639e-04, 1.1142429e-04, 1.4267978e-04, 1.8270270e-04,
                        2.3395241e-04, 2.9957810e-04, 3.8361238e-04, 4.9121901e-04, 6.2901024e-04,
                        8.0545312e-04, 1.0313898e-03, 1.3207036e-03, 1.6911725e-03, 2.1655612e-03,
                        2.7730201e-03, 3.5508765e-03, 4.5469285e-03, 5.8223817e-03, 7.4556107e-03,
                        9.5469747e-03, 1.2224985e-02, 1.5654200e-02, 2.0045340e-02, 2.5668233e-02,
                        3.2868397e-02, 4.2088270e-02, 5.3894398e-02, 6.9012248e-02, 8.8370787e-02,
                        1.0000000e-01, 1.1216216e-01, 1.2432432e-01, 1.3648649e-01, 1.4864865e-01,
                        1.6081081e-01, 1.7297297e-01, 1.8513514e-01, 1.9729730e-01, 2.0945946e-01,
                        2.2162162e-01, 2.3378378e-01, 2.4594595e-01, 2.5810811e-01, 2.7027027e-01,
                        2.8243243e-01, 2.9459459e-01, 3.0675676e-01, 3.1891892e-01, 3.3108108e-01,
                        3.4324324e-01, 3.5540541e-01, 3.6756757e-01, 3.7972973e-01, 3.9189189e-01,
                        4.0405405e-01, 4.1621622e-01, 4.2837838e-01, 4.4054054e-01, 4.5270270e-01,
                        4.6486486e-01, 4.7702703e-01, 4.8918919e-01, 5.0135135e-01, 5.1351351e-01,
                        5.2567568e-01, 5.3783784e-01, 5.5000000e-01, 5.6216216e-01, 5.7432432e-01,
                        5.8648649e-01, 5.9864865e-01, 6.1081081e-01, 6.2297297e-01, 6.3513514e-01,
                        6.4729730e-01, 6.5945946e-01, 6.7162162e-01, 6.8378378e-01, 6.9594595e-01,
                        7.0810811e-01, 7.2027027e-01, 7.3243243e-01, 7.4459459e-01, 7.5675676e-01,
                        7.6891892e-01, 7.8108108e-01, 7.9324324e-01, 8.0540541e-01, 8.1756757e-01,
                        8.2972973e-01, 8.4189189e-01, 8.5405405e-01, 8.6621622e-01, 8.7837838e-01,
                        8.9054054e-01, 9.0270270e-01, 9.1486486e-01, 9.2702703e-01, 9.3918919e-01,
                        9.5135135e-01, 9.6351351e-01, 9.7567568e-01, 9.8783784e-01, 1.0000000e+00]


    def build_xgrid(self):
        x_grid = np.array(self.x_nodes)
        return x_grid


class input_pdfs(object):

    """
    This formats the input replicas.
    It returns a multi-dimensional array with shape
    (nb_flavors, nb_pdf_members, xgrid_size)
    """

    def __init__(self, pdf_name, x_pdf, nb_replicas, Q_value, flavors):
        self.x_pdf    = x_pdf
        self.Q_value  = Q_value
        self.pdf_name = pdf_name
        self.flavors  = flavors
        self.nb_replicas = nb_replicas

    def compute_central(self):
        pdfx = lhapdf.mkPDF(self.pdf_name, 0)
        pdf_output = [pdfx.xfxQ2(self.flavors, x, self.Q_value) for x in self.x_pdf]
        return pdf_output

    def build_pdf(self):

        # Take n samples from the whole members
        if self.nb_replicas == 1:
            # Take the central replicas as the default
            pdf_central = [lhapdf.mkPDF(self.pdf_name, 0)]
        else:
            pdf_init = lhapdf.mkPDFs(self.pdf_name)
            pdf_central = sample(pdf_init, self.nb_replicas)

        # Generate the array
        data  = []
        for pdf in pdf_central:
            row = []
            for x in self.x_pdf:
                if self.flavors == 3 or self.flavors == 21:
                    row.append(pdf.xfxQ2(self.flavors,x,self.Q_value))
                else:
                    row.append(pdf.xfxQ2(self.flavors,x,self.Q_value))
                    # row.append(pdf.xfxQ2(self.flavors,x,self.Q_value)-pdf.xfxQ2(-self.flavors,x,self.Q_value))
            data.append(row)
        return np.array(data)
