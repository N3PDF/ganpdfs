# Some useful Functions

import logging
import numpy as np
from math import gcd
from tqdm import trange
from scipy import interpolate
from scipy.linalg import sqrtm
from rich.console import Console

console = Console()
logger = logging.getLogger(__name__)


def do_nothing(tensor):
    """Function that does nothing.

    Parameters
    ----------
    tensor: tf.tensor
        Tensor
    """
    return tensor


def gan_summary(critic, generator, adversarial):
    console.print("\n• GanPDFs Architecture:", style="bold blue")
    generator.summary()
    critic.summary()
    adversarial.summary()


def axes_width(ax, lw=1):
    """Change width of axes.
    Parameters
    ----------
    ax : matplotlib.axes.Axes
    Figure's axes
    lw : float
    Define width of axes
    """

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(lw)


def latent_noise(pdf, rndgen, s=0.4):
    """Apply gaussian noise to the latent input.

    Parameters
    ----------
    pdf: np.array
        Input PDF to be noised
    rndgen:
        random number generator
    s: float
        Intensity of the noise
    """
    arrmin = np.min(pdf)
    arrmax = np.max(pdf)
    gaussian = rndgen.normal(loc=0, scale=1, size=pdf.shape)
    reslt = np.clip(pdf * (1 + s * gaussian), arrmin, arrmax)
    return reslt


def latent_sampling(pdf, nb_output, rndgen):
    """latent_sampling.

    Parameters
    ----------
    pdf: np.array
        Array/Grid of input PDFs
    nb_output: int
       Total number of replica
    rndgen:
        Random replica generator
    """

    # assert pdf.shape[0] <= nb_output
    if nb_output is None: nbgen = pdf.shape[0]
    else: nbgen = nb_output
    extra_latent = []
    for n in range(nbgen):
        selected = []
        for _ in range(rndgen.integers(1, 4)):
            select = pdf[rndgen.integers(pdf.shape[0])]
            selected.append(select)
        rslt = np.array(selected)
        rslt = np.sum(rslt, axis=0) / rslt.shape[0]
        extra_latent.append(rslt)
    freslt = np.array(extra_latent)
    return freslt


def factorize_number(number):
    """Factorize_number using Pollard's rho algorithm. This takes
    a number a return a list of the factors.

    Example:
        Given an integer 70, this can be factorized a produc of the
        following integers [7,5,2].

    Notice that by definition, 1 is not included.

    Parameters
    ----------
    number : int
        number to be factorized

    Returns
    -------
    list(int)
        list of the factors
    """

    factors = []

    def get_factor(number):
        x = 2
        factor = 1
        x_fixed = 2
        cycle_size = 2

        while factor == 1:
            for count in range(cycle_size):
                if factor > 1:
                    break
                x = (x * x + 1) % number
                factor = gcd(x - x_fixed, number)
            cycle_size *= 2
            x_fixed = x
        return factor

    while number > 1:
        new_number = get_factor(number)
        factors.append(new_number)
        number //= new_number

    return factors


def construct_cnn(number, nb_layer):
    """Factorize_number using Pollard's rho algorithm that is defined
    by the `factorize_number` method. This is used in order to define
    the dimension of the `strides` for performing the convolution in
    the model class `DCNNWassersteinGanModel`.

    The issue is the following: given a pair of two integers (m, n)
    such that n < m, how can we decompose m into n factors.

    Example:
        Given a pair (70, 3), we have [7,5,2]

    If m cannot be decomposed further, then the remaining cases are
    naturally set to 1.

    Example:
        Given a pair (58, 4) we have [29, 2, 1, 1]

    Noe that the final result is sorted.

    Parameters
    ----------
    number : int
        numbker to be factorized
    nb_layer : int
        dimension of the list that contains the factors


    Returns
    -------
    list(int)
        list of the factors with length `nb_layer`
    """

    factorized = factorize_number(number)
    size = len(factorized)
    cnn_dim = sorted(factorized)
    if size < nb_layer:
        for _ in range(1, nb_layer - size + 1):
            cnn_dim.append(1)
    elif size > nb_layer:
        cnn_dim = cnn_dim[:nb_layer - 1]
        new_elem = np.prod(np.array(cnn_dim))
        cnn_dim.append(new_elem)
    else:
        pass
    return cnn_dim


def interpol(fake_pdf, gan_grid, lhapdf_grid, mthd="Intperp1D"):
    """Interpolate the generated output according to the x-grid in
    order to match with the LHAPDF grid-format. It uses the `interpolate`
    module from `scipy`. For more details, refere to
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html

    Parameters
    ----------
    fake_pdf : np.array
        generated PDF replicas of shape (nb_repl, nb_flv, gan_grid)
    gan_grid : np.array
        custom x-grid of shape (gan_grid,)
    lhapdf_grid :
        lhapdf-like grid of shape (lhapdf_grid,)

    Returns
    -------
    np.array(float)
        fake PDF replica of shape (nb_repl, nb_flv, gan_grid)
    """
    final_grid = []
    with trange(fake_pdf.shape[0]) as iter_range:
        for k in iter_range:
            fl_space = []
            for fl in fake_pdf[k]:
                if mthd == "Intperp1D":
                    f_interpol = interpolate.interp1d(
                            gan_grid,
                            fl,
                            kind="cubic",
                            fill_value="extrapolate"
                    )
                    new_grid = f_interpol(lhapdf_grid)
                elif mthd == "SplineEv":
                    f_interpol = interpolate.splrep(gan_grid, fl, s=0)
                    new_grid = interpolate.splev(lhapdf_grid, f_interpol, der=0)
                elif mthd == "CubicSpline":
                    f_interpol = interpolate.CubicSpline(gan_grid, fl)
                    new_grid = f_interpol(lhapdf_grid)
                elif mthd == "InterpSpline":
                    f_interpol = interpolate.make_interp_spline(gan_grid, fl)
                    new_grid = f_interpol(lhapdf_grid)
                elif mthd == "UnivariateSpline":
                    weights = np.isnan(fl)
                    fl[weights] = 0.
                    f_interpol = interpolate.UnivariateSpline(
                            gan_grid,
                            fl,
                            w=~weights
                    )
                    f_interpol.set_smoothing_factor(0.01)
                    new_grid = f_interpol(lhapdf_grid)
                else:
                    raise ValueError(f"{mthd} is not an interpolation.")
                fl_space.append(new_grid)
                iter_range.set_description(
                        f"{k+1} synthetic replicas out of {fake_pdf.shape[0]}"
                )
            final_grid.append(fl_space)
    return np.array(final_grid)


def smm(prior, generated):
    """Similarity Metric Measure that measures the quality of the
    generated PDF replicas using the `Fréchet Inception Distance`
    (FID).

    TODO: Check how the total/final FIDs is computed.

    Parameters
    ----------
    prior : np.array(float)
        Prior MC PDF replicas of shape (N, Nf, X)
    generated : np.array(float)
        Generated MC PDF replicas of shape (\tilde{N}, Nf, X)

    Returns
    -------
    float:
        FID
    """

    dim = prior.shape
    # Reshape if the architecture is a DCNN
    if prior.ndim > 3:
        prior = prior.reshape((dim[0], dim[1], dim[2]))
        generated = generated.reshape((dim[0], dim[1], dim[2]))

    # Transpose results w.r.t. the flavors
    prior = np.transpose(prior, axes=[1, 0, 2])
    generated = np.transpose(generated, axes=[1, 0, 2])
    # Prepare FID array
    fid_arr = np.zeros(dim[1])

    def compute_fid(fl_prior, fl_generated):
        """Measure the quality of the generated PDF using the `Fréchet
        Inception Distance` (FID). The Frechet distance between two
        multivariate Gaussians X_1 ~ N(mu_1, C_1) and X_2 ~ N(mu_2, C_2)
        is:
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2 * sqrt(C_1 * C_2)).

        If the generated PDF replica is exactly the same as the prior,
        the value of the FID is zero; that means that the smaller the
        value of the FID is, the similar the generated replica is to
        the prior.

        For details about the FID's Inception Score measure, refer to
        the following: https://arxiv.org/abs/1706.08500

        Parameters
        ----------
        fl_prior : np.array(float)
            Array of prior PDF replica for a given flavor
        fl_generated : np.array(float)
            Array of generated PDF replica for a given flavor

        Returns
        -------
        float:
            FID value
        """

        # calculate mean and covariance statistics
        mu1, sigma1 = fl_prior.mean(axis=0), np.cov(fl_prior, rowvar=False)
        mu2, sigma2 = fl_generated.mean(axis=0), np.cov(fl_generated, rowvar=False)
        # Check if Infs or NaNs and return a big nnumber
        if (np.isnan(mu2).any() or np.isnan(sigma2).any()):
            return np.random.randint(400, 1000)

        # calculate sum squared difference between means
        ssdiff = np.sum((mu1 - mu2) ** 2.0)
        # calculate sqrt of product between cov
        covmean = sqrtm(sigma1.dot(sigma2))
        # check and correct imaginary numbers from sqrt
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        # calculate score
        fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
        return fid

    for fl in range(dim[1]):
        fid_arr[fl] = compute_fid(prior[fl], generated[fl])
    fids_stdv = np.std(fid_arr)
    fids_mean = np.mean(fid_arr)
    fids_resc = fids_stdv * fids_mean

    return fids_resc, fid_arr
