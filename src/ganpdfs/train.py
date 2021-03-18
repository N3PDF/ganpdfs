import logging
import json
import pathlib
import shutil
import numpy as np
import matplotlib.pyplot as plt
import NNPDF as nnpath

from tqdm import tqdm
from tqdm import trange
from ganpdfs.utils import smm
from numpy.random import PCG64
from numpy.random import Generator
from rich.console import Console

from tensorflow.keras import Model
from tensorflow.keras.layers import Input

from ganpdfs.model import WGanModel
from ganpdfs.model import DWGanModel
from ganpdfs.utils import interpol
from ganpdfs.utils import axes_width
from ganpdfs.utils import gan_summary
from ganpdfs.custom import get_optimizer
from ganpdfs.utils import latent_sampling
from ganpdfs.writer import WriterWrapper
from ganpdfs.custom import save_ckpt
from ganpdfs.custom import wasserstein_loss
from ganpdfs.custom import load_generator
from ganpdfs.custom import GradientPenalty
from ganpdfs.custom import turn_on_training


console = Console()
logger = logging.getLogger(__name__)
STYLE = "bold blue"


class GanTrain:
    """The `GanTrain` class controls the training of the Generative
    Adversarial Neural Networks. It sets the intrplay between the two
    neural networks: Generator and Discriminator.

    Parameters
    ----------
    xgrid : np.array(float)
        array of x-grid from which the input PDF grid for the GAN was
        computed. Notice that this does not have to be the same as the
        LHAPDF grid.
    pdfs : tuple(np.array(float), np.array(float))
        The tupple contains two differnent PDF grids: one generated with
        the GAN x-grid and one generated with the LHAPDF grid.
    params: dict
        Dictionary containing all the input parameters.
    """

    def __init__(self, xgrid, pdfs, params):
        self.noise = 100
        self.norm, pdf, self.lhaPDFs = pdfs
        self.xgrid = xgrid
        self.params = params
        dim, flsize, xgssize = pdf.shape
        self.batch = dim * params["batch_size"] // 100
        self.hyperopt = params.get("scan")
        self.rndgen = Generator(PCG64(seed=0))
        self.folder = params.get("save_output")
        discparams = params.get("disc_parameters")

        # Prepare latent space
        self.latent_pdf = latent_sampling(
                pdf,
                params.get("tot_replicas"),
                self.rndgen,
                nsx=params.get("gauss_noise", 1e-5)
        )
        params["latent_space"] = self.latent_pdf

        # Define Models architecture
        if params.get("architecture") == "dnn":
            real_shape = Input(shape=(flsize, xgssize))
            genparams = params.get("gen_parameters")
            if genparams.get("structure") == "standard":
                synt_shape = Input(shape=(self.noise,))
            elif genparams.get("structure") == "custom":
                synt_shape = Input(shape=(flsize, xgssize))
            self.pdf = pdf
            self.gan = WGanModel(self.pdf, params)
        elif params.get("architecture") == "dcnn":
            logger.warning("DCNN is not fully operational yet!")
            self.pdf = pdf.reshape(pdf.shape + (1,))
            real_shape = Input(shape=(flsize, xgssize, 1))
            synt_shape = Input(shape=(self.noise,))
            self.gan = DWGanModel(self.pdf, params, self.noise)
        else:
            raise ValueError("Invalid Achitecture!")

        # Initialize Models
        self.generator = self.gan.generator_model()
        self.critic = self.gan.critic_model()
        self.adversarial = self.gan.adversarial_model(
            self.generator, self.critic
        )

        # Make sure the Models are trainable
        turn_on_training(self.critic, self.generator)

        # Define Graph for the combined Discriminator. It takes both
        # real PDF samples and Latent space/noise. The Latent Space
        # (noise) is run through the Generator to generate the fake
        # PDF replicas. Both the real and fake PDFs run through the
        # Discriminator for evaluation.
        # The implementation below includes a "Gradient Penalty" for
        # the discriminator, to consider it during the training the
        # training, just modify the loss weight to 1.0. In this case
        # the weights clipping in the input runcard needs to be relaxed.
        synt_discr = self.generator(synt_shape)
        discr_real = self.critic(real_shape)
        discr_synt = self.critic(synt_discr)

        partial_gp = GradientPenalty(
                self.critic,
                self.batch,
                params.get("architecture"),
                discparams.get("gp_weight", 10)
        )([real_shape, synt_discr])

        self.discriminator = Model(
                inputs=[real_shape, synt_shape],
                outputs=[discr_real, discr_synt, partial_gp]
        )
        inputloss = discparams.get("loss", "wasserstein")
        if inputloss != "wasserstein":
            dloss = self.discparams.get("loss")
        else: dloss = wasserstein_loss
        opt_name = discparams.get("optimizer")
        ds_optmz = get_optimizer(opt_name)
        self.discriminator.compile(
                optimizer=ds_optmz,
                loss=[dloss, dloss, "mse"],
                loss_weights=[1.0, 1.0, discparams.get("gp_loss", 1.0)]
        )

        if not self.hyperopt and not params.get("use_saved_model"):
            gan_summary(self.critic, self.generator, self.adversarial)

        # Save Checkpoints
        self.ckpt = save_ckpt(self.generator, self.critic, self.adversarial)

    def real_samples(self, arrpdf, batch):
        """Prepare the real samples. This samples a half-batch of real
        dataset and assign to them target labels (+1) indicating that
        the samples are reals.

        Parameters
        ----------
        batch : int
            dimension of the half batch

        Returns
        -------
        tuple(np.array, np.array)
            containing the random real samples and the target
        labels
        """

        #############################################################
        # Training Description:                                     #
        # --------------------                                      #
        # true_pdfs|==> Critic_Model ==>predicted_labels|+1|==>LOSS #
        #                     ^_________________________________|   #
        #                          TUNING / BACKPROPAGATION         #
        #############################################################
        pdf_index = self.rndgen.integers(0, self.pdf.shape[0], batch)
        pdf_batch = self.pdf[pdf_index]
        y_disc = np.ones((batch, 1))
        return pdf_batch, y_disc

    def sample_latent_space(self, batch):
        """Latent vector that is given as input to the generator. (1) For the
        deep neural network, the latent space vector is constructed as linear
        combinations of the real input PDF with a guassian noise. (2) For the
        deep convolutional network case, the latent vector is just a vector
        of random noise as in the standard GAN/WGAN.

        Parameters
        ----------
        batch : int
            dimension of the half batch

        Returns
        -------
        np.array(float)
            array of random numbers
        """

        genparams = self.params.get("gen_parameters")
        noise_dim = genparams.get("noise_dim", 100)
        if self.params["architecture"]=="dcnn" or genparams["structure"]=="standard":
            latent = self.rndgen.random(noise_dim * batch)
            return latent.reshape(batch, noise_dim)
        elif self.params["architecture"]=="dnn" and genparams["structure"]=="custom":
            pdf_index = self.rndgen.integers(0, self.pdf.shape[0], batch)
            pdf_as_latent = self.latent_pdf[pdf_index]
            expected_shape = (batch, self.pdf.shape[1], self.pdf.shape[2])
            assert pdf_as_latent.shape == expected_shape
            return pdf_as_latent
        else:
            raise ValueError("There is an Error in the architecture!")

    def fake_samples(self, generator, batch):
        """Generate fake samples from the Generator. `fake_samples`
        takes input from the latent space and generate synthetic dataset.
        The synthetic dataset are assigned with target labels (1). The
        output of this is then gets fed to the Critic/Discriminator and
        used to train the later.

        Parameters
        ----------
        generator : ganpdfs.model.WGanModel.generator
            generator neural networks
        batch : int
            dimension of the half batch


        Reuturns
        --------
        tuple(np.array, np.array)
            containing samples from the generated data and the target
            labels
        """

        #############################################################
        # Training Description:                                     #
        # --------------------                                      #
        # fake_pdfs|==> Critic_Model ==>predicted_labels|1|==>LOSS  #
        #                     ^_________________________________|   #
        #                          TUNING / BACKPROPAGATION         #
        #############################################################
        noise = self.sample_latent_space(batch)
        pdf_fake = generator.predict(noise)
        y_disc = -np.ones((batch, 1))
        return pdf_fake, y_disc

    def sample_output_noise(self, batch):
        """Sample the latent space for a given batch and returns the
        corresponding predictions--which is a vector of (-1).

        Parameters
        ----------
        batch: int
            dimension of the batch

        Returns
        -------
        tuple(np.array, np.array)
            noises and the corresponding target labels
        """

        #####################################################################
        # Training Description:                                             #
        # --------------------                                              #
        # noise|==> Generator ==> generated_pdfs|==>Critic==>Label==>LOSS   #
        #              ^______________________________________________|     #
        #                        TUNING / BACKPROPAGATION                   #
        #####################################################################
        noise = self.sample_latent_space(batch)
        y_gen = -np.ones((batch, 1))
        return noise, y_gen

    def dummy_samples(self, batch):
        """Dummy samples that is passed to the gradient penalty. This
        is not used during the training.

        Parameters
        ----------
        batch: int
            dimension of the batch

        Returns
        -------
        np.array(int):
            Vector of zeros
        """

        return np.zeros((batch, 1))

    def train(self, nb_epochs=1000):
        """Train the GANs networks for a given batch size. The training
        is done in such a way that the training of the generator and the
        critic/discriminator is well balanced (more details to be added).

        In order to to be able to evolve the generated grids, the format
        of the x-grid has to be the same as the default x-grid in the
        central replicas file. If this is no the case, then this function
        also performs the interpolation.

        The grids are then written into a file using the `WriterWrapper`
        module.

        In case `hyperopt` is on, the similarity metric--that is used as
        a sort of figrue of merit to assess the performance of the model
        --is computed. This is afterwards used by the 'hyperscan' module.

        Parameters
        ----------
        nb_epochs : int
            total number of epochs

        Returns
        -------
        float:
            similarity metric value
        """

        metric = 0  # Initialize the value of metric
        if not self.params.get("use_saved_model"):
            rdloss, fdloss, advloss = [], [], []
            with trange(nb_epochs, disable=self.hyperopt) as iter_range:
                for k in iter_range:
                    iter_range.set_description("Training")
                    for _ in range(self.params.get("nd_steps", 4)):
                        r_input, r_ydisc = self.real_samples(
                                self.pdf, self.batch
                        )
                        f_input, f_ydisc = self.sample_output_noise(
                                self.batch
                        )
                        d_ydisc = self.dummy_samples(self.batch)
                        dloss = self.discriminator.train_on_batch(
                                [r_input, f_input],
                                [r_ydisc, f_ydisc, d_ydisc]
                        )
                    for _ in range(self.params.get("ng_steps", 3)):
                        noise, y_gen = self.sample_output_noise(self.batch)
                        gloss = self.adversarial.train_on_batch(noise, y_gen)
                    iter_range.set_postfix(rdisc=dloss[0], fdisc=dloss[1], adv=gloss)

                    if not self.hyperopt and (k + 1) % 100 == 0:
                        rdloss.append(dloss)
                        advloss.append(gloss)

            # Save generator model into a folder
            self.generator.save("pre-trained-model")

            if not self.hyperopt:
                loss_info = [{"rdloss": rdloss, "fdloss": fdloss, "advloss": advloss}]
                output_losses = self.params.get("save_output")
                with open(f"{output_losses}/losses_info.json", "w") as outfile:
                    json.dump(loss_info, outfile, indent=2)
        else:
            # Generate fake replicas with the trained model
            console.print(
                "\n• Making predictions using a pre-trained Generator model.",
                style="bold magenta"
            )
            self.generator = load_generator("pre-trained-model")

        fake_pdf, _ = self.fake_samples(self.generator, self.params.get("out_replicas"))

        if not self.hyperopt:

            xgrid = self.xgrid
            # Undo normalization
            fake_pdf = (fake_pdf * self.norm) + self.norm
            #############################################################
            # Interpolate the grids if the GANS-grids is not the same   #
            # as the input PDF.                                         #
            #############################################################
            if self.params.get("architecture") == "dcnn":
                fake_pdf = np.squeeze(fake_pdf, axis=3)
            if self.xgrid.shape != self.params.get("pdfgrid").shape:
                xgrid = self.params.get("pdfgrid")
                console.print("\n• Interpolate GANs grid to LHAPDF grid:", style=STYLE)
                fake_pdf = interpol(fake_pdf, self.xgrid, xgrid, mthd="Intperp1D")
            # Combine the PDFs
            comb_pdf = np.concatenate([self.lhaPDFs, fake_pdf])

            #############################################################
            # Construct the output grids in the same structure as the   #
            # N3FIT outputs. This allows for easy evolution.            #
            #############################################################
            console.print("\n• Write grids to file:", style=STYLE)
            with tqdm(total=comb_pdf.shape[0]) as evolbar:
                for rpid, replica in enumerate(comb_pdf, start=1):
                    grid_path = f"{self.folder}/nnfit/replica_{rpid}"
                    write_grid = WriterWrapper(
                            self.folder,
                            replica,
                            xgrid,
                            rpid,
                            self.params.get("q")
                        )
                    write_grid.write_data(grid_path)
                    evolbar.update(1)
                    evolbar.set_description("Progress")

            #############################################################
            # Copy fit runcard to the enhanced folder                   #
            #############################################################
            try:
                fitpath = nnpath.get_results_path()
                fitpath = fitpath + f"{self.params['pdf']}/filter.yml"
                shutil.copy(fitpath, f"{self.folder}/filter.yml")
            except IOError as excp:
                console.print(
                        f"[bold red]WARNING: Fit run card for {self.params['pdf']}"
                        f" not found. Put it manually into {self.folder} in order"
                        f" to run evolven3fit. {excp}"
                )
        else:
            # Compute FID inception score
            metric, _ = smm(self.pdf, fake_pdf)

        return metric
