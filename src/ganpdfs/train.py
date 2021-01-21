import json
import pathlib
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from tqdm import trange
from ganpdfs.utils import smm
from numpy.random import PCG64
from numpy.random import Generator

from ganpdfs.model import WGanModel
from ganpdfs.model import DWGanModel
from ganpdfs.utils import axes_width
from ganpdfs.utils import interpolate
from ganpdfs.utils import gan_summary
from ganpdfs.writer import WriterWrapper
from ganpdfs.utils import latent_sampling

logger = logging.getLogger(__name__)


class GanTrain:
    """Training class that controls the training of the GANs. It sets
    the interplay between the two different networks (Discriminator &
    Generator) and the generator.

    Parameters
    ----------
    xgrid : np.array(float)
        array of x-grid
    pdf : np.array(float)
        grid of PDF replicas of shape (nb_rep, nb_flv, xgrid_size)
    optmz : list
        list of optimizers
    """

    def __init__(self, xgrid, pdfs, params):
        pdf, self.lhaPDFs = pdfs
        self.xgrid, self.params = xgrid, params
        self.hyperopt = params.get("scan")
        self.rndgen = Generator(PCG64(seed=0))
        self.folder = params.get("save_output")

        # Choose architecture
        if params.get("architecture") == "cnn":
            self.pdf = pdf
            self.gan = WGanModel(self.pdf, params)
        else:
            raise NotImplementedError("Not implemented yet!")
            # self.pdf = pdf.reshape(pdf.shape + (1,)))
            # self.gan = DWGanModel(self.pdf, params, activ)

        # Initialize Models
        self.critic = self.gan.critic_model()
        self.generator = self.gan.generator_model()
        self.adversarial = self.gan.adversarial_model(
            self.generator, self.critic
        )
        # Prepare latent space
        self.latent_pdf = latent_sampling(
                pdf,
                params.get("tot_replicas"),
                self.rndgen
        )

        # Print Summary when not in Hyperopt
        if not self.hyperopt:
            gan_summary(self.critic, self.generator, self.adversarial)

    def real_samples(self, half_batch):
        """Prepare the real samples. This samples a half-batch of real
        dataset and assign to them target labels (-1) indicating that
        the samples are reals.

        Parameters
        ----------
        half_batch : int
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
        # true_pdfs|==> Critic_Model ==>predicted_labels|-1|==>LOSS #
        #                     ^_________________________________|   #
        #                          TUNING / BACKPROPAGATION         #
        #############################################################
        pdf_index = self.rndgen.integers(0, self.pdf.shape[0], half_batch)
        pdf_batch = self.pdf[pdf_index]
        # Use (-1) as label for true pdfs
        y_disc = -np.ones((half_batch, 1))
        return pdf_batch, y_disc

    def sample_latent_space(self, half_batch):
        """Construct the random input noise that gets fed into the generator.

        Parameters
        ----------
        half_batch : int
            dimension of the half batch

        Returns
        -------
        np.array(float)
            array of random numbers
        """
        # Generate points in the latent space
        pdf_index = self.rndgen.integers(0, self.pdf.shape[0], half_batch)
        pdf_as_latent = self.latent_pdf[pdf_index]
        assert pdf_as_latent.shape == (half_batch, self.pdf.shape[1], self.pdf.shape[2])
        return pdf_as_latent

    def fake_samples(self, generator, half_batch):
        """Generate fake samples from the Generator. `fake_samples`
        takes input from the latent space and generate synthetic dataset.
        The synthetic dataset are assigned with target labels (1). The
        output of this is then gets fed to the Critic/Discriminator and
        used to train the later.

        Parameters
        ----------
        generator : ganpdfs.model.WGanModel.generator
            generator neural networks
        half_batch : int
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
        noise = self.sample_latent_space(half_batch)
        pdf_fake = generator.predict(noise)
        # Use (1) as label for fake pdfs
        y_disc = np.ones((half_batch, 1))
        return pdf_fake, y_disc

    def sample_output_noise(self, batch_size):
        """Samples output noises.

        Parameters
        ----------
        batch_size : int
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
        noise = self.sample_latent_space(batch_size)
        y_gen = -np.ones((batch_size, 1))
        return noise, y_gen

    def plot_generated_pdf(self, generator, nb_output, folder, niter):
        """Generate grid-plots at a given iteration throughout the training.
        In each grid is compared the generated dataset against the real one
        for a specific flavor. The plots are then saved into a folder.

        Parameters
        ----------
        generator : ganpdfs.model.WassersteinGanModel.generator
            generator neural networks
        nb_output : int
            number of fake PDFs to be generated in the plot
        folder : str
            name of the folder in which the plots will be saved
        niter : int
            iteration at which the plots should be generated
        """

        # Check the targed folder
        output_path = f"{folder}/iterations"
        output_folder = pathlib.Path().absolute() / output_path
        output_folder.mkdir(exist_ok=True)

        # Generate Fake Samples
        generated_pdf, _ = self.fake_samples(generator, nb_output)

        # Initialize Grid Plots
        fig, axes = plt.subplots(ncols=2, nrows=3, figsize=[20, 26])

        for i, axis in enumerate(axes.reshape(-1)):
            # Plot true replicas
            for true_rep in self.pdf:
                axis.plot(
                        self.xgrid[650:],
                        true_rep[i + 2][650:],
                        color="r",
                        label="true",
                        alpha=0.25
                )
            # Plot fake replicas
            for fake_rep in generated_pdf:
                axis.plot(
                        self.xgrid[650:],
                        fake_rep[i + 2][650:],
                        color="b",
                        label="fake",
                        alpha=0.35
                )
            axes_width(axis, lw=1.5)
            axis.grid(alpha=0.1, linewidth=1.5)
            axis.tick_params(length=7, width=1.5)
            axis.tick_params(which="minor", length=4, width=1)
            axis.set_xlim(1e-1, self.xgrid[-1])
            # axis.set_xlim(self.xgrid[50], self.xgrid[-1])
            #axis.set_xscale('log')

        fig.suptitle("Samples at Iteration %d" % niter)
        fig.savefig(
            f"{folder}/iterations/pdf_generated_at_{niter}.png",
            dpi=100
        )
        plt.close("all")

    def train(self, nb_epochs=5000, batch_size=64):
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
        batch_size : int
            dimension of the batch size

        Returns
        -------
        float:
            similarity metric value
        """
        # Initialize the value of metric
        metric = 0
        batch_per_epoch = int(self.pdf.shape[0] / batch_size)
        nb_steps = nb_epochs
        #nb_steps = batch_per_epoch * nb_epochs
        if batch_size < 2:
            half_batch = 1
        else:
            half_batch = int(batch_size / 2)

        # Initialize Losses storing and checkpoint folder path.
        logger.info("Training:")
        rdloss, fdloss, advloss = [], [], []
        with trange(nb_steps, disable=self.hyperopt) as iter_range:
            for k in iter_range:
                iter_range.set_description("Training")
                # Train the Critic
                # Make sure the Critic is trainable
                self.critic.trainable = True
                for _ in range(self.params.get("nd_steps", 4)):
                    # Train with the real samples
                    r_xinput, r_ydisc = self.real_samples(half_batch)
                    r_dloss = self.critic.train_on_batch(r_xinput, r_ydisc)
                    # Train with the fake samples
                    f_xinput, f_ydisc = self.fake_samples(self.generator, half_batch)
                    f_dloss = self.critic.train_on_batch(f_xinput, f_ydisc)

                # Train the Generator
                # Make sure that the Critic is not trainable
                self.critic.trainable = False
                for _ in range(self.params.get("ng_steps", 3)):
                    noise, y_gen = self.sample_output_noise(batch_size)
                    gloss = self.adversarial.train_on_batch(noise, y_gen)

                # Update progression bar
                iter_range.set_postfix(Disc=r_dloss+f_dloss, Adv=gloss)

                if not self.hyperopt:
                    # Print log output
                    if k % 100 == 0:
                        advloss.append(gloss)
                        rdloss.append(r_dloss)
                        fdloss.append(f_dloss)

                    if k % 100 == 0:
                        self.plot_generated_pdf(
                            self.generator,
                            self.params.get("out_replicas"),
                            self.folder,
                            k
                        )

        if not self.hyperopt:
            loss_info = [{"rdloss": rdloss, "fdloss": fdloss, "advloss": advloss}]
            output_losses = self.params.get("save_output")
            with open(f"{output_losses}/losses_info.json", "w") as outfile:
                json.dump(loss_info, outfile, indent=2)

        # Generate fake replicas with the trained model
        logger.info("Generating fake replicas with the trained model.")
        fake_pdf, _ = self.fake_samples(self.generator, self.params.get("out_replicas"))

        if not self.hyperopt:

            xgrid = self.xgrid
            #############################################################
            # Interpolate the grids if the GANS-grids is not the same   #
            # as the input PDF.                                         #
            #############################################################
            if self.params.get("architecture") == "dcnn":
                fake_pdf = np.squeeze(fake_pdf, axis=3)
            if self.xgrid.shape != self.params.get("pdfgrid").shape:
                xgrid = self.params.get("pdfgrid")
                logger.info("Interpolate and/or Extrapolate GANs grid to PDF grid.")
                fake_pdf = interpolate(fake_pdf, self.xgrid, xgrid, mthd="Intperp1D")
            # Combine the PDFs
            comb_pdf = np.concatenate([self.lhaPDFs, fake_pdf])

            #############################################################
            # Construct the output grids in the same structure as the   #
            # N3FIT outputs. This allows for easy evolution.            #
            #############################################################
            logger.info("Write grids to file.")
            for rpid, replica  in enumerate(comb_pdf, start=1):
                grid_path = f"{self.folder}/nnfit/replica_{rpid}"
                write_grid = WriterWrapper(
                        self.folder,
                        replica,
                        xgrid,
                        rpid,
                        self.params.get("q")
                    )
                write_grid.write_data(grid_path)
        else:
            # Compute FID inception score
            metric, _ = smm(self.pdf, fake_pdf)

        return metric
