import json
import pathlib
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from tqdm import trange
from ganpdfs.utils import smm
from ganpdfs.utils import axes_width
from ganpdfs.writer import WriterWrapper
from ganpdfs.utils import save_checkpoint
from ganpdfs.utils import interpolate_grid
from ganpdfs.model import WassersteinGanModel
from ganpdfs.model import DCNNWassersteinGanModel

logger = logging.getLogger(__name__)


class GanTrain:
    """Training class that controls the training of the GANs. It sets the interplay
    between the two different networks, namely the critic/discriminator and the
    generator.

    Parameters
    ----------
    xgrid : np.array(float)
        array of x-grid
    pdf : np.array(float)
        grid of PDF replicas of shape (nb_rep, nb_flv, xgrid_size)
    noise_size : int
        dimension of the noise input
    params : dict
        dictionary that contains all the specific parameters
    optmz : list
        list of optimizers
    nb_replicas: int, optional
        number of replicas
    """

    def __init__(self, xgrid, pdf, noise_size, params, activ, optmz, nb_replicas=10):
        self.xgrid = xgrid
        self.params = params
        self.noise_size = noise_size
        self.nb_replicas = nb_replicas
        self.hyperopt = params.get("scan")
        self.folder = params.get("save_output")

        # Choose architecture
        if params.get("architecture") == "dcnn":
            self.pdf = pdf.reshape((pdf.shape[0], pdf.shape[1], pdf.shape[2], 1))
            self.gan_model = DCNNWassersteinGanModel(self.pdf, params, noise_size, activ, optmz)
        else:
            self.pdf = pdf
            self.gan_model = WassersteinGanModel(self.pdf, params, noise_size, activ, optmz)

        # Initialize Models
        self.critic = self.gan_model.critic_model()
        self.generator = self.gan_model.generator_model()
        self.adversarial = self.gan_model.adversarial_model(self.generator, self.critic)

        # Print Summary when not in Hyperopt
        if not self.hyperopt:
            self.generator.summary()
            self.critic.summary()
            self.adversarial.summary()

        # Init. Checkpoint
        self.checkpoint = save_checkpoint(self.generator, self.critic, self.adversarial)

    def generate_real_samples(self, half_batch_size):
        """Prepare the real samples. This samples a half-batch of real dataset and
        assign to them target labels (-1) indicating that the samples are reals.

        Parameters
        ----------
        half_batch_size : int
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
        pdf_index = np.random.randint(0, self.pdf.shape[0], half_batch_size)
        pdf_batch = self.pdf[pdf_index]
        # Use (-1) as label for true pdfs
        y_disc = -np.ones((half_batch_size, 1))
        return pdf_batch, y_disc

    def sample_latent_space(self, half_batch_size):
        """Construct the random input noise that gets fed into the generator.

        Parameters
        ----------
        half_batch_size : int
            dimension of the half batch

        Returns
        -------
        np.array(float) 
            array of random numbers
        """
        # Generate points in the latent space
        latent = np.random.randn(self.noise_size * half_batch_size)
        return latent.reshape(half_batch_size, self.noise_size)

    def generate_fake_samples(self, generator, half_batch_size):
        """Generate fake samples from the Generator. `generate_fake_samples`
        takes input from the latent space and generate synthetic dataset.
        The synthetic dataset are assigned with target labels (1). The
        output of this is then gets fed to the Critic/Discriminator and used 
        to train the later.

        Parameters
        ----------
        generator : ganpdfs.model.WassersteinGanModel.generator
            generator neural networks
        half_batch_size : int
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
        noise = self.sample_latent_space(half_batch_size)
        pdf_fake = generator.predict(noise)
        # Use (1) as label for fake pdfs
        y_disc = np.ones((half_batch_size, 1))
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
        """Generate grid-plots at a given iteration throughout the training. In each
        grid is compared the generated dataset against the real one for a specific
        flavor. The plots are then saved into a folder.

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
        generated_pdf, _ = self.generate_fake_samples(generator, nb_output)

        # Interpolates
        xgrid = self.params.get("pdfgrid")
        interpol_fake = interpolate_grid(
                generated_pdf,
                self.xgrid,
                xgrid,
                interpol_type="UnivariateSpline"
        )

        # Initialize Grid Plots
        fig, axes = plt.subplots(ncols=2, nrows=3, figsize=[20, 26])

        # Define Evolution Basis Labels
        # TODO: Check if this is correct
        evolbasis = [r"\bar{u}", r"\bar{d}", "g", "d", "u", "s"]

        for i, axis in enumerate(axes.reshape(-1)):
            # Plot true replicas
            for true_rep in self.pdf:
                axis.plot(
                        self.xgrid[26:],
                        true_rep[i + 2][26:],
                        color="r",
                        label="true",
                        alpha=0.25
                )
            # Plot fake replicas
            for fake_rep in interpol_fake:
                axis.plot(
                        xgrid[36:],
                        fake_rep[i + 2][36:],
                        color="b",
                        label="fake",
                        alpha=0.35
                )
            axes_width(axis, lw=1.5)
            axis.grid(alpha=0.1, linewidth=1.5)
            axis.tick_params(length=7, width=1.5)
            axis.tick_params(which="minor", length=4, width=1)
            axis.set_xlim(self.xgrid[26], self.xgrid[-1])
            axis.set_xscale('log')
            axis.set_title(evolbasis[i], fontsize=16)

        fig.suptitle("Samples at Iteration %d" % niter)
        fig.savefig(
            f"{folder}/iterations/pdf_generated_at_{niter}.png",
            dpi=100
        )

    def train(self, nb_epochs=5000, batch_size=64):
        """Train the GANs networks for a given batch size. The training is done in
        such a way that the training of the generator and the critic/discriminator
        is well balanced (more details to be added).

        In order to to be able to evolve the generated grids, the format of the x-grid
        has to be the same as the default x-grid in the central replicas file. If this
        is no the case, then this function also performs the interpolation.

        The grids are then written into a file using the `WriterWrapper` module.

        In case `hyperopt` is on, the similarity metric--that is used as a sort of 
        figrue of merit to assess the performance of the model--is computed. This
        is afterwards used by the 'hyperscan' module.

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
        nb_steps = batch_per_epoch * nb_epochs
        if batch_size < 2:
            half_batch_size = 1
        else:
            half_batch_size = int(batch_size / 2)

        # Initialize Losses storing and checkpoint
        # folder path.
        rdloss, fdloss, advloss = [], [], []
        check_dir = f"./{self.folder}/checkpoint/ckpt"

        logger.info("Training:")
        with trange(nb_steps, disable=self.hyperopt) as iter_range:
            for k in iter_range:
                iter_range.set_description("Training")
                # Train the Critic
                # Make sure the Critic is trainable
                self.critic.trainable = True
                for _ in range(self.params.get("nd_steps", 4)):
                    # Train with the real samples
                    r_xinput, r_ydisc = self.generate_real_samples(half_batch_size)
                    r_dloss = self.critic.train_on_batch(r_xinput, r_ydisc)
                    # Train with the fake samples
                    f_xinput, f_ydisc = self.generate_fake_samples(self.generator, half_batch_size)
                    f_dloss = self.critic.train_on_batch(f_xinput, f_ydisc)

                # Train the Generator
                # Make sure that the Critic is not trainable
                self.critic.trainable = False
                for _ in range(self.params.get("ng_steps", 3)):
                    noise, y_gen = self.sample_output_noise(batch_size)
                    gloss = self.adversarial.train_on_batch(noise, y_gen)

                iter_range.set_postfix(Disc=r_dloss+f_dloss, Adv=gloss)

                if not self.hyperopt:
                    # Print log output
                    if k % 100 == 0:
                        advloss.append(gloss)
                        rdloss.append(r_dloss)
                        fdloss.append(f_dloss)
                        self.checkpoint.save(file_prefix=check_dir)

                    if k % 1000 == 0:
                        # TODO: Fix arguments plot
                        self.plot_generated_pdf(
                            self.generator,
                            self.params.get("out_replicas"),
                            self.folder,
                            k
                        )

        # Save Losses
        if not self.hyperopt:
            loss_info = [{"rdloss": rdloss, "fdloss": fdloss, "advloss": advloss}]
            output_losses = self.params.get("save_output")
            with open(f"{output_losses}/losses_info.json", "w") as outfile:
                json.dump(loss_info, outfile, indent=2)

        # Generate fake replicas with the trained model
        logger.info("Generating fake replicas with the trained model.")
        fake_pdf, _ = self.generate_fake_samples(self.generator, self.params.get("out_replicas"))

        if  not self.hyperopt:

            xgrid = self.xgrid
            comb_pdf = np.concatenate([self.pdf, fake_pdf])
            #############################################################
            # Interpolate the grids if the GANS-grids is not the same   #
            # as the input PDF.                                         #
            #############################################################
            if self.params.get("architecture") == "dcnn":
                comb_pdf = fake_pdf.reshape((comb_pdf.shape[0], comb_pdf.shape[1], comb_pdf.shape[2]))
            if self.xgrid.shape != self.params.get("pdfgrid").shape:
                xgrid = self.params.get("pdfgrid")
                logger.info("Interpolate and/or Extrapolate GANs grid to PDF grid.")
                comb_pdf = interpolate_grid(comb_pdf, self.xgrid, xgrid)

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
