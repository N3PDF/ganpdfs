import os
import sys
import json
import tempfile
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from ganpdfs.model import WassersteinGanModel
from ganpdfs.model import DCNNWassersteinGanModel


class GanTrain:
    """GanTrain.
    """

    def __init__(self, x_pdf, pdf, noise_size, params, activ, optmz, nb_replicas=1):
        self.x_pdf = x_pdf
        self.params = params
        self.sampled_pdf = pdf
        self.noise_size = noise_size
        self.nb_replicas = nb_replicas
        self.output_size = (pdf.shape[1], pdf.shape[2])
        self.xgan_model = DCNNWassersteinGanModel(
            noise_size, self.output_size, x_pdf, params, activ, optmz
        )

    def generate_real_samples(self, half_batch_size):
        """generate_real_samples.

            Parameters
            ----------
            half_batch_size :
                half_batch_size
        """
        # This meta-model is used to train the Critic from the real
        # samples. The output is then compared to the label (-1).

        # true_pdfs|==> Critic_Model ==>predicted_labels|-1|==>LOSS
        #                     ^_________________________________|
        #                          TUNING / BACKPROPAGATION
        # pdf_index = np.random.choice(
        #           self.sampled_pdf.shape[0], half_batch_size, replace=False
        # )
        pdf_index = np.random.randint(0, self.sampled_pdf.shape[0], half_batch_size)
        pdf_batch = self.sampled_pdf[pdf_index]
        # Use (-1) as label for true pdfs
        y_disc = -tf.ones([1, half_batch_size])
        return pdf_batch, y_disc

    def generate_fake_samples(self, half_batch_size):
        """generate_fake_samples.

            Parameters
            ----------
            half_batch_size :
                half_batch_size
        """
        # This meta-model is used to train the Critic from the fake samples.
        # The output is compared to the label (1).

        # fake_pdfs|==> Critic_Model ==>predicted_labels|1|==>LOSS
        #                     ^_________________________________|
        #                          TUNING / BACKPROPAGATION
        noise = self.sample_latent_space(half_batch_size, self.noise_size)
        pdf_fake = self.xgan_model.generator.predict(noise)
        # Use (1) as label for fake pdfs
        y_disc = tf.ones([1, half_batch_size])
        return pdf_fake, y_disc

    # def sample_input_and_gen(self, batch_size):
    #     """
    #     This meta-model is used to train the Critic. The prior and the 
    #     generated pdfs get both fed into the Critic Model:
    #
    #     true_pdfs|                                    |-1|
    #              |==> Critic_Model ==>predicted_labels|  |==>LOSS
    #     fake_pdfs|          ^                         |+1|    |
    #                         |_________________________________|
    #                              TUNING / BACKPROPAGATION
    #     """
    #     pdf_index = np.random.choice(
    #               self.sampled_pdf.shape[0], batch_size, replace=False
    #     )
    #     pdf_batch = self.sampled_pdf[pdf_index]
    #     noise = np.random.normal(0,1,size=[batch_size, self.noise_size])
    #     pdf_fake = self.xgan_model.generator.predict(noise)
    #     xinput = np.concatenate([pdf_batch, pdf_fake])
    #     y_disc = np.ones(2*batch_size)
    #     # Use (-1) as label for true pdfs and keep the
    #     # the label of the fake pdfs to be (1)
    #     y_disc[:batch_size] = -1.0
    #     return xinput, y_disc

    def sample_latent_space(self, batch_size, noise_size):
        """sample_latent_space.

            Parameters
            ----------
            batch_size :
                batch_size
            noise_size :
                noise_size
        """
        # This method construct the latent space (that gets fed into the
        # Generator) which is a random vector of noises. For instance,
        # it is a Guassian random generated variables
        return tf.random.normal([batch_size, noise_size])

    def latent_prior_noised(self, batch_size, noise_size):
        """latent_prior_noised.

            Parameters
            ----------
            batch_size :
                batch_size
            noise_size :
                noise_size
        """
        # This method construct the latent space (that gets fed into the
        # Generator) by adding noise to the true prior.
        xlatent = []
        for px in self.sampled_pdf[:batch_size]:
            xlatent.append(px + np.random.random((7, 70, 1)) * 0.02)
        return np.array(xlatent)

    def sample_output_noise(self, batch_size):
        """sample_output_noise.

            Parameters
            ----------
            batch_size :
                batch_size
        """
        # This meta-model is used to train the Generator. Only the Random
        # Vector gets fed into the Generator and the generated pdfs will
        # enter in the Critic which in turn produces predicted labels that
        # is used to tune and update the Generator Model during the training.

        # noise|==> Generator ==> generated_pdfs|==>Critic==>Label==>LOSS
        #              ^______________________________________________|
        #                        TUNING / BACKPROPAGATION
        noise = self.sample_latent_space(batch_size, self.noise_size)
        y_gen = -tf.ones([1, batch_size])
        return noise, y_gen

    def plot_generated_pdf(self, nth_epochs, nrep, folder):
        """This method plots the comparison of the true
        and generated PDF replicas at each iterations for
        a given flavour.
        """
        # Check the targed folder
        if not os.path.exists("%s/iterations" % folder):
            os.mkdir("%s/iterations" % folder)
        else:
            pass
        # Generate random vector and use it to make a prediction
        # for the Generator after a given training
        noise = self.sample_latent_space(nrep, self.noise_size)
        generated_pdf = self.xgan_model.generator.predict(noise)

        # Initialize the figure as a 4x4 grid
        fig = plt.figure(constrained_layout=True)
        spec = gridspec.GridSpec(ncols=2, nrows=2, figure=fig)
        pl1 = fig.add_subplot(spec[0, 0])
        pl2 = fig.add_subplot(spec[0, 1])
        pl3 = fig.add_subplot(spec[1, 0])
        pl4 = fig.add_subplot(spec[1, 1])

        # Define list of flavours and grids
        fl = [0, 1, 3, 6]
        ls_pl = [pl1, pl2, pl3, pl4]

        for fl, pos in zip(fl, ls_pl):
            # Plot true replicas
            for true_rep in self.sampled_pdf:
                pos.plot(
                        self.x_pdf,
                        true_rep[fl],
                        color="r",
                        label="true",
                        alpha=0.35
                )
            # Plot fake replicas
            for fake_rep in generated_pdf:
                pos.plot(
                        self.x_pdf,
                        fake_rep[fl],
                        color="b",
                        label="fake",
                        alpha=0.35
                )
            # Plot in log scale
            pos.set_xscale("log")

        fig.suptitle("Samples at Iteration %d" % nth_epochs)
        fig.savefig(
            "{}/iterations/pdf_generated_at_{}.png".format(
                folder, nth_epochs
            ),
            dpi=100,
            bbox_inches="tight",
            pad_inches=0.2,
        )

    def pretrain_disc(self, batch_size, epochs=4):
        """pretrain_disc.

            Parameters
            ----------
            batch_size :
                batch_size
            epochs :
                epochs
        """
        xinput, y_disc = self.sample_input_and_gen(batch_size)
        self.xgan_model.critic.trainable = True
        self.xgan_model.critic.fit(xinput, y_disc, epochs=epochs, batch_size=batch_size)

    def train(self, nb_epochs=1000, batch_size=1, verbose=False):
        """train.

            Parameters
            ----------
            nb_epochs :
                nb_epochs
            batch_size :
                batch_size
            verbose :
                verbose
        """
        # Initialize the value of metric
        metric = 0
        # Calculate the number of bacthes per training epochs
        batch_per_epoch = int(self.sampled_pdf.shape[0] / batch_size)
        # Calculate the number of training iterations
        nb_steps = batch_per_epoch * nb_epochs
        # Calculate the size of a half batch sample
        if batch_size < 2:
            half_batch_size = 1
        else:
            half_batch_size = int(batch_size / 2)

        # Initialize the file for the losses
        if not self.params["scan"]:
            f = open("%s/losses_info.csv" % self.params["save_output"], "w")
            f.write("Iter., Disc_Loss_Real, Disc_Loss_Fake, Adv_loss\n")
        for k in range(1, nb_steps + 1):

            # # Train the Critic
            # # Make sure to train the Critic
            # self.xgan_model.critic.trainable = True
            # for _ in range(self.params['nd_steps']):
            #     xinput, y_disc = self.sample_input_and_gen(batch_size)
            #     dloss = self.xgan_model.critic.train_on_batch(xinput, y_disc)

            # Train the Critic
            # Make sure the Critic is trainable
            self.xgan_model.critic.trainable = True
            for _ in range(self.params["nd_steps"]):
                # Train with the real samples
                r_xinput, r_ydisc = self.generate_real_samples(half_batch_size)
                r_dloss = self.xgan_model.critic.train_on_batch(r_xinput, r_ydisc)
                # Train with the real samples
                f_xinput, f_ydisc = self.generate_fake_samples(half_batch_size)
                f_dloss = self.xgan_model.critic.train_on_batch(f_xinput, f_ydisc)

            # Train the GAN
            # Make sure that the Critic is not trainable
            self.xgan_model.critic.trainable = False
            for _ in range(self.params["ng_steps"]):
                noise, y_gen = self.sample_output_noise(batch_size)
                gloss = self.xgan_model.adversarial.train_on_batch(noise, y_gen)

            # Print log output
            if verbose and not self.params["scan"]:
                if k % 100 == 0:
                    print(
                        "Iter:{} out of {}. dloss real: {:.3e}. dloss fake: {:.3e}. Adv loss: {:.3e}".format(
                            k, nb_steps, r_dloss, f_dloss, gloss
                        )
                    )
                    f.write(
                        "{0}, \t{1}, \t{2}, \t{3} \n".format(k, r_dloss, f_dloss, gloss)
                    )
                if k % 100 == 0:
                    self.plot_generated_pdf(
                        k, self.params["out_replicas"], self.params["save_output"]
                    )

        # Close the loss file
        if not self.params["scan"]:
            f.close()

        if self.params["scan"]:
            # Generate fake replicas with the trained model
            rnd_noise = self.sample_latent_space(self.nb_replicas, self.noise_size)
            fake_pdf = self.xgan_model.generator.predict(rnd_noise)

            # Compute SMM for hyperoptimization
            # TODO use a function call here instead of creating class + calling method and then discarding
            # metric = smm(self.sampled_pdf, fake_pdf, self.params).ERF()
            metric = 0

        return metric
