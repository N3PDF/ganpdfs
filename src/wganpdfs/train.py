import os, sys
import numpy as np
import json, tempfile
from scipy import stats
import matplotlib.pyplot as plt
from keras.models import Sequential
from wganpdfs.custom import xmetrics
from wganpdfs.pdformat import input_pdfs
from wganpdfs.model import wasserstein_xgan_model
from wganpdfs.model import dcnn_wasserstein_xgan_model


class xgan_train(object):

    """
    Train the model and plot the result.
    """

    def __init__(self, x_pdf, pdf_name, noise_size, params, activ, optmz, nb_replicas=1,
                 Q_value=1.7874388, flavors=2):
        self.sampled_pdf = input_pdfs(pdf_name, x_pdf, nb_replicas,
                                      Q_value, flavors).build_pdf()
        self.x_pdf = x_pdf
        self.nb_replicas = nb_replicas
        self.output_size = len(x_pdf)
        self.noise_size = noise_size
        self.params = params
        self.xgan_model = wasserstein_xgan_model(noise_size, self.output_size,
                                                 x_pdf, params, activ, optmz)

    def plot_generated_pdf(self, nth_training, nrep, folder):
        """
        This method plots the comparison of the true
        and generated PDF replicas at each iterations.
        In addition, it shows the histogram comparisons
        of the two distributions for a given specific 
        value of x.
        """
        # Check the targed folder
        if not os.path.exists('%s/iterations' % folder):
            os.mkdir('%s/iterations' % folder)
        else:
            pass
        # Generate random vector and use it to make a prediction
        # for the Generator after a given training
        noise = self.sample_latent_space(nrep, self.noise_size)
        generated_pdf = self.xgan_model.generator.predict(noise)
        # # Save each generated replicas into a numpy file
        # np.save('%s/generated_at_%d_iteration.npy' % (folder, nth_training), generated_pdf)

        # Define the x values
        xv = [20, 21, 28, 29]

        # Initialize the figure as a 4x4 grid
        grid_size = (4,4)
        fig = plt.figure(figsize=(16,17))
        main  = plt.subplot2grid(grid_size, (0,0), colspan=2, rowspan=2)
        hist1 = plt.subplot2grid(grid_size, (0,2))
        hist2 = plt.subplot2grid(grid_size, (0,3))
        hist3 = plt.subplot2grid(grid_size, (1,2))
        hist4 = plt.subplot2grid(grid_size, (1,3))

        lst_hist = [hist1, hist2, hist3, hist4]

        for i in range(self.sampled_pdf.shape[0]):
            main.plot(self.x_pdf, self.sampled_pdf[i], color='deeppink', alpha=0.35)
        for j in range(generated_pdf.shape[0]):
            main.plot(self.x_pdf, generated_pdf[j], color='dodgerblue', alpha=0.45)

        for x, position in zip(xv, lst_hist):
            true_hist = np.array([repl[x] for repl in self.sampled_pdf])
            fake_hist = np.array([repl[x] for repl in generated_pdf])
            position.hist(true_hist, histtype='stepfilled', bins=20,
                            color="deeppink", alpha=0.65, label="true", density=True)
            position.hist(fake_hist, histtype='stepfilled', bins=20,
                            color="dodgerblue", alpha=0.65, label="fake", density=True)
        main.set_xscale('log')
        main.set_xlim([1e-4,1])

        fig.suptitle('Samples at Iteration %d'%nth_training, y=0.98)
        fig.tight_layout()
        fig.savefig('%s/iterations/pdf_generated_at_training_%d.png'
                    %(folder, nth_training), dpi=100,
                    bbox_inches='tight', pad_inches=0.2)
        fig.legend()

    # def sample_input_and_gen(self, batch_size):
    #     """
    #     This meta-model is used to train the Critic. The prior and the generated 
    #     pdfs get both fed into the Critic Model:
    #     
    #     true_pdfs|                                    |-1|
    #              |==> Critic_Model ==>predicted_labels|  |==>LOSS
    #     fake_pdfs|          ^                         |+1|    |
    #                         |_________________________________|
    #                              TUNING / BACKPROPAGATION
    #     """
    #     pdf_index = np.random.choice(self.sampled_pdf.shape[0], batch_size, replace=False)
    #     pdf_batch = self.sampled_pdf[pdf_index]
    #     noise = np.random.normal(0,1,size=[batch_size, self.noise_size])
    #     pdf_fake = self.xgan_model.generator.predict(noise)
    #     xinput = np.concatenate([pdf_batch, pdf_fake])
    #     y_disc = np.ones(2*batch_size)
    #     # Use (-1) as label for true pdfs and keep the
    #     # the label of the fake pdfs to be (1)
    #     y_disc[:batch_size] = -1.0
    #     return xinput, y_disc

    def generate_real_samples(self, half_batch_size):
        """
        This meta-model is used to train the Critic from the real
        samples. The output is then compared to the label (-1).
        
        true_pdfs|==> Critic_Model ==>predicted_labels|-1|==>LOSS
                            ^_________________________________|
                                 TUNING / BACKPROPAGATION
        """
        # pdf_index = np.random.choice(self.sampled_pdf.shape[0], half_batch_size, replace=False)
        pdf_index = np.random.randint(0, self.sampled_pdf.shape[0], half_batch_size)
        pdf_batch = self.sampled_pdf[pdf_index]
        # Use (-1) as label for true pdfs
        y_disc = -np.ones(half_batch_size)
        return pdf_batch, y_disc

    def generate_fake_samples(self, half_batch_size):
        """
        This meta-model is used to train the Critic from the fake samples.
        The output is compared to the label (1).  
        
        fake_pdfs|==> Critic_Model ==>predicted_labels|1|==>LOSS
                            ^_________________________________|
                                 TUNING / BACKPROPAGATION
        """
        noise    = self.sample_latent_space(half_batch_size, self.noise_size)
        pdf_fake = self.xgan_model.generator.predict(noise)
        # Use (1) as label for fake pdfs
        y_disc = np.ones(half_batch_size)
        return pdf_fake, y_disc

    def sample_latent_space(self, batch_size, noise_size):
        """
        This method construct the latent space (that gets fed into the Generator)
        which is a random vector of noises. For instance, it is a Guassian random
        generated variables
        """
        xlatent = np.random.randn(batch_size * noise_size)
        return xlatent.reshape(batch_size, noise_size)

    def latent_prior_noised(self, batch_size, noise_size):
        """
        This method construct the latent space (that gets fed into the Generator)
        by adding noise to the true prior.
        """
        xlatent = []
        for px in self.sampled_pdf[:batch_size]:
            xlatent.append(px + np.random.random(self.output_size) * 0.02)
        return np.array(xlatent)

    def sample_output_noise(self, batch_size):
        """
        This meta-model is used to train the Generator. Only the Random Vector 
        gets fed into the Generator and the generated pdfs will enter in the 
        Critic which in turn produces predicted labels that is used to tune and
        update the Generator Model during the training.

        random_noise|==> Generator ==> generated_pdfs|==>Critic==>Label==>LOSS
                            ^______________________________________________|
                                      TUNING / BACKPROPAGATION
        """
        noise = self.sample_latent_space(batch_size,self.noise_size)
        y_gen = -np.ones(batch_size)
        return noise, y_gen

    def pretrain_disc(self, batch_size, epochs=4):
        xinput, y_disc = self.sample_input_and_gen(batch_size)
        self.xgan_model.critic.trainable = True
        self.xgan_model.critic.fit(xinput, y_disc, epochs=epochs, batch_size=batch_size)

    def train(self, nb_epochs=1000, batch_size=1, verbose=False):
        # Calculate the number of bacthes per training epochs
        batch_per_epoch = int(self.sampled_pdf.shape[0]/batch_size)
        # Calculate the number of training iterations
        nb_steps = batch_per_epoch * nb_epochs
        # Calculate the size of a half batch sample
        if batch_size < 2:
            half_batch_size = 1
        else:
            half_batch_size = int(batch_size/2)
        
        # Initialize the file for the losses
        f = open('%s/losses_info.csv' %self.params['save_output'],'w')
        f.write('Iter., Disc_Loss_Real, Disc_Loss_Fake, Adv_loss, metric\n')
        for k in range(1, nb_steps+1):

            # # Train the Critic
            # # Make sure to train the Critic
            # self.xgan_model.critic.trainable = True
            # for _ in range(self.params['nd_steps']):
            #     xinput, y_disc = self.sample_input_and_gen(batch_size)
            #     dloss = self.xgan_model.critic.train_on_batch(xinput, y_disc)

            # Train the Critic
            # Make sure the Critic is trainable
            self.xgan_model.critic.trainable = True
            for _ in range(self.params['nd_steps']):
                # Train with the real samples
                r_xinput, r_ydisc = self.generate_real_samples(half_batch_size)
                r_dloss = self.xgan_model.critic.train_on_batch(r_xinput, r_ydisc)
                # Train with the real samples
                f_xinput, f_ydisc = self.generate_fake_samples(half_batch_size)
                f_dloss = self.xgan_model.critic.train_on_batch(f_xinput, f_ydisc)
    
            # Train the GAN
            # Make sure that the Critic is not trainable
            self.xgan_model.critic.trainable = False
            for _ in range(self.params['ng_steps']):
                noise, y_gen = self.sample_output_noise(batch_size)
                gloss = self.xgan_model.adversarial.train_on_batch(noise, y_gen)

            # Defines the SMM to be hyperoptimized
            metric = gloss

            # Print log output
            if verbose:
                if k % 100 == 0:
                    print ("Iter:{} out of {}. dloss real: {:6f}. dloss fake: {:6f}. Adv loss: {:6f}"
                            .format(k, nb_steps+1, r_dloss, f_dloss, gloss))
                    f.write("{0}, \t{1}, \t{2}, \t{3} \n".format(k, r_dloss, f_dloss, gloss))
                if k % 1000 == 0:
                    self.plot_generated_pdf(k, self.params['out_replicas'], self.params['save_output'])

        # Close the loss file
        f.close()

        return metric
