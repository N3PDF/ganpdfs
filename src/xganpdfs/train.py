import os, sys
import numpy as np
import json, tempfile
from scipy import stats
import matplotlib.pyplot as plt
from keras.models import Sequential
from xganpdfs.custom import xmetrics
from xganpdfs.pdformat import input_pdfs
from xganpdfs.model import dc_xgan_model
from xganpdfs.model import vanilla_xgan_model
from tensorflow.python.client import timeline


class TimeLiner:
        _timeline_dict = None

        def update_timeline(self, chrome_trace):
            # convert crome trace to python dict
            chrome_trace_dict = json.loads(chrome_trace)
            # for first run store full trace
            if self._timeline_dict is None:
                self._timeline_dict = chrome_trace_dict
                # for other - update only time consumption, not definitions
            else:
                for event in chrome_trace_dict['traceEvents']:
                    #events time consumption started with 'ts' prefix
                    if 'ts' in event:
                        self._timeline_dict['traceEvents'].append(event)

        def save(self, f_name):
            with open(f_name, 'w') as f:
                json.dump(self._timeline_dict, f)

class xgan_train(object):

    """
    Train the model and plot the result.
    """

    def __init__(self, x_pdf, pdf_name, noise_size, params, activ, optmz, nb_replicas=1, Q_value=1.7874388, flavors=2):
        self.sampled_pdf = input_pdfs(pdf_name, x_pdf, nb_replicas, Q_value, flavors).build_pdf()
        self.x_pdf = x_pdf
        self.nb_replicas = nb_replicas
        self.output_size = len(x_pdf)
        self.noise_size = noise_size
        self.params = params
        self.run_timeline = TimeLiner()
        self.xgan_model = vanilla_xgan_model(noise_size, self.output_size, x_pdf, params, activ, optmz)

    def plot_generated_pdf(self, nth_training, nrep, folder):
        if not os.path.exists('%s/iterations' % folder):
            os.mkdir('%s/iterations' % folder)
        else:
            pass
        noise = np.random.normal(0, 1, size=[nrep, self.noise_size])
        generated_pdf = self.xgan_model.generator.predict(noise)

        grid_size = (4,4)
        xv = [64, 69, 72, 75]

        nfake_bins = 15
        # ntrue_bins = (nfake_bins*self.sampled_pdf.shape[0])//generated_pdf.shape[0]
        ntrue_bins = nfake_bins

        fig = plt.figure(figsize=(11,14))
        main  = plt.subplot2grid(grid_size, (0,0), colspan=2, rowspan=2)
        hist1 = plt.subplot2grid(grid_size, (0,2))
        hist2 = plt.subplot2grid(grid_size, (0,3))
        hist3 = plt.subplot2grid(grid_size, (1,2))
        hist4 = plt.subplot2grid(grid_size, (1,3))

        l = [hist1, hist2, hist3, hist4]

        for i in range(self.sampled_pdf.shape[0]):
            main.plot(self.x_pdf, self.sampled_pdf[i], color='deeppink', alpha=0.45)
        for j in range(generated_pdf.shape[0]):
            main.plot(self.x_pdf, generated_pdf[j], color='dodgerblue', alpha=0.45)

        for x, position in zip(xv, l):
            true_hist = np.array([repl[x] for repl in self.sampled_pdf])
            fake_hist = np.array([repl[x] for repl in generated_pdf])
            position.hist(true_hist, histtype='stepfilled', bins=15,
                            color="deeppink", alpha=0.65, label="true", density=True)
            position.hist(fake_hist, histtype='stepfilled', bins=15,
                            color="dodgerblue", alpha=0.65, label="fake", density=True)

        fig.suptitle('Samples at Iteration %d'%nth_training, y=0.98)
        fig.tight_layout()
        fig.savefig('%s/iterations/pdf_generated_at_training_%d.png' %(folder,nth_training), dpi=250,
                bbox_inches = 'tight', pad_inches = 0.2)
        fig.legend()
        # fig.close()

    def sample_input_and_gen(self, batch_size):
        pdf_index = np.random.choice(self.sampled_pdf.shape[0], batch_size, replace=False)
        pdf_batch = self.sampled_pdf[pdf_index]
        noise = np.random.normal(0,1,size=[batch_size, self.noise_size])
        pdf_fake = self.xgan_model.generator.predict(noise)
        xinput = np.concatenate([pdf_batch, pdf_fake])
        y_disc = np.zeros(2*batch_size)
        y_disc[:batch_size] = 1.0
        return xinput, y_disc

    def sample_output_noise(self, batch_size):
        noise = np.random.normal(0,1,size=[batch_size,self.noise_size])
        y_gen = np.ones(batch_size)
        return noise, y_gen

    def pretrain_disc(self, batch_size, epochs=4):
        xinput, y_disc = self.sample_input_and_gen(batch_size)
        self.xgan_model.discriminator.trainable = True
        self.xgan_model.discriminator.fit(xinput, y_disc, epochs=epochs, batch_size=batch_size)

    def test_model(self, nrep):
        noise = np.random.normal(0, 1, size=[nrep, self.noise_size])
        generated_pdf = self.xgan_model.generator.predict(noise)
        # First test for one input Replicas only
        return xmetrics(self.sampled_pdf, generated_pdf).euclidean()

    def train(self, nb_training=20000, batch_size=1, verbose=False):
        f = open('%s/losses_info.csv' %self.params['save_output'],'w')
        f.write('Iter., Disc_Loss, Gen_Loss, Disc_acc, Gen_acc, metric\n')
        for k in range(1, nb_training+1):
            for _ in range(int(self.sampled_pdf.shape[0]/batch_size)):

                # Train the Discriminator
                # Make sure to train the Discriminator
                self.xgan_model.discriminator.trainable = True
                for _ in range(self.params['nd_steps']):
                    xinput, y_disc = self.sample_input_and_gen(batch_size)
                    dloss = self.xgan_model.discriminator.train_on_batch(xinput, y_disc)

                # Train the GAN
                # Make sure that the Discriminator is not trained
                self.xgan_model.discriminator.trainable = False
                for _ in range(self.params['ng_steps']):
                    noise, y_gen = self.sample_output_noise(batch_size)
                    gloss = self.xgan_model.gan.train_on_batch(noise, y_gen)

            # # Evaluate performance using KL divergence
            # metric = self.test_model(self.params['out_replicas'])
            metric = gloss[0]

            # timeline save
            if self.xgan_model.options is not None:
                trace = timeline.Timeline(step_stats=self.xgan_model.metadata.step_stats)
                fetch = trace.generate_chrome_trace_format()
                self.run_timeline.update_timeline(fetch)

            if verbose:

                if k % 100 == 0:
                    print ("Iter: {} out of {} . Disc loss: {:6f} . Gen loss: {:6f} . metric: {:6f}"
                            .format(k, nb_training, dloss[0], gloss[0], metric))
                    f.write("{0}, \t{1}, \t{2}, \t{3}, \t{4}, \t{5}\n".format(k,dloss[0],gloss[0],dloss[1],gloss[1],metric))

                if k % 1000 == 0:
                    self.plot_generated_pdf(k, self.params['out_replicas'], self.params['save_output'])

        self.run_timeline.save('test.json')
        f.close()

        return metric
