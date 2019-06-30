import os, sys
import json, tempfile
import numpy as np
import matplotlib.pyplot as plt
from xganpdfs.custom import xlayer
from xganpdfs.pdformat import input_pdfs
from xganpdfs.model import dc_xgan_model
from xganpdfs.model import vanilla_xgan_model
from keras.models import Sequential
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

        self.xgan_model = dc_xgan_model(noise_size, self.output_size, x_pdf, params, activ, optmz)
        # self.xgan_model.generator = self.xgan_model.generator()
        # self.xgan_model.discriminator = self.xgan_model.discriminator()
        # self.xgan_model.gan = self.xgan_model.gan()

    def plot_generated_pdf(self, nth_training, nrep, folder):
        if not os.path.exists('%s/iterations' % folder):
            os.mkdir('%s/iterations' % folder)
        else:
            pass
        noise = np.random.normal(0, 1, size=[nrep, self.noise_size])
        generated_pdf = self.xgan_model.generator.predict(noise)

        plt.figure()
        for i in range(self.sampled_pdf.shape[0]):
            plt.plot(self.x_pdf, self.sampled_pdf[i], color='blue', alpha=0.45)
        for j in range(generated_pdf.shape[0]):
            plt.plot(self.x_pdf, generated_pdf[j], color='red', alpha=0.45)
        plt.title('Samples at Iteration %d'%nth_training)
        plt.tight_layout()
        plt.savefig('%s/iterations/pdf_generated_at_training_%d.png' %(folder,nth_training), dpi=250)
        plt.close()

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


    def train(self, nb_training=20000, batch_size=1, verbose=False):
        f = open('%s/losses_info.csv' %self.params['save_output'],'w')
        f.write('Iter., Disc_Loss, Gen_Loss, Disc_acc, Gen_acc\n')
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

            # timeline save
            if self.xgan_model.options is not None:
                trace = timeline.Timeline(step_stats=self.xgan_model.metadata.step_stats)
                fetch = trace.generate_chrome_trace_format()
                self.run_timeline.update_timeline(fetch)

            if verbose:

                if k % 100 == 0:
                    print ("Iterations: %d\t out of %d\t. Discriminator loss: %.4f\t Generator loss: %.4f"
                            %(k, nb_training, dloss[0], gloss[0]))
                    f.write("%d,\t%f,\t%f,\t%f,\t%f\n" % (k,dloss[0],gloss[0],dloss[1],gloss[1]))

                if k % 1000 == 0:
                    self.plot_generated_pdf(k, self.params['out_replicas'], self.params['save_output'])

        self.run_timeline.save('test.json')
        f.close()

        return gloss
