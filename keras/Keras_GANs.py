import lhapdf
import math
import numpy as np 
from random import sample
import matplotlib.pyplot as plt

from keras import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
from keras.optimizers import Adam, RMSprop, Adadelta
from keras.layers.advanced_activations import LeakyReLU

# Get the PDF4LHC15 for test purpose and print some description
pdf = lhapdf.getPDFSet("NNPDF31_nnlo_as_0118")
print(pdf.description)
pdf_central   = pdf.mkPDF(0)

# Define the scale 
Q_pdf = 1.5
# Define the list of flavors
flavors_list = [1,2]

# Define a function which does the sampling
def sample_pdf(n):
    x_pdf, data  = [], []
    m = math.floor(n/3)
    for i in np.logspace(-3,-1,m): x_pdf.append(i)
    for i in np.linspace(0.1,1,n-m): x_pdf.append(i)
    for x in x_pdf:
        row = []
        for fl in flavors_list:
            row.append(pdf_central.xfxQ2(fl,x,Q_pdf)-pdf_central.xfxQ2(-fl,x,Q_pdf))
        data.append(row)
    return x_pdf, np.array(data)

# Generate the data
x_val, pdf_dataX = sample_pdf(256)
length = pdf_dataX.shape[0]*pdf_dataX.shape[1]
pdf_data = pdf_dataX.reshape(length,)



# Define parameters
random_noise_dim = 100
learning_rate = 0.001
optimizer1 = RMSprop(lr=learning_rate)
optimizer2 = Adadelta() # this one has a different learning rate per weight, just to see

# Generator Architecture 
Generator = Sequential([
    Dense(32, input_dim=random_noise_dim),
    LeakyReLU(0.2),
    Dense(64),
    LeakyReLU(0.2),
    Dense(length)
    ])
Generator.compile(loss='binary_crossentropy', optimizer=optimizer1)

# Discriminator Architercture 
Discriminator = Sequential([
    Dense(256, input_dim = length),
    LeakyReLU(0.2),
    Dropout(0.1),
    Dense(128),
    LeakyReLU(0.2),
    Dropout(0.1),
    Dense(32),
    LeakyReLU(0.2),
    Dropout(0.1),
    Dense(1, activation='sigmoid')
    ])
Discriminator.compile( loss='binary_crossentropy', optimizer=optimizer2)

# Choose to train the Discriminator or Not
always_train_Discriminator = False

gan_input = Input(shape = (random_noise_dim,))
x = Generator(gan_input)
gan_output = Discriminator(x)

if not always_train_Discriminator:
    Discriminator.trainable = False
GAN = Model(inputs = gan_input, outputs = gan_output)
GAN.compile(loss = 'binary_crossentropy', optimizer = optimizer2)

# Set the number of training 
number_training = 10000
batch_size = 1

# Number of steps to train G&D
nd_steps = 4
ng_steps = 1

# Plot pdfs for every x trainings
def plot_generated_pdf(training, generator, pdf_repl=1, figsize=(25, 25)):
    noise = np.random.normal(0, 1, size=[pdf_repl, random_noise_dim])
    generated_pdf = generator.predict(noise)
    generated_pdf = generated_pdf.reshape(pdf_repl, pdf_dataX.shape[0], pdf_dataX.shape[1])

    plt.figure(figsize=figsize)
    for i in range(generated_pdf.shape[0]):
        for j in range(pdf_dataX.shape[1]):
            plt.plot(x_val,pdf_dataX[:,j],color='r')
            plt.plot(x_val,generated_pdf[i][:,j],color='g')
    plt.tight_layout()
    if always_train_Discriminator:
        plt.savefig('gan_generated_pdf_at_training_%d.png' % training)
    else:
        plt.savefig('gan_generated_pdf_at_training_%d.png' % training)

for k in range(1,number_training+1):
    noise = np.random.normal(0,1,size=[batch_size,random_noise_dim])
    pdf_batch = [pdf_data] # put 10 members here so you compare 10 trues, 10 fakes each time
    pdf_fake  = Generator.predict(noise)

    xinput = np.concatenate([pdf_batch,pdf_fake])

    for m in range(nd_steps): 
        # Train the Discriminator with trues and fakes
        y_Discriminator = np.zeros(2*batch_size)
        y_Discriminator[:batch_size] = 1.0
        if not always_train_Discriminator:
            Discriminator.trainable = True  # Ensure the Discriminator is trainable
        Discriminator.train_on_batch(xinput, y_Discriminator)

    for n in range(ng_steps):
        # Train the generator by generating fakes and lying to the Discriminator
        noise = np.random.normal(0, 1, size = [batch_size, random_noise_dim])
        y_gen = np.ones(batch_size)                                   
        if not always_train_Discriminator:
            Discriminator.trainable = False
        GAN.train_on_batch(noise, y_gen)

    if k % 1000 ==0:
        print("training {0}".format(k))
        plot_generated_pdf(k,Generator)
