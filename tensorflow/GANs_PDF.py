# Import the Libraries
import lhapdf
import math
import random
import tensorflow as tf
import numpy as np
import seaborn as sb
from random import sample
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
sb.set_style("whitegrid")

# Get the PDF4LHC15 for test purpose and print some description
pdf = lhapdf.getPDFSet("NNPDF31_nnlo_as_0118")
print(pdf.description)
# Take only the central value
pdf_init = pdf.mkPDFs()
pdf_central = sample(pdf_init,25)
size_member = len(pdf_central)

# Define the scale 
Q_pdf = 10

# Define a log uniform function
def loguniform(low=0, high=1, size=None):
    return np.exp(np.random.uniform(low, high, size))

# Define a function which does the sampling
def sample_pdf(n=1000):
    data  = []
    x_pdf = []
    m = math.floor(n/3)
    # Take m random values between 1e-3 and 1e-1 in logscale
    for i in loguniform(low=-3,high=-1,size=m): x_pdf.append(i)
    # Taake (n-m) values between 0.1 and 1 
    for i in np.random.uniform(0.1,1,n-m): x_pdf.append(i)
    # Construct the sampling
    flavors_list = [1,2,3,21]
    for p in pdf_central:
        repl = []
        for x in x_pdf:
            row = [x]
            for fl in flavors_list:
                if (fl<3): row.append(p.xfxQ2(fl,x,Q_pdf)-p.xfxQ2(-fl,x,Q_pdf))
                else : row.append(p.xfxQ2(fl,x,Q_pdf)/3)
            repl.append(row)
        data.append(repl)
    return np.array(data)

# Define the function which generates the noise data
# The shape of the input noise does not have to be same as the true pdf inputs
# It can be a 1-dimensional vector (but this does not perform well as the below)
def sample_noise(m, n):
    return np.random.uniform(0,1.,size=[m, n])

# Define a function which sorts a multi-dimensional list wtr to the 1st row
def sorting(lst):
    new_lst = []
    ordering = lst[0].argsort()
    for i in range(len(lst)):
        new_lst.append(lst[i][ordering])
    return new_lst

# Define the hidden layers 
nb_points = 256
hidden_gen = [16,32]
hidden_dis = [32,16]
# Take the data shape
data_shape = sample_pdf(n=nb_points).shape[2]

# Implement the GENERATOR Model

# The following function takes the follwoing as input:
# A Random Sample Z
# A list which contains the Structure of the NN (layers)
# A variable called "reuse"
def generator(input_noise, layers_size=hidden_gen,reuse=False):
    # Create and Share a variable named "Generator"
    # Here "reuse" allows us to share the variable
    with tf.variable_scope("Generator",reuse=reuse):
        # Define the 1st and 2nd layer with "leaky_relu" as an activation fucnction
        L1 = tf.layers.dense(input_noise,layers_size[0],activation=tf.nn.leaky_relu)
        L2 = tf.layers.dense(L1,layers_size[1],activation=tf.nn.leaky_relu)
        # Define the output layer with data_shape nodes
        # This dimension is correspond to the dimension of the "real dataset"
        output = tf.layers.dense(L2,data_shape)
    return output

# Implement the DISCRIMINATOR Model

# The following function takes the following as input:
# A sample from the "REAL" dataset
# A list which contains the structure of the first 2 "hidden layer"
# A variable called "reuse"
def discriminator(input_true,layers_size=hidden_dis,reuse=False):
    # Create and share a variable named "Discriminator"
    with tf.variable_scope("Discriminator",reuse=reuse):
        # Define the 1st and 2nd layer with "leaky_relu" as an activation fucnction
        L1 = tf.layers.dense(input_true,layers_size[0],activation=tf.nn.leaky_relu)
        L2 = tf.layers.dense(L1,layers_size[1],activation=tf.nn.leaky_relu)
        # Fix the third layer to 2 nodes so we can visualize the transformed feature space in a 2D plane
        L3 = tf.layers.dense(L2,data_shape)
        # Define the output layer (logit)
        output = tf.layers.dense(L3,1)
    return output, L3

# Adversarial Training

# Initialize the placeholder for the real sample
X = tf.placeholder(tf.float32,[None,nb_points,data_shape])
# Initialize the placeholder for the random sample
Z = tf.placeholder(tf.float32,[None,nb_points,data_shape])

# Define the Graph which Generate fake data from the Generator and feed the Discriminator
G_sample = generator(Z)
r_logits, r_rep = discriminator(X)
f_logits, g_rep = discriminator(G_sample,reuse=True)

# Define the Loss Function for G and D
DiscriminatorLoss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=r_logits,labels=tf.ones_like(r_logits)) +
                           tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logits,labels=tf.zeros_like(f_logits)))
GeneratorLoss     = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logits,labels=tf.ones_like(f_logits)))

# Collect the variables in the graph
GeneratorVars     = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="Generator")
DiscriminatorVars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="Discriminator")

# Define the Optimizer for G&D
GeneratorStep     = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(GeneratorLoss,var_list = GeneratorVars)
DiscriminatorStep = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(DiscriminatorLoss,var_list = DiscriminatorVars)

# sess = tf.Session(config=config)
sess = tf.Session()
tf.global_variables_initializer().run(session=sess)

nd_steps   = 10
ng_steps   = 10

# Fetch the real PDF in order to plot them
x_plot  = sample_pdf(n=nb_points)
# sx_plot = x_plot[x_plot[:,0].argsort()] 

f = open('loss.csv','w')
f.write('Iteration,Discriminator Loss,Generator Loss\n')

# Plot pdfs for every x trainings
# Generate random colors
col1 = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])for i in range(data_shape)]
col2 = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])for i in range(data_shape)]
def plot_generated_pdf(generator, noise, iteration, replicas, shape_data=data_shape, s=14, a=0.95):
    g_plot  = sess.run(generator, feed_dict={Z: noise})
    # sg_plot = g_plot[g_plot[:,0].argsort()] 

    plt.figure()
    for r in range(replicas):
        for gen in range(1,shape_data):
            plt.scatter(x_plot[r][:,0],x_plot[r][:,gen],color=col1[gen],s=14,alpha=a)
            plt.scatter(g_plot[r][:,0],g_plot[r][:,gen],color=col2[gen],s=14,alpha=a)
    plt.title('Samples at Iteration %d'%iteration)
    plt.xlim([0.001,1])
    plt.ylim([0,0.8])
    plt.tight_layout()
    plt.savefig('iterations/iteration_%d.png'%iteration, dpi=250)
    plt.close()

numb_training = 10001
batch_size  = 5
batch_count = int(sample_pdf().shape[0] / batch_size)

# Training
for i in range(numb_training):
    X_batch = sample_pdf(n=nb_points)[np.random.randint(0, sample_pdf().shape[0], size = batch_size)]
    Z_batch = [sample_noise(nb_points,data_shape) for i in range(batch_size)]

    # Train independently G&D in multiple steps
    for _ in range(nd_steps):
        _, dloss = sess.run([DiscriminatorStep, DiscriminatorLoss], feed_dict={X: X_batch, Z: Z_batch})

    for _ in range(ng_steps):
        _, gloss = sess.run([GeneratorStep, GeneratorLoss], feed_dict={Z: Z_batch})

    print ("Iterations: %d\t out of %d\t. Discriminator loss: %.4f\t Generator loss: %.4f"%(i,numb_training-1,dloss,gloss))
    if i%100 == 0:
        f.write("%d,%f,%f\n"%(i,dloss,gloss))

    # Plot each 1000 iteration
    if i%1000 == 0:
        plot_generated_pdf(G_sample, Z_batch, i, 5, shape_data=data_shape)

f.close()