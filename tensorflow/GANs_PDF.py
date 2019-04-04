# Import the Libraries
import lhapdf
import math
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
pdf_central = pdf.mkPDF(0)

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
    for x in x_pdf:
        y_pdf1 = pdf_central.xfxQ2(21,x,Q_pdf)/5                                # gluon
        y_pdf2 = pdf_central.xfxQ2(2,x,Q_pdf)-pdf_central.xfxQ2(-2,x,Q_pdf)     # valence u_quark
        y_pdf3 = pdf_central.xfxQ2(1,x,Q_pdf)-pdf_central.xfxQ2(-1,x,Q_pdf)     # valence u_quark
        data.append([x,y_pdf1,y_pdf2,y_pdf3])
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
hidden = [16,16]
# Take the data shape
data_shape = sample_pdf().shape[1]


# Implement the GENERATOR Model

# The following function takes the follwoing as input:
# A Random Sample Z
# A list which contains the Structure of the NN (layers)
# A variable called "reuse"
def generator(input_noise, layers_size=hidden,reuse=False):
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
def discriminator(input_true,layers_size=hidden,reuse=False):
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
X = tf.placeholder(tf.float32,[None,data_shape])
# Initialize the placeholder for the random sample
Z = tf.placeholder(tf.float32,[None,data_shape])

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

batch_size = 256
nd_steps   = 10
ng_steps   = 10

# Fetch the real PDF in order to plot them
x_plot = sample_pdf(n=batch_size)

f = open('loss.csv','w')
f.write('Iteration,Discriminator Loss,Generator Loss\n')

numb_training = 10001

# Training
for i in range(numb_training):
    X_batch = sample_pdf(n=batch_size)
    Z_batch = sample_noise(batch_size,data_shape)

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
        
        # Fetch the generated data
        g_plot = sess.run(G_sample, feed_dict={Z: Z_batch})

        plt.figure()
        xax_g = plt.scatter(x_plot[:,0],x_plot[:,1],s=14)
        gax_g = plt.scatter(g_plot[:,0],g_plot[:,1],s=14)

        xax_u = plt.scatter(x_plot[:,0],x_plot[:,2],s=14)
        gax_u = plt.scatter(g_plot[:,0],g_plot[:,2],s=14)

        xax_d = plt.scatter(x_plot[:,0],x_plot[:,3],s=14)
        gax_d = plt.scatter(g_plot[:,0],g_plot[:,3],s=14)

        plt.legend((xax_g,gax_g,xax_u,gax_u,xax_d,gax_d), ("xg/5 Real PDF","xg/5 Generated PDF","xu_v Real PDF","xu_v Generated PDF","xd_v Real PDF","xd_v Generated PDF"))
        plt.title('Samples at Iteration %d'%i)
        plt.tight_layout()
        plt.savefig('iterations/iteration_%d.png'%i, dpi=250)
        plt.close()

f.close()