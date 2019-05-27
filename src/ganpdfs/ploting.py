import random
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from scipy import stats
sb.set_style("whitegrid")

# Plot pdfs for every x trainings
# Generate random colors
# col1 = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])for i in range(len(x_val))]
# col2 = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])for i in range(len(x_val))]
col1 = ['r','m']
col2 = ['g','b']
def plot_generated_pdf(x_val, pdfData, noise_dim, training, generator, dis_train, pdf_repl=1):
    noise = np.random.normal(0, 1, size=[pdf_repl, noise_dim])
    generated_pdf = generator.predict(noise)
    # generated_pdf = generated_pdf.reshape(pdf_repl, pdfData.shape[0], pdfData.shape[1])

    # import pdb
    # pdb.set_trace()

    plt.figure()
    for i in range(generated_pdf.shape[0]):
        # for j in range(pdfData.shape[1]):
        plt.plot(x_val,pdfData[i],color='blue', alpha=0.75)
        plt.plot(x_val,generated_pdf[i],color='red', alpha=0.75)
    if bool(dis_train):
        plt.title('Samples at Iteration %d'%training)
        plt.tight_layout()
        plt.savefig('iterations/gan_dis_on_generated_pdf_at_training_%d.png' % training, dpi=250)
    else:
        plt.title('Samples at Iteration %d'%training)
        plt.tight_layout()
        plt.savefig('iterations/gan_generated_pdf_at_training_%d.png' % training, dpi=250)
    plt.close()

def plot_generated_repl(x_val, generator, nb_repl, noise_dim, xo_index):
    noise = np.random.normal(0, 1, size=[nb_repl, noise_dim])
    gen_repl = generator.predict(noise, batch_size=nb_repl)

    # Plot the generated Replicas
    plt.figure()
    for repl in gen_repl:
        plt.plot(x_val, repl, color='blue', alpha=0.45, linewidth=0.45)
    plt.tight_layout()
    plt.savefig('gen_repl.png', dpi=250)
    plt.close()

    # Plot the distreibution for a given x as an histogram
    plt.figure()
    for xo in xo_index:
        gen_hist = np.array([repl[xo] for repl in gen_repl])
        plt.hist(gen_hist, bins=25, histtype='step', stacked=True, fill=False)
    plt.savefig('hist_gen.png', dpi=250)
    plt.close()
