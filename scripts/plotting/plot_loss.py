#!/usr/bin/env python3
import csv
import argparse
import seaborn as sb
import matplotlib.pyplot as plt
sb.set_style("whitegrid")

def read_csv(csv_reader):
    line_count = 0
    iteration, DisLoss = [], []
    GenLoss, KL_valu   = [], []
    for row in csv_reader:
        if line_count == 0:
            pass
            line_count += 1
        else:
            iteration.append(float(row[0]))
            DisLoss.append(float(row[1]))
            GenLoss.append(float(row[2]))
            KL_valu.append(float(row[5]))
            line_count += 1
    return iteration, DisLoss, GenLoss, KL_valu


def plot_losses(iteration, DisLoss, GenLoss, KL_valu):

    # plot the losses
    plt.figure()
    dis = plt.plot(iteration,DisLoss)
    gen = plt.plot(iteration,GenLoss)
    plt.legend([dis[0],gen[0]], ("Generator Loss","Discriminator Loss"))
    plt.title('GANs Loss')
    plt.tight_layout()
    plt.savefig('losses.png', dpi=150)
    plt.close()


def main(args):
    """
    Load the loss file and plot the results.
    """
    # Open the .csv file
    with open(args.losses) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        iteration, DisLoss, GenLoss, KL_valu = read_csv(csv_reader)

    # Plot the losses
    plot_losses(iteration, DisLoss, GenLoss, KL_valu)

if __name__ == "__main__":
    """
    Read command line arguments.
    """
    parser = argparse.ArgumentParser(description='Analyse GANPDFs losses.')
    parser.add_argument('losses', help='Take as input a .csv file with loss values.')
    args = parser.parse_args()
    main(args)
