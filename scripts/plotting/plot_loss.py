#!/usr/bin/env python3
import csv
import argparse
import seaborn as sb
import matplotlib.pyplot as plt
sb.set_style("whitegrid")

def read_csv(csv_reader):
    line_count = 0
    GenLoss, KL_valu   = [], []
    iteration, DisLossReal, DisLossFake = [], [], []
    for row in csv_reader:
        if line_count == 0:
            pass
            line_count += 1
        else:
            iteration.append(float(row[0]))
            DisLossReal.append(float(row[1]))
            DisLossFake.append(float(row[2]))
            GenLoss.append(float(row[3]))
            line_count += 1
    return iteration, DisLossReal, DisLossFake, GenLoss


def plot_losses(iteration, DisLossReal, DisLossFake, GenLoss):

    # plot the losses
    plt.figure()
    r_dis = plt.plot(iteration, DisLossReal)
    f_dis = plt.plot(iteration, DisLossFake)
    gen   = plt.plot(iteration, GenLoss)
    plt.legend([r_dis[0],f_dis[0],gen[0]], ("Dis_real", "Dis_fake", "Gen"))
    plt.title('GANs Losses')
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
        iteration, DisLossReal, DisLossFake, GenLoss  = read_csv(csv_reader)

    # Plot the losses
    plot_losses(iteration, DisLossReal, DisLossFake, GenLoss)

if __name__ == "__main__":
    """
    Read command line arguments.
    """
    parser = argparse.ArgumentParser(description='Analyse GANPDFs losses.')
    parser.add_argument('losses', help='Take as input a .csv file with loss values.')
    args = parser.parse_args()
    main(args)
