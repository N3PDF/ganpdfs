import csv
import seaborn as sb
import matplotlib.pyplot as plt
sb.set_style("whitegrid")

with open('loss.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    iteration, DisLoss, GenLoss = [], [], []
    for row in csv_reader:
        if line_count == 0:
            pass
            line_count += 1
        else:
            iteration.append(float(row[0]))
            DisLoss.append(float(row[1]))
            GenLoss.append(float(row[3]))
            line_count += 1

plt.plot(iteration,DisLoss)
plt.figure()
dis = plt.plot(iteration,DisLoss)
gen = plt.plot(iteration,GenLoss)
plt.legend([dis[0],gen[0]], ("Discriminator Loss","Generator Loss"))
plt.title('GANs Loss')
plt.tight_layout()
plt.savefig('loss.png', dpi=150)
plt.close()