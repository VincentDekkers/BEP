from matplotlib import pyplot as plt
import csv
import numpy as np
data = []
with open('test_00000_10.csv') as file:
    reader = csv.reader(file)
    for row in reader:
        data.append(row)
data = np.array([[eval(el) for el in line] for line in data[2:]]).transpose()
plt.plot(data[0], data[1])
plt.plot(data[0],data[2])
plt.plot(data[0],data[3])
plt.show()

