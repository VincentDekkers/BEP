from matplotlib import pyplot as plt
import csv
import numpy as np
totdata = []
for i in range(1,4):
    data = []    
    with open(f'C{i}--pulse20mbar--00000.csv') as file:
        reader = csv.reader(file)
        for row in reader:
            data.append(row)
        data = list(np.array([[eval(el) for el in line] for line in data[6:]]).transpose())
    totdata += data

fig, ax1 = plt.subplots()
totdata[0] = np.array(totdata[0])*10**6
color = 'tab:red'
ax1.set_xlabel(r'time ($\mu s$)')
ax1.set_ylabel('voltage (V)', color=color)
ax1.plot(totdata[0], totdata[1], color=color)
ax1.tick_params(axis='y', labelcolor=color)
# ax1.arrow(-0.15,2500,0.5,0, width=0.1)

ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('current (A)', color=color)  # we already handled the x-label with ax1
ax2.plot(totdata[0], totdata[3], color=color)
ax2.tick_params(axis='y', labelcolor=color)
ax2.plot(totdata[0], np.array(totdata[5])/40, color ='green')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.xlim([-6*10**-1,2.5])
plt.show()

