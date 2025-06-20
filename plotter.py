import numpy as np
import matplotlib.pyplot as plt
data = []

for i in range(100):
    try:
        with open(f'67mbar/datapoint{i}.txt', 'r') as file:
            lines = file.readlines()
            data.append([float(j[:-1]) for j in lines])
    except:
        break
data = np.array(data).transpose()
errorbars = np.array([np.std(j) for j in data])
totdata = np.array([sum(j) for j in data],dtype='float64')
totdata /= i 
xvals = np.array(list(range(2,len(totdata)+2)), dtype='float64')
# xvals = np.array(list(range(10,205,3)), dtype='float64')
xvals /= 2

# plt.plot(xvals,totdata, color='blue')
# plt.plot(xvals,totdata+errorbars,'b:')
# plt.plot(xvals,totdata-errorbars,'b:')
# plt.xlabel(r'$\Delta t \: (\mu s)$')
# plt.ylabel('Average maximum branch length (pixels)')
# plt.xscale('log')
data2=[]
for k in range(100):
    try:
        with open(f'67mbar3/datapoint{k}.txt', 'r') as file:
            lines = file.readlines()
            data2.append([float(j[:-1]) for j in lines])
    except:
        break
data2 = np.array(data2).transpose()
data2 = data/data2
errorbars2 = np.array([np.std(j) for j in data2])
totdata2 = np.array([sum(j) for j in data2],dtype='float64')
totdata2 /= k 
# xvals2 = np.array(list(range(2,101)), dtype='float64')
# xvals2 = np.array(list(range(10,205,3)), dtype='float64')
# xvals2 /= 2

# plt.plot(xvals2,totdata2, color='blue')
# plt.plot(xvals2,totdata2+errorbars2,'b:')
# plt.plot(xvals2,totdata2-errorbars2,'b:')
plt.plot(xvals, totdata2)

plt.show()
        
    
