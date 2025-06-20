import numpy as np
import matplotlib.pyplot as plt
data = []

def f(num):
    try: 
        return float(num)
    except:
        return float('Nan')

for i in range(60,100):
    try:
        with open(f'datapoint{i}.txt', 'r') as file:
            lines = file.readlines()
            data.append([j[1:-2].split(', ') for j in lines])
    except:
        break
data = data[10:]
data2 = [[f(el[1]) for el in row] for row in data]
data1 = [[f(el[0]) for el in row] for row in data]
for el in data1:
    print(len(el))
data1 = np.array(data1, dtype='float64').transpose()-13
data2 = np.array(data2, dtype='float64').transpose()
data = data2/data1
errorbars1 = np.array([np.nanstd(j) for j in data])
totdata1 = np.array([np.nanmean(j) for j in data],dtype='float64')
 
xvals1 = np.array(list(range(1000,10100,100)), dtype='float64')
# xvals = np.array(list(range(10,205,3)), dtype='float64')
data=[]
for i in range(60,100):
    try:
        with open(f'datapoint{i}.txt', 'r') as file:
            lines = file.readlines()
            data.append([j[1:-2].split(', ') for j in lines])
    except:
        break
data=data[:10]
data2 = [[f(el[1]) for el in row] for row in data]
data1 = [[f(el[0]) for el in row] for row in data]
data1 = np.array(data1, dtype='float64').transpose()-13
data2 = np.array(data2, dtype='float64').transpose()
data = data2/data1
errorbars2 = np.array([np.nanstd(j) for j in data])
totdata2 = np.array([np.nanmean(j) for j in data],dtype='float64')
 
xvals2 = np.array(list(range(10000,101000,1000)), dtype='float64')

plt.plot(xvals1,totdata1, color='blue')
plt.plot(xvals1,totdata1+errorbars1,'b:')
plt.plot(xvals1,totdata1-errorbars1,'b:')
# plt.plot(xvals2,totdata2, color='blue')
# plt.plot(xvals2,totdata2+errorbars2,'b:')
# plt.plot(xvals2,totdata2-errorbars2,'b:')
plt.xlabel(r'$\Delta t \: (\mu s)$')
plt.ylabel('Average maximum branch length (pixels)')
# plt.xscale('log')
ax= plt.gca()
ax.axhline(y=np.mean(totdata2), color='blue')
plt.show()


        
    
