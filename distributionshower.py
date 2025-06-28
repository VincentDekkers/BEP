import numpy as np
import matplotlib.pyplot as plt


xvals = np.linspace(-1,1,1001)
yvals = 1/(np.pi * np.sqrt(1-xvals**2))
plt.plot(xvals,yvals)
plt.ylim([0,1.5])
plt.gca().axvline(x=0, color='grey')
plt.show()