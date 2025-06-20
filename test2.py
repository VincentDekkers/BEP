# import matplotlib.pyplot as plt
# import numpy as np
# data = []

# for i in range(1,100):
#     try:
#         with open(f'datapoint{i}.txt','r') as file:
#             lines = file.readlines()
#             lines = [[float(i) for i in line[1:-3].split(', ')] for line in lines]
#             data.append([j[0] for j in lines])
#     except:
#         break
# data = np.array(data).transpose()
# errorbars = np.array([np.std(j) for j in data])
# totdata = np.array([sum(j) for j in data],dtype='float64')
# totdata /= i
# xvals = np.array(list(range(2,101)), dtype='float64')
# # xvals = np.array(list(range(10,205,3)), dtype='float64')
# xvals /= 2
# plt.plot(xvals,totdata, color='blue')
# plt.plot(xvals,totdata+errorbars,'b:')
# plt.plot(xvals,totdata-errorbars,'b:')
# plt.xlabel(r'$\Delta t \: (\mu s)$')
# plt.ylabel('Average maximum branch length (pixels)')
# plt.show()


# import matplotlib.cm as cmx
# import matplotlib.colors as colors
# import numpy as np
# import cv2
# import matplotlib.pyplot as plt

# w,h = 1280,720
# values = np.array([[i+j for i in range(w)] for j in range(h)])
# cNorm = colors.Normalize(vmin=0, vmax=np.max(np.max(values)))
# scalarMap = cmx.ScalarMappable(norm=cNorm, cmap='nipy_spectral')
# values = scalarMap.to_rgba(values)[:,:,:3]
# values *= 255
# values = values.astype(np.uint8)

# out = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 10, (w,h))
# for _ in range(100):
#     out.write(values) # frame is a numpy.ndarray with shape (1280, 720, 3)
# out.release()

# import matplotlib.pyplot as plt
# vals = [1,2,4,3]
# plt.plot(vals)
# vals = [2,5,1,3,5,3]
# plt.plot(vals)
# plt.show()



# def is_prime(n):
#     '''
#     Check if the number "n" is prime, with n > 1.

#     Returns a boolean, True if n is prime.
#     '''
#     max_val = n ** 0.5
#     stop = int(max_val + 1)
#     for i in range(2, stop):
#         if n % i == 0:
#             return False
#     return True

# def find_primes(size):
#     primes = []
#     for n in range(size):
#         flag = is_prime(n)
#         if flag:
#             primes.append(n)
#     return primes


# def main():
#     print('start calculating')
#     primes = find_primes(100000)
#     print(f'done calculating. Found {len(primes)} primes.')


# if __name__ == '__main__':
#     main()
# import math
# from numba import jit
# import numpy as np
# import time

# @jit(nopython=True)
# def distanceparallel():
#     for p in range(1000000):
#         i = []
#         if p % 100000 == 0:
#                 print(p)
#         for j in range(100):
            
#             i.append(j)
    
# start = time.time()

# distanceparallel()
# stop = time.time()


# print(stop-start)


# string = 'Hoi allemaal, ik ben\nVincent'
# print(string[:-7])
# print(string[:-8])
# print(string[:-9])


from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 

img = Image.open("Capture4.JPG")
draw = ImageDraw.Draw(img)
# font = ImageFont.truetype(<font-file>, <font-size>)
font = ImageFont.truetype("Qdbettercomicsans-jEEeG.ttf",size=50)
# draw.text((x, y),"Sample Text",(r,g,b))
draw.text((300, 500),"HALLO JIP",(255,255,255), font=font)
draw.text((301, 499),"HALLO JIP",(0,0,0), font=font)
img.save('sample-out.jpg')