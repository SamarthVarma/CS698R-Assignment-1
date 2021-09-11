from gym import error, spaces, utils
import numpy as np
import math


def x(i):
    return i,2


y = np.zeros((6,10))
z = np.zeros((6,10))
z[1][1] = 1
y[4][3] = 2
z = (z + y)/2
print(z)