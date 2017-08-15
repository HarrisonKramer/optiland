# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


# ==========================================

P = 1
w = 0.5
lap = 0.25
N = 50


# ==========================================


def gaussian(dx,dy):
    x = np.linspace(-3,3,N)
    [X,Y] = np.meshgrid(x+dx,x+dy)
    R = np.sqrt(X**2+Y**2)

    return P/(np.pi*w**2/2)*np.exp(-2*R**2/w**2)
    

dx = np.linspace(-1,1,N)
dy = np.linspace(-1,1,N)

tot = np.zeros((N,N))
for m in range(N):
    for n in range(N):
        tot += gaussian(dx[m],dy[n])


fig, ax = plt.subplots()

ax.imshow(tot)
plt.show()