# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from scipy.integrate import solve_ivp  # https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html#r179348322575-1
import h5py
# import os.path.join as pjoin

f = h5py.File('F:/TONSTAD_TWO/Q40.mat', 'r') # https://docs.h5py.org/en/stable/quick.html#quick
 # list(f.keys())
 # ['#refs#', 'LEUC', 'LSUC', 'UEUC', 'USUC', 'Umx', 'Vmx', 'x', 'y']
 # x.shape
 
x = f['x'][0]
y = f['y'][0]

Umx = f['Umx']
Vmx = f['Vmx']

Um = np.nanmean(Umx,0)
Vm = np.nanmean(Vmx,0)

up = Umx - Um
vp = Vmx - Vm




# %%

import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate

x = [0, 0, 1, 1, 2, 2,  0,    1,   2]
y = [1, 2, 1, 2, 1, 2, 1.5, 1.5, 1.5]
u = [0.5, -1, 0, 0, np.nan, 1, 0, 0, 0.75]
v = [1, 1, 1, 1, np.nan, 1, 1, 1, 1]

plt.figure(1)
plt.quiver(x, y, u, v)

xx = np.linspace(0, 2, 10)
yy = np.linspace(1, 2, 10)

xx= np.array([0.25,1.0])
yy= np.array([1.9,1.8])
# xx, yy = np.meshgrid(xx, yy)

points = np.transpose(np.vstack((x, y)))
u_interp = interpolate.griddata(points, u, (xx, yy), method='cubic')  # https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.griddata.html
v_interp = interpolate.griddata(points, v, (xx, yy), method='cubic')

plt.figure(1)
plt.quiver(xx, yy, u_interp, v_interp)
plt.show()



# u_interp = interpolate.griddata(np.transpose(np.vstack(([0, 0, 1, 1, 2, 2,  0,    1,   2], [1, 2, 1, 2, 1, 2, 1.5, 1.5, 1.5]))), u, (xx, yy), method='cubic')