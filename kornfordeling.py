# -*- coding: utf-8 -*-
"""
Created on Sat Jul 10 21:24:39 2021

@author: havrevol
"""
import random
import numpy as np
from scipy import interpolate


PSD = np.array([[0.1,0],
[0.149,0.02],	
[0.297,0.095],	
[0.594,0.28],	
[1.188,0.55],	
[2.376,0.79],	
[4.752,0.91],	
[9.504,0.98],	
[12.0,1]]).T

f = interpolate.interp1d(PSD[1], PSD[0])

tal = int(1e9)

# np.random.seed()
a = np.random.uniform(0.0, 1.0, size = tal)
korndiameter = f(a)
b = np.random.uniform(0.0, 88.0, size = tal)
c = np.random.uniform(0.0, 3.0, size = tal)
d = np.vstack((korndiameter,b,c)).T

import matplotlib.pyplot as plt

bins = 1.20**(np.arange(-4,18))
print( "bins: ", bins)

x = np.histogram(korndiameter,np.concatenate(([0],bins)))
x1 = np.cumsum(x[0])

# x = np.linspace(-5, 2, 100)
# y1 = x**3 + 5*x**2 + 10
# y2 = 3*x**2 + 10*x
# y3 = 6*x + 10
myDPI = 300
fig, ax = plt.subplots(figsize=(1190/myDPI,800/myDPI),dpi=myDPI)
ax.semilogx(bins, x1, color="blue", label="kornfordeling")
# ax.set_xticks(bins)
# ax.set_xticklabels(bins)
# ax.plot(x, y2, color="red", label="y'(x)")
# ax.plot(x, y3, color="green", label="y”(x)")
# ax.set_xlabel("x")
# ax.set_ylabel("y")
# ax.legend()

# logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))

# n, bins, patches = ax.hist(korndiameter, cumulative=True, logx=True, bins=bins)