# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import matplotlib as mpl
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

u_bar = np.nanmean(Umx,0)
v_bar = np.nanmean(Vmx,0)

up = Umx - u_bar
vp = Vmx - v_bar

up_sq = (up*up)
vp_sq = (vp*vp)

up_sq_bar = np.nanmean(up_sq,0)
vp_sq_bar = np.nanmean(vp_sq,0)

Re_stressp = -1*up*vp
Re_stressm = np.nanmean(Re_stressp ,0)

I = 126  # horisontal lengd
J = 127  # vertikal lengd
m = 3 #define number of columns to be cut at the lefthand side of the window
n = 2 #define number of columns to be cut at the righthand side of the window
b = 3 #define number of columns to be cut at the bottom of the window
t = 3 #define number of columns to be cut at the top of the window

x_reshape1 = x.reshape((J,I))      # x_reshape=(x_reshape1(t+1:J-b,m+1:I-n))
y_reshape1 = y.reshape((J,I))      # y_reshape=(y_reshape1(t+1:J-b,m+1:I-n));
u_reshape1 = u_bar.reshape((J,I))  # u_reshape=(u_reshape1(t+1:J-b,m+1:I-n));
v_reshape1 = v_bar.reshape((J,I))  # v_reshape=(v_reshape1(t+1:J-b,m+1:I-n));

Re_str_reshape1 = Re_stressm.reshape((J,I))   #   Re_str_reshape=(Re_str_reshape1(t+1:J-b,m+1:I-n));
up_sq_bar_reshape1 = up_sq_bar.reshape((J,I))  #   up_sq_bar_reshape=(up_sq_bar_reshape1(t+1:J-b,m+1:I-n));
vp_sq_bar_reshape1 = vp_sq_bar.reshape((J,I)) #   vp_sq_bar_reshape=(vp_sq_bar_reshape1(t+1:J-b,m+1:I-n));

#%%

fig, axes = plt.subplots(1,2, figsize=(23,10))
# ax.plot(x, y1, color="blue", label="x")
# ax.plot(x, y2, color="red", label="y'(x)")
# ax.plot(x, y3, color="green", label="y”(x)")


p= axes[0].pcolor(x_reshape1,y_reshape1, u_reshape1)
axes[0].set_xlabel(r'$x$', fontsize=18)
axes[0].set_ylabel(r'$y$', fontsize=18)
cb = fig.colorbar(p, ax=axes[0])
cb.set_label(r"$\overline{u}$", fontsize=18)

k = 5
# https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/quiver_demo.html
axes[0].quiver(x_reshape1[::k, ::k], y_reshape1[::k, ::k], u_reshape1[::k, ::k], v_reshape1[::k, ::k] ) # kunne lagt inn angles='xy', prøv det
 

axes[0].axis('equal')

q= axes[1].pcolor(x_reshape1,y_reshape1, v_reshape1)
axes[1].set_xlabel(r'$x$', fontsize=18)
axes[1].set_ylabel(r'$y$', fontsize=18)
cb2 = fig.colorbar(q, ax=axes[1])
cb2.set_label(r"$\overline{v}$", fontsize=18)
axes[1].axis('equal')
#plt.show()


#%%

#Ta vekk alle nan frå u_bar og v_bar

nonanindex=np.invert(np.isnan(x))*np.invert(np.isnan(y))*np.invert(np.isnan(u_bar))

#%%
''' Byrja på å berekna stien til ein partikkel '''

# https://stackoverflow.com/questions/59071446/why-does-scipy-griddata-return-nans-with-cubic-interpolation-if-input-values


f1 = interpolate.griddata(np.transpose(np.vstack((x, y))), u_bar, np.array([0,0]),method='linear')

#Her tek me vekk alle nan frå x, y og uv.
nonanindex=np.invert(np.isnan(x)) * np.invert(np.isnan(y)) * np.invert(np.isnan(u_bar)) * np.invert(np.isnan(v_bar))
def f(xn,yn):
    return interpolate.griddata(np.transpose(np.vstack((x[nonanindex], y[nonanindex]))), u_bar[nonanindex], np.array([xn,yn]),method='cubic')

g = f(0,0)
