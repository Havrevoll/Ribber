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

up_sq=(up*up)
vp_sq=(vp*vp)

up_sq_bar=np.nanmean(up_sq,0)
vp_sq_bar=np.nanmean(vp_sq,0)

u_bar=np.nanmean(Umx,0)
v_bar=np.nanmean(Vmx,0)

Re_stressp = -1*up*vp
Re_stressm=np.nanmean(Re_stressp ,0);


x_reshape1=(reshape(x,I,J))';       x_reshape=(x_reshape1(t+1:J-b,m+1:I-n));
y_reshape1=(reshape(y,I,J))';       y_reshape=(y_reshape1(t+1:J-b,m+1:I-n));
u_reshape1=(reshape(u_bar,I,J))';   u_reshape=(u_reshape1(t+1:J-b,m+1:I-n));
v_reshape1=(reshape(v_bar,I,J))';    v_reshape=(v_reshape1(t+1:J-b,m+1:I-n));

Re_str_reshape1=(reshape(Re_stress,I,J))';      Re_str_reshape=(Re_str_reshape1(t+1:J-b,m+1:I-n));
up_sq_bar_reshape1=(reshape(up_sq_bar,I,J))';   up_sq_bar_reshape=(up_sq_bar_reshape1(t+1:J-b,m+1:I-n));
vp_sq_bar_reshape1=(reshape(vp_sq_bar,I,J))';   vp_sq_bar_reshape=(vp_sq_bar_reshape1(t+1:J-b,m+1:I-n));


# %%
x = [0, 0, 1, 1, 2, 2, 0, 1, 2]
y = [1, 2, 1, 2, 1, 2, 1.5, 1.5, 1.5]
u = [0.5, -1, 0, 0, 0.25, 1, 0, 0, 0.75]
v = [1, 1, 1, 1, 1, 1, 1, 1, 1]

plt.figure(1)
plt.quiver(x, y, u, v)

xx = np.linspace(0, 2, 10)
yy = np.linspace(1, 2, 10)
xx, yy = np.meshgrid(xx, yy)

points = np.transpose(np.vstack((x, y)))
u_interp = interpolate.griddata(points, u, (xx, yy), method='cubic')  # https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.griddata.html
v_interp = interpolate.griddata(points, v, (xx, yy), method='cubic')

plt.figure(2)
plt.quiver(xx, yy, u_interp, v_interp)
plt.show()
