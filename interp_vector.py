# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from matplotlib import animation
# import matplotlib as mpl

import numpy as np
from scipy import interpolate
from scipy.integrate import solve_ivp  # https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html#r179348322575-1
import h5py

from math import floor, ceil
# import os.path.join as pjoin

fil = h5py.File('D:/Q40.mat', 'r') # https://docs.h5py.org/en/stable/quick.html#quick
 # list(f.keys())
 # ['#refs#', 'LEUC', 'LSUC', 'UEUC', 'USUC', 'Umx', 'Vmx', 'x', 'y']
 # x.shape
 
x = fil['x'][0]
y = fil['y'][0]

Umx = np.array(fil['Umx'])*1000
Vmx = np.array(fil['Vmx'])*1000
V_mag = np.sqrt(Umx * Umx + Vmx * Vmx)

u_bar = np.nanmean(Umx,0)
v_bar = np.nanmean(Vmx,0)


up = Umx - u_bar
vp = Vmx - v_bar

up_sq = (up*up)
vp_sq = (vp*vp)

up_sq_bar = np.nanmean(up_sq,0)
vp_sq_bar = np.nanmean(vp_sq,0)

# Re_stressp = -1*up*vp
# Re_stressm = np.nanmean(Re_stressp ,0)

I = 126  # horisontal lengd
J = 127  # vertikal lengd
# m = 3 #define number of columns to be cut at the lefthand side of the window
# n = 2 #define number of columns to be cut at the righthand side of the window
# b = 3 #define number of columns to be cut at the bottom of the window
# t = 3 #define number of columns to be cut at the top of the window

x_reshape1 = x.reshape((J,I))      # x_reshape=(x_reshape1(t+1:J-b,m+1:I-n))
y_reshape1 = y.reshape((J,I))      # y_reshape=(y_reshape1(t+1:J-b,m+1:I-n));
u_reshape1 = u_bar.reshape((J,I))  # u_reshape=(u_reshape1(t+1:J-b,m+1:I-n));
v_reshape1 = v_bar.reshape((J,I))  # v_reshape=(v_reshape1(t+1:J-b,m+1:I-n));
v_bar_mag = np.sqrt(u_reshape1 * u_reshape1 + v_reshape1 * v_reshape1)

Umx_reshape = Umx.reshape((len(Umx),J,I))
Vmx_reshape = Vmx.reshape((len(Vmx),J,I))
V_mag_reshape = V_mag.reshape((len(V_mag),J,I))
t_3d,y_3d,x_3d = np.meshgrid(np.arange(3600.0),y_reshape1[:,0],x_reshape1[0,:],indexing='ij')

# Re_str_reshape1 = Re_stressm.reshape((J,I))   #   Re_str_reshape=(Re_str_reshape1(t+1:J-b,m+1:I-n));
up_sq_bar_reshape1 = up_sq_bar.reshape((J,I))  #   up_sq_bar_reshape=(up_sq_bar_reshape1(t+1:J-b,m+1:I-n));
vp_sq_bar_reshape1 = vp_sq_bar.reshape((J,I)) #   vp_sq_bar_reshape=(vp_sq_bar_reshape1(t+1:J-b,m+1:I-n));

#%%

vort = np.zeros((3600,J,I))

d_l = 186/I

for t in np.arange(3600):
    print(t, end = '')
    print(' ', end = '')
    for j in np.arange(1,J-1):
        for i in np.arange(1,I-1):
            vort[t,j,i] = (Umx_reshape[t,j-1,i]-Umx_reshape[t,j+1,i]) / 2 + (Vmx_reshape[t,j,i+1]-Vmx_reshape[t,j,i-1]) / 2
            

print("ferdig")
vort = vort/d_l

vort_bar = np.nanmean(vort,0)

#%%

fig, axes = plt.subplots(1,2, figsize=(18,8))
# ax.plot(x, y1, color="blue", label="x")
# ax.plot(x, y2, color="red", label="y'(x)")
# ax.plot(x, y3, color="green", label="y”(x)")


p= axes[0].pcolor(x_reshape1,y_reshape1, u_reshape1)
axes[0].set_xlabel(r'$x$ [mm]', fontsize=18)
axes[0].set_ylabel(r'$y$ [mm]', fontsize=18)
cb = fig.colorbar(p, ax=axes[0])
cb.set_label(r"$\overline{u}$ [mm/s]", fontsize=18)


k = 5
# https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/quiver_demo.html
axes[0].quiver(x_reshape1[::k, ::k], y_reshape1[::k, ::k], u_reshape1[::k, ::k], v_reshape1[::k, ::k])
# Kva med dette: Straumlinefelt: https://stackoverflow.com/questions/39619128/plotting-direction-field-in-python
 

axes[0].axis('equal')

q= axes[1].pcolor(x_reshape1,y_reshape1, v_bar_mag)
axes[1].set_xlabel(r'$x$ [mm]', fontsize=18)
axes[1].set_ylabel(r'$y$ [mm]', fontsize=18)
cb2 = fig.colorbar(q, ax=axes[1])
cb2.set_label(r"$\overline{v}$ [mm/s]", fontsize=18)
axes[1].axis('equal')

# https://matplotlib.org/gallery/images_contours_and_fields/plot_streamplot.html#sphx-glr-gallery-images-contours-and-fields-plot-streamplot-py
q= axes[1].streamplot(x_reshape1,y_reshape1,u_reshape1, v_reshape1, arrowsize=1, linewidth=(.01*v_bar_mag)) # tok vekk color=v_bar_mag
#plt.show()
fig.savefig('straumfelt.png')

#%%
''' Straumfeltet og piler '''

myDPI = 300
fig, axes = plt.subplots(figsize=(2050/myDPI,1450/myDPI),dpi=myDPI)


p= axes.pcolor(x_reshape1,y_reshape1, v_bar_mag)
axes.set_xlabel(r'$x$ [mm]', fontsize=18)
axes.set_ylabel(r'$y$ [mm]', fontsize=18)
cb = fig.colorbar(p, ax=axes)
cb.set_label(r"$\overline{u}$ [mm/s]", fontsize=18)

k = 3
axes.quiver(x_reshape1[::k, ::k], y_reshape1[::k, ::k], u_reshape1[::k, ::k], v_reshape1[::k, ::k])
axes.set_xlim()

axes.axis('equal')
axes.axis([-91, 91, -75, 90])

fig.savefig('straumfelt_hires.png')

#%%
''' Vortisiteten i heile feltet '''

myDPI = 300
fig, axes = plt.subplots(figsize=(2050/myDPI,1450/myDPI),dpi=myDPI)


p= axes.pcolor(x_reshape1,y_reshape1, vort_bar)
axes.set_xlabel(r'$x$ [mm]', fontsize=18)
axes.set_ylabel(r'$y$ [mm]', fontsize=18)
cb = fig.colorbar(p, ax=axes)
cb.set_label(r"Vorticity", fontsize=18)

axes.axis('equal')
axes.axis([-91, 91, -75, 90])

fig.savefig('vorticity.png')

#%%
'''Nærbilete av kvervelen ved ribba '''
myDPI = 300
fig, axes = plt.subplots(figsize=(2050/myDPI,1050/myDPI),dpi=myDPI)

p= axes.pcolor(x_reshape1,y_reshape1, v_bar_mag)
axes.set_xlabel(r'$x$ [mm]', fontsize=18)
axes.set_ylabel(r'$y$ [mm]', fontsize=18)
cb = fig.colorbar(p, ax=axes)
cb.set_label(r"$\overline{u}$ [mm/s]", fontsize=18)

k = 1
axes.quiver(x_reshape1[::k, ::k], y_reshape1[::k, ::k], u_reshape1[::k, ::k], v_reshape1[::k, ::k],scale=100)
axes.set_xlim()

axes.axis('equal')
axes.axis([-25, 15, -20, 5])

fig.savefig('vortex.png')

#%%

fig, ax = plt.subplots()

field = ax.imshow(V_mag_reshape[0,55:80,45:67], extent=[x_reshape1[0,45],x_reshape1[0,67], y_reshape1[80,0], y_reshape1[55,0]])
pil = ax.quiver(x_reshape1[55:80,45:67], y_reshape1[55:80,45:67], Umx_reshape[0,55:80,45:67], Vmx_reshape[0,55:80,45:67], scale=1000)

def nypkt(i):
    field.set_data(V_mag_reshape[i,55:80,45:67])
    pil.set_UVC(Umx_reshape[i,55:80,45:67], Vmx_reshape[i,55:80,45:67])
    return field,pil

print("Skal byrja på filmen")
#ax.axis('equal')
ani = animation.FuncAnimation(fig, nypkt, frames=np.arange(1,600),interval=50)
plt.show()
print("ferdig med animasjon, skal lagra")
ani.save("kvervel.mp4")



#%%

fig = plt.figure(figsize=(2050/myDPI,1050/myDPI),dpi=myDPI)
ax = fig.gca(projection='3d')

# Plot the surface.
surf = ax.plot_surface(x_reshape1, y_reshape1, v_bar_mag)

# Customize the z axis.
# ax.set_zlim(-1.01, 1.01)
# ax.zaxis.set_major_locator(LinearLocator(10))
# ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf)

# plt.show()


#%%
''' Funksjon for å laga eit kontinuerleg vektorfelt '''

# https://stackoverflow.com/questions/59071446/why-does-scipy-griddata-return-nans-with-cubic-interpolation-if-input-values

#Her tek me vekk alle nan frå x, y og uv.
nonanindex=np.invert(np.isnan(x)) * np.invert(np.isnan(y)) * np.invert(np.isnan(u_bar)) * np.invert(np.isnan(v_bar))
nonancoords= np.transpose(np.vstack((x[nonanindex], y[nonanindex])))
nonanu = u_bar[nonanindex]
nonanv = v_bar[nonanindex]

nonanxindex = np.invert(np.isnan(Umx))
nonanyindex = np.invert(np.isnan(Vmx))



def f(t,yn, method='nearest'): # yn er array-like, altså np.array(xn,yn)
    return np.hstack([interpolate.griddata((x,y), u_bar, yn, method=method), interpolate.griddata((x,y), v_bar, yn, method=method)]) 

def interp_lin_near(coords,values, yn):
    new = interpolate.griddata(coords, values, yn, method='linear')
    if np.isnan(new):
        return interpolate.griddata(coords, values, yn, method='nearest')
    else:
        return new

def f_t(t,yn):
    t_0 = floor(t)
    t_1 = ceil(t)
    
    if t_0 == t_1:
        u_0 = interp_lin_near((x[nonanxindex[t_0]], y[nonanxindex[t_0]]), Umx[t_0,nonanxindex[t_0,:]], yn) #interpolate.griddata((x[nonanxindex[t_0]], y[nonanxindex[t_0]]), Umx[t_0,nonanxindex[t_0,:]], yn)
        v_0 = interp_lin_near((x[nonanyindex[t_0]], y[nonanyindex[t_0]]), Vmx[t_0,nonanyindex[t_0,:]], yn)
        
        return np.hstack([u_0,v_0])
    
    u_0 = interp_lin_near((x[nonanxindex[t_0]], y[nonanxindex[t_0]]), Umx[t_0,nonanxindex[t_0,:]], yn)
    v_0 = interp_lin_near((x[nonanyindex[t_0]], y[nonanyindex[t_0]]), Vmx[t_0,nonanyindex[t_0,:]], yn)
    
    u_1 = interp_lin_near((x[nonanxindex[t_1]], y[nonanxindex[t_1]]), Umx[t_1,nonanxindex[t_1,:]], yn)
    v_1 = interp_lin_near((x[nonanyindex[t_1]], y[nonanyindex[t_1]]), Vmx[t_1,nonanyindex[t_1,:]], yn)
    
    u_x = u_0 + (t- t_0) * (u_1 - u_0) / (t_1 - t_0) 
    v_y = v_0 + (t- t_0) * (v_1 - v_0) / (t_1 - t_0) 
    
    print("ferdig med interpolering")
    print(t,yn,np.hstack([u_x,v_y]))
    return np.hstack([u_x,v_y])
    
#g = f(0,[0,0])

#%%
f_t(0.5,[-90,85])

#%%
''' Heimelaga Runge-Kutta-metode '''

def rk(t0, y0, L, h=0.02):
    N=int(L/h)

    t=[0]*N # initialize lists
    y=[0]*N # initialize lists
    
    t[0] = t0
    y[0] = y0
    
    for n in range(0, N-1):
        #print(n,t[n], y[n], f(t[n],y[n]))
        k1 = h*f(t[n], y[n])
        k2 = h*f(t[n] + 0.5 * h, y[n] + 0.5 * k1)
        k3 = h*f(t[n] + 0.5 * h, y[n] + 0.5 * k2)
        k4 = h*f(t[n] + h, y[n] + k3)
        
        if (np.isnan(k4+k3+k2+k1).any()):
            #print(k1,k2,k3,k4)
            return t,y
        
        t[n+1] = t[n] + h
        y[n+1] = y[n] + 1/6 * (k1 + 2*k2 + 2*k3 + k4)
        
    return t,y

#%%

p_x,p_y = np.meshgrid([-90,-200],[85,75,65,55,45,35,25,15,5,0,-20,-30,-40,-50,-60])

p_x = p_x.T.reshape(-1)
p_y= p_y.T.reshape(-1)

sti = []

t_start= 0
t_end = 10
fps= 60

for par in np.column_stack((p_x,p_y)):
    sti.append(solve_ivp(f_t, [t_start,t_end], par, t_eval=np.arange(t_start, t_end, 1/fps)))
    
sti_ny=[]

for el in sti:
    sti_ny.append(el.y.T)

sti_ny = np.array(sti_ny)


#%%

fig, ax = plt.subplots()

field = ax.imshow(Umx_reshape[0,:,:], extent=[x_reshape1[0,0],x_reshape1[0,-1], y_reshape1[-1,0], y_reshape1[0,0]])
particle, =ax.plot(sti_ny[:,0,0], sti_ny[:,0,1], 'ro')
ax.set_xlim([x_reshape1[0,0],x_reshape1[0,-1]])

def nypkt(i):
    field.set_data(Umx_reshape[i,:,:])
    particle.set_data(sti_ny[:,i,0], sti_ny[:,i,1])
    return field,particle

print("Skal byrja på filmen")
#ax.axis('equal')
ani = animation.FuncAnimation(fig, nypkt, frames=np.arange(1,600),interval=50)
plt.show()
print("ferdig med animasjon, skal lagra")
ani.save("sti2.mp4")

