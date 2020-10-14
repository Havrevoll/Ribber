# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from matplotlib import animation
# import matplotlib as mpl

import numpy as np
from scipy import interpolate
from scipy.integrate import solve_ivp  # https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html#r179348322575-1
import h5py
# import os.path.join as pjoin

fil = h5py.File('D:/Q40.mat', 'r') # https://docs.h5py.org/en/stable/quick.html#quick
 # list(f.keys())
 # ['#refs#', 'LEUC', 'LSUC', 'UEUC', 'USUC', 'Umx', 'Vmx', 'x', 'y']
 # x.shape
 
x = fil['x'][0]
y = fil['y'][0]

Umx = np.array(fil['Umx'])*1000
Vmx = np.array(fil['Vmx'])*1000

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
Umx_reshape = Umx.reshape((len(Umx),J,I))
Vmx_reshape = Vmx.reshape((len(Vmx),J,I))
t_3d,y_3d,x_3d = np.meshgrid(np.arange(3600.0),y_reshape1[:,0],x_reshape1[0,:],indexing='ij')

# Re_str_reshape1 = Re_stressm.reshape((J,I))   #   Re_str_reshape=(Re_str_reshape1(t+1:J-b,m+1:I-n));
up_sq_bar_reshape1 = up_sq_bar.reshape((J,I))  #   up_sq_bar_reshape=(up_sq_bar_reshape1(t+1:J-b,m+1:I-n));
vp_sq_bar_reshape1 = vp_sq_bar.reshape((J,I)) #   vp_sq_bar_reshape=(vp_sq_bar_reshape1(t+1:J-b,m+1:I-n));

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

v_mag = np.sqrt(u_reshape1 * u_reshape1 + v_reshape1 * v_reshape1)

q= axes[1].pcolor(x_reshape1,y_reshape1, v_mag)
axes[1].set_xlabel(r'$x$ [mm]', fontsize=18)
axes[1].set_ylabel(r'$y$ [mm]', fontsize=18)
cb2 = fig.colorbar(q, ax=axes[1])
cb2.set_label(r"$\overline{v}$ [mm/s]", fontsize=18)
axes[1].axis('equal')

# https://matplotlib.org/gallery/images_contours_and_fields/plot_streamplot.html#sphx-glr-gallery-images-contours-and-fields-plot-streamplot-py
q= axes[1].streamplot(x_reshape1,y_reshape1,u_reshape1, v_reshape1, arrowsize=1, linewidth=(.01*v_mag)) # tok vekk color=v_mag
#plt.show()
fig.savefig('straumfelt.png')


#%%
''' Byrja på å berekna stien til ein partikkel '''

# https://stackoverflow.com/questions/59071446/why-does-scipy-griddata-return-nans-with-cubic-interpolation-if-input-values


# f1 = interpolate.griddata(np.transpose(np.vstack((x, y))), u_bar, np.array([0,0]),method='linear')

#Her tek me vekk alle nan frå x, y og uv.
nonanindex=np.invert(np.isnan(x)) * np.invert(np.isnan(y)) * np.invert(np.isnan(u_bar)) * np.invert(np.isnan(v_bar))

nonancoords= np.transpose(np.vstack((x[nonanindex], y[nonanindex])))
nonanu = u_bar[nonanindex]
nonanv = v_bar[nonanindex]


from math import floor, ceil

def f(t,yn, method='nearest',t_av=True): # yn er array-like, altså np.array(xn,yn)
    if not t_av:
        #return np.hstack([interpolate.griddata((x,y), Umx[round(t),:], yn, method=method), interpolate.griddata((x,y), Vmx[round(t),:], yn, method=method)])
        interpolate.interp1d(np.array([floor(t),ceil(t)]),np.hstack([interpolate.griddata((x,y), Umx[floor(t),:], yn, method='linear'), 
                                                        interpolate.griddata((x,y), Umx[ceil(t),:], yn, method='linear')]))(t)
        #t_low = np.hstack([interpolate.griddata((x,y), Umx[floor(t),:], yn, method=method), interpolate.griddata((x,y), Vmx[floor(t),:], yn, method=method)])
        #t_hi = np.hstack([interpolate.griddata((x,y), Umx[ceil(t),:], yn, method=method), interpolate.griddata((x,y), Vmx[ceil(t),:], yn, method=method)])
    else:
        return np.hstack([interpolate.griddata((x,y), u_bar, yn, method=method), interpolate.griddata((x,y), v_bar, yn, method=method)]) 

    # return np.hstack([interpolate.griddata(nonancoords, nonanu, yn ,method=method),
            # interpolate.griddata(nonancoords, nonanv, yn ,method=method)]) 
    
def f_t(t,yn):
    return f(t,yn,t_av=False)

#g = f(0,[0,0])

#%%


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

p_x,p_y = np.meshgrid([-91],[85,75,65,55,45,35,25,15,5,0,-20,-30,-40,-50,-60])

sti = []

for par in np.column_stack((p_x,p_y)):
    sti_t, sti_y = rk(0, par, 1)
    sti.append([sti_t,sti_y])
    
sti = np.array(sti)

#%%
#solve_ivp(fun, t_span, y0, method='RK45', t_eval=None, dense_output=False, events=None, vectorized=False, args=None, **options)



#%%

# animer ein partikkel https://stackoverflow.com/questions/49119896/animating-particles-path-with-disappearing-tail-in-python



punkt = np.vstack(sti[:,1,0])


fig, ax = plt.subplots()

ax.pcolor(x_reshape1,y_reshape1, v_mag)
part, =ax.plot(punkt[:,0], punkt[:,1], 'ro')

def nypkt(i):
    part.set_data(np.vstack(sti[:,1,i])[:,0], np.vstack(sti[:,1,i])[:,1])
    return part,

ax.axis('equal')
ani = animation.FuncAnimation(fig, nypkt, np.arange(1,100),interval=100)
plt.show()
ani.save("sti.mp4")
    
    
# im = ax.pcolor(x_reshape1,y_reshape1, v_mag)

# if i == 0:
#     ax.pcolor(x_reshape1,y_reshape1, v_mag)
# # axes[1].set_xlabel(r'$x$ [mm]', fontsize=18)
# # axes[1].set_ylabel(r'$y$ [mm]', fontsize=18)
# # cb2 = fig.colorbar(q, ax=axes[1])
# # cb2.set_label(r"$\overline{v}$ [mm/s]", fontsize=18)
# ax.axis('equal')



   
# ax.plot(punktx,punkty,'o')

# ims.append([im])
#fig.savefig('straumfelt{0}.png'.format(i))
   
#ani= animation.ArtistAnimation(fig, ims, interval=50, blit=False, repeat_delay=1000)
#ani.save("sti.mp4")

#%%

p_x,p_y = np.meshgrid([-91],[85,75,65,55,45,35,25,15,5,0,-20,-30,-40,-50,-60])

sti = []

for par in np.column_stack((p_x,p_y)):
    sti.append(solve_ivp(f, , y0, options))
    
sti = np.array(sti)

punkt = np.vstack(sti[:,1,0])


fig, ax = plt.subplots()

ax.pcolor(x_reshape1,y_reshape1, Umx_reshape[0,:,:])
part, =ax.plot(punkt[:,0], punkt[:,1], 'ro')

def nypkt(i):
    part.set_data(np.vstack(sti[:,1,i])[:,0], np.vstack(sti[:,1,i])[:,1])
    return part,

ax.axis('equal')
ani = animation.FuncAnimation(fig, nypkt, np.arange(1,100),interval=100)
plt.show()
ani.save("sti.mp4")