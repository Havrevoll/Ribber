# -*- coding: utf-8 -*-
'''køyr funksjonar som plottingar(fil['vassføringar'])'''

import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib as mpl

import numpy as np
from scipy import interpolate
from scipy.integrate import solve_ivp  # https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html#r179348322575-1
from scipy.optimize import fsolve
import h5py
import re

from math import ceil, floor, log
# import os.path.join as pjoin



fil = h5py.File("D:/Tonstad/alle.hdf5", 'a')

discharges = [20,40,60,80,100,120,140]

class MidpointNormalize(mpl.colors.Normalize):
    '''https://stackoverflow.com/questions/7404116/defining-the-midpoint-of-a-colormap-in-matplotlib 
    http://chris35wills.github.io/matplotlib_diverging_colorbar/
    https://matplotlib.org/tutorials/colors/colormapnorms.html'''
    
    def __init__(self, vmin, vmax, midpoint=0, clip=False):
        self.midpoint = midpoint
        mpl.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        normalized_min = max(0, 1 / 2 * (1 - abs((self.midpoint - self.vmin) / (self.midpoint - self.vmax))))
        normalized_max = min(1, 1 / 2 * (1 + abs((self.vmax - self.midpoint) / (self.midpoint - self.vmin))))
        normalized_mid = 0.5
        x, y = [self.vmin, self.midpoint, self.vmax], [normalized_min, normalized_mid, normalized_max]
        return np.ma.masked_array(np.interp(value, x, y))


def vegglov(u_star, y, v):
    nu = 1 # 1 mm²/s
    y = y + 75
    return 1/0.4 * log(u_star * y / nu) + 5.5 - v/u_star


def finn_u(y,v):
    u = np.zeros(127)
    
    for i in np.arange(95,113):
        u[i]= fsolve(vegglov, 2, args=(y[i],v[i]))
 
    
    return u

def hentdata(flow_case):
    
    filnamn =  "D:/Q{}.mat".format(flow_case)
    
    fil = h5py.File(filnamn, 'r') # https://docs.h5py.org/en/stable/quick.html#quick
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
    
    Re_stressp = -1*up*vp
    Re_stressm = np.nanmean(Re_stressp ,0)
    
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
    
    Re_str_reshape1 = Re_stressm.reshape((J,I))   #   Re_str_reshape=(Re_str_reshape1(t+1:J-b,m+1:I-n));
    up_sq_bar_reshape1 = up_sq_bar.reshape((J,I))  #   up_sq_bar_reshape=(up_sq_bar_reshape1(t+1:J-b,m+1:I-n));
    vp_sq_bar_reshape1 = vp_sq_bar.reshape((J,I)) #   vp_sq_bar_reshape=(vp_sq_bar_reshape1(t+1:J-b,m+1:I-n));

    u_profile = np.nanmean(u_reshape1,1)


    vort = np.zeros((3600,J,I))

    d_l = 186/I

    for t in np.arange(3600):
        if t % 100 == 0: 
            print(t, end = '')
            print(' ', end = '')
        for j in np.arange(1,J-1):
            for i in np.arange(1,I-1):
                vort[t,j,i] = (Umx_reshape[t,j-1,i]-Umx_reshape[t,j+1,i]) / 2 + (Vmx_reshape[t,j,i+1]-Vmx_reshape[t,j,i-1]) / 2
                
    
    vort = vort/d_l
    
    vort_bar = np.nanmean(vort,0)
    
    
    # https://stackoverflow.com/questions/59071446/why-does-scipy-griddata-return-nans-with-cubic-interpolation-if-input-values

    #Her tek me vekk alle nan frå x, y og uv.
    nonanindex=np.invert(np.isnan(x)) * np.invert(np.isnan(y)) * np.invert(np.isnan(u_bar)) * np.invert(np.isnan(v_bar))
    nonancoords= np.transpose(np.vstack((x[nonanindex], y[nonanindex])))
    nonanu = u_bar[nonanindex]
    nonanv = v_bar[nonanindex]
    
    nonanxindex = np.invert(np.isnan(Umx))
    nonanyindex = np.invert(np.isnan(Vmx))
    

    loc = locals()
    return dict([(i,loc[i]) for i in loc])

def fyllopp(discharges):
    cases = {}
    for q in discharges:
        print("byrja på Q", q)
        cases[q]=hentdata(q)
    return cases



#%%

def straumfelt(case):
    
    x_reshape1 = np.array(case['x_reshape1'])
    y_reshape1 = np.array(case['y_reshape1'])
    u_reshape1 = np.array(case['u_reshape1'])
    v_reshape1 = np.array(case['v_reshape1'])
    u_profile = np.array(case['u_profile'])
    
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
    fig.canvas.draw()
   
    
    axes[1].plot(u_profile,y_reshape1[:,0])
    axes[1].set_xlabel(r'$x$ [mm/s]', fontsize=18)
    axes[1].set_ylabel(r'$y$ [mm]', fontsize=18)
    axes[1].set_ylim(axes[0].get_ylim())
    #axes[1].set_xlim(0,500)
    # https://matplotlib.org/gallery/images_contours_and_fields/plot_streamplot.html#sphx-glr-gallery-images-contours-and-fields-plot-streamplot-py
    
    filnamn = "straumfeltQ{}.png".format(re.split(r'/',case.name)[-1])
    
    fig.savefig(filnamn)
    plt.close()
    
    
def straumfelt_normalisert(case):
    
    x_reshape1 = np.array(case['x_reshape1'])
    y_reshape1 = np.array(case['y_reshape1'])
    u_reshape1 = np.array(case['u_reshape1'])
    v_reshape1 = np.array(case['v_reshape1'])
    u_profile = np.array(case['u_profile'])
    
    fig, axes = plt.subplots(1,2, figsize=(18,8))
    # ax.plot(x, y1, color="blue", label="x")
    # ax.plot(x, y2, color="red", label="y'(x)")
    # ax.plot(x, y3, color="green", label="y”(x)")
    
    
    p= axes[0].pcolor(x_reshape1,y_reshape1, u_reshape1,vmin=0, vmax=500)
    axes[0].set_xlabel(r'$x$ [mm]', fontsize=18)
    axes[0].set_ylabel(r'$y$ [mm]', fontsize=18)
    
    cb = fig.colorbar(p, ax=axes[0])
    cb.set_label(r"$\overline{u}$ [mm/s]", fontsize=18)
       
    k = 5
    # https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/quiver_demo.html
    axes[0].quiver(x_reshape1[::k, ::k], y_reshape1[::k, ::k], u_reshape1[::k, ::k], v_reshape1[::k, ::k])
    # Kva med dette: Straumlinefelt: https://stackoverflow.com/questions/39619128/plotting-direction-field-in-python
     

    axes[0].axis('equal')
    fig.canvas.draw()
   
    
    axes[1].plot(u_profile,y_reshape1[:,0])
    axes[1].set_xlabel(r'$x$ [mm/s]', fontsize=18)
    axes[1].set_ylabel(r'$y$ [mm]', fontsize=18)
    axes[1].set_ylim(axes[0].get_ylim())
    axes[1].set_xlim(0,500)
    # https://matplotlib.org/gallery/images_contours_and_fields/plot_streamplot.html#sphx-glr-gallery-images-contours-and-fields-plot-streamplot-py
    
    filnamn = "straumfelt_normQ{}.png".format(re.split(r'/',case.name)[-1])
    
    fig.savefig(filnamn)
    plt.close()

#%% 

# Burde laga eit plott av Reynolds skjerspenning
# 

def reynolds_plot(case):
    x_reshape1 = np.array(case['x_reshape1'])
    y_reshape1 = np.array(case['y_reshape1'])
    Re_str_reshape1 = np.array(case['Re_str_reshape1'])

    vmin =  np.nanmin(Re_str_reshape1)
    vmax = np.nanmax(Re_str_reshape1)
    
    norm = MidpointNormalize(vmin=vmin, vmax=vmax, midpoint=0)
    # cmap = 'RdBu_r' 

    # plt.imshow(vals, cmap=cmap, norm=norm)
    # plt.colorbar()
    
    fig,axes = plt.subplots()
    p = axes.imshow(Re_str_reshape1, extent=[x_reshape1[0,0],x_reshape1[0,-1], y_reshape1[-1,0], y_reshape1[0,0]], cmap='RdGy', norm=norm)
    
    fig.colorbar(p, ax=axes)
    
    filnamn = "reynolds_stress_Q{}.png".format(re.split(r'/',case.name)[-1])
    
    fig.savefig(filnamn)
    plt.close()
    

#%%

def straumfelt_og_piler(case):
    ''' Straumfeltet og piler '''

    x_reshape1= np.array(case['x_reshape1'])
    y_reshape1 = np.array(case['y_reshape1'])
    u_reshape1 = np.array(case['u_reshape1'])
    v_reshape1 = np.array(case['v_reshape1'])
    v_bar_mag = np.array(case['v_bar_mag'])
    
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
    
    filnamn = "straumfelt_hires_Q{}.png".format(re.split(r'/',case.name)[-1])
    
    fig.savefig(filnamn)
    plt.close()

#%%
def vortisiteten(case):
    ''' Vortisiteten i heile feltet '''
    
    x_reshape1= np.array(case['x_reshape1'])
    y_reshape1 = np.array(case['y_reshape1'])
    vort_bar = np.array(case['vort_bar'])
    
    myDPI = 300
    fig, axes = plt.subplots(figsize=(2050/myDPI,1450/myDPI),dpi=myDPI)
    
    
    p= axes.pcolor(x_reshape1,y_reshape1, vort_bar)
    axes.set_xlabel(r'$x$ [mm]', fontsize=18)
    axes.set_ylabel(r'$y$ [mm]', fontsize=18)
    cb = fig.colorbar(p, ax=axes)
    cb.set_label(r"Vorticity", fontsize=18)
    
    axes.axis('equal')
    axes.axis([-91, 91, -75, 90])
    
    filnamn = "vorticityQ{}.png".format(re.split(r'/',case.name)[-1])
    
    fig.savefig(filnamn)
    plt.close()

#%%

def kvervel_naerbilete(case):
    '''Nærbilete av kvervelen ved ribba '''
    
    x_reshape1= np.array(case['x_reshape1'])
    y_reshape1 = np.array(case['y_reshape1'])
    u_reshape1 = np.array(case['u_reshape1'])
    v_reshape1 = np.array(case['v_reshape1'])
    v_bar_mag = np.array(case['v_bar_mag'])
    
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
    
    filnamn = "vortexQ{}.png".format(re.split(r'/',case.name)[-1])
    fig.savefig(filnamn)
    plt.close()

#%%

def film_fartogpiler(case):
    ''' Lagar ein film av området nedstraums for ribba, med absoluttverdien av farten i bakgrunnen, og piler for farten oppå.'''
    
    x_reshape1= np.array(case['x_reshape1'])
    y_reshape1 = np.array(case['y_reshape1'])
    V_mag_reshape = np.array(case['V_mag_reshape'])
    Umx_reshape = np.array(case['Umx_reshape'])
    Vmx_reshape = np.array(case['Vmx_reshape'])
    
    myDPI = 200
    fig, ax = plt.subplots(figsize=(1000/myDPI,1000/myDPI),dpi=myDPI)
    
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
    
    filnamn = "kvervelQ{}.mp4".format(re.split(r'/',case.name)[-1])
    ani.save(filnamn)
    plt.close()

#%%

def film_vortisitetogpiler(case):
    ''' Lagar ein film av området nedstraums for ribba, med vortisiteten i bakgrunnen, og piler for farten oppå.'''
    
    x_reshape1= np.array(case['x_reshape1'])
    y_reshape1 = np.array(case['y_reshape1'])
    vort = np.array(case['vort'])
    Umx_reshape = np.array(case['Umx_reshape'])
    Vmx_reshape = np.array(case['Vmx_reshape'])
    
    myDPI = 200
    fig, ax = plt.subplots(figsize=(1000/myDPI,1000/myDPI),dpi=myDPI)
    
    field = ax.imshow(vort[0,55:80,45:67], extent=[x_reshape1[0,45],x_reshape1[0,67], y_reshape1[80,0], y_reshape1[55,0]])
    pil = ax.quiver(x_reshape1[55:80,45:67], y_reshape1[55:80,45:67], Umx_reshape[0,55:80,45:67], Vmx_reshape[0,55:80,45:67], scale=1000)
    
    def nypkt(i):
        field.set_data(vort[i,55:80,45:67])
        pil.set_UVC(Umx_reshape[i,55:80,45:67], Vmx_reshape[i,55:80,45:67])
        return field,pil
    
    print("Skal byrja på filmen")
    #ax.axis('equal')
    ani = animation.FuncAnimation(fig, nypkt, frames=np.arange(1,600),interval=50)
    plt.show()
    print("ferdig med animasjon, skal lagra")
    
    filnamn = "kvervel2Q{}.mp4".format(re.split(r'/',case.name)[-1])
    ani.save(filnamn)
    plt.close()


#%%

def tredimensjonalt_felt(case):
    '''lag eit tredimensjonalt diagram av fartsfeltet'''
    
    x_reshape1= np.array(case['x_reshape1'])
    y_reshape1 = np.array(case['y_reshape1'])
    v_bar_mag = np.array(case['v_bar_mag'])
    
    myDPI = 200
    
    fig = plt.figure(figsize=(2050/myDPI,1050/myDPI),dpi=myDPI)
    ax = fig.gca(projection='3d')
    
    # Plot the surface.
    surf = ax.plot_surface(x_reshape1, y_reshape1, v_bar_mag)
    
    # Add a color bar which maps values to colors.
    fig.colorbar(surf)
    
    # plt.show()


#%%

# def f(t,yn, method='nearest'): # yn er array-like, altså np.array(xn,yn)
#     return np.hstack([interpolate.griddata((x,y), u_bar, yn, method=method), interpolate.griddata((x,y), v_bar, yn, method=method)]) 

def interp_lin_near(coords,values, yn):
    new = interpolate.griddata(coords, values, yn, method='linear')
    if np.isnan(new):
        return interpolate.griddata(coords, values, yn, method='nearest')
    else:
        return new

def lag_ft(case):
    ''' Funksjon for å laga eit kontinuerleg vektorfelt '''
    nonanxindex = np.array(case['nonanxindex'])
    nonanyindex = np.array(case['nonanyindex'])
    Umx = np.array(case['Umx'])
    Vmx = np.array(case['Vmx'])
    x = np.array(case['x'])
    y = np.array(case['y'])
        
    def f_t(t, yn):
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
    return f_t
    

#%%
''' Heimelaga Runge-Kutta-metode '''

# def rk(t0, y0, L, h=0.02):
#     N=int(L/h)

#     t=[0]*N # initialize lists
#     y=[0]*N # initialize lists
    
#     t[0] = t0
#     y[0] = y0
    
#     for n in range(0, N-1):
#         #print(n,t[n], y[n], f(t[n],y[n]))
#         k1 = h*f(t[n], y[n])
#         k2 = h*f(t[n] + 0.5 * h, y[n] + 0.5 * k1)
#         k3 = h*f(t[n] + 0.5 * h, y[n] + 0.5 * k2)
#         k4 = h*f(t[n] + h, y[n] + k3)
        
#         if (np.isnan(k4+k3+k2+k1).any()):
#             #print(k1,k2,k3,k4)
#             return t,y
        
#         t[n+1] = t[n] + h
#         y[n+1] = y[n] + 1/6 * (k1 + 2*k2 + 2*k3 + k4)
        
#     return t,y

#%%
def lag_sti(case):
    f_t = lag_ft(case)
    
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
    
    return np.array(sti_ny)
    



#%%
def sti_animasjon(case):
    
    x_reshape1= np.array(case['x_reshape1'])
    y_reshape1 = np.array(case['y_reshape1'])
    V_mag_reshape = np.array(case['V_mag_reshape'])
    sti_ny = np.array(case['sti_ny'])
    
    fig, ax = plt.subplots()
    
    field = ax.imshow(V_mag_reshape[0,:,:], extent=[x_reshape1[0,0],x_reshape1[0,-1], y_reshape1[-1,0], y_reshape1[0,0]])
    particle, =ax.plot(sti_ny[:,0,0], sti_ny[:,0,1], 'ro')
    ax.set_xlim([x_reshape1[0,0],x_reshape1[0,-1]])
    
    def nypkt(i):
        field.set_data(V_mag_reshape[i,:,:])
        particle.set_data(sti_ny[:,i,0], sti_ny[:,i,1])
        return field,particle
    
    print("Skal byrja på filmen")
    #ax.axis('equal')
    ani = animation.FuncAnimation(fig, nypkt, frames=np.arange(1,600),interval=50)
    plt.show()
    print("ferdig med animasjon, skal lagra")
    
    filnamn = "stiQ{}.mp4".format(re.split(r'/',case.name)[-1])
    ani.save(filnamn)
    plt.close()


#%%
def lagra(dataset):
    f = h5py.File('alle2.hdf5','w')
    
    vassf = f.create_group("vassføringar")
    
    for q in dataset:
        gr = vassf.create_group(str(q))
        for k in dataset[q]:
            gr.create_dataset(k, data=dataset[q][k], compression="gzip", compression_opts=9)
    f.close()

#%%
# lag straumfelt-bilete
def plottingar(cases):
    '''kall med plottingar(fil['vassføringar'])'''
    for q in discharges:
        # straumfelt_normalisert(cases[str(q)])
        # straumfelt(cases[str(q)])
        reynolds_plot(cases[str(q)])
        # straumfelt_og_piler(cases[str(q)])
        # vortisiteten(cases[str(q)])
        # kvervel_naerbilete(cases[str(q)])
        # film_fartogpiler(cases[str(q)])
        # film_vortisitetogpiler(cases[str(q)])
        
# re.split(r'/',fil['vassføringar']['20'].name)[-1]


vass = fil['vassføringar']
v_mean = {}
for q in vass:
    v_mean[q] = np.mean(vass[q]['u_profile'][67:114])