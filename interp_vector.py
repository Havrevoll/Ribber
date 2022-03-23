# -*- coding: utf-8 -*-
'''køyr funksjonar som plottingar(fil['vassføringar'])'''

import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "STIXGeneral"
plt.rcParams['mathtext.fontset'] = 'stix'
from matplotlib import animation, colors
import matplotlib as mpl
from matplotlib.patches import Rectangle

import numpy as np
# from scipy import interpolate
# from scipy.integrate import solve_ivp  # https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html#r179348322575-1
from scipy.optimize import fsolve
import h5py
import re
import scipy.stats as stats

from hjelpefunksjonar import ranges, draw_rect

# from IPython.display import clear_output


from math import ceil, floor, hypot, log, sqrt
# import os.path.join as pjoin

# fil = h5py.File("D:/Tonstad/alle.hdf5", 'a')
# vass = fil['vassføringar']


discharges = [20,40,60,80,100,120,140]
data = "../Q40.hdf5"

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

def get_reshape(dataset):
    """Ein metode som tek inn eit datasett og gjer alle reshapings-tinga for x og y, u og v pluss ribbelista.

    Args:
        dataset (string): filnamnet på hdf5-fila.

    Returns:
        tuple: Ein tuple med x, y, U, V (2d-arrays) og ribs (original-lista)
    """
 
    x,y,Umx,Vmx,I,J,ribs = get_xyuvijribs(dataset)
    
    x_reshape = x.reshape((J,I))
    y_reshape = y.reshape((J,I))
    
    Umx_reshape = Umx.reshape((len(Umx),J,I))
    Vmx_reshape = Vmx.reshape((len(Vmx),J,I))
    
    return x_reshape, y_reshape, Umx_reshape, Vmx_reshape, ribs

def get_xyuvijribs(dataset):
    """Ein metode som hentar ut alle dataa frå ei hdf5-fil og returnerer dei som ndarrays.

    Args:
        dataset (string): filnamnet på hdf5-fila

    Returns:
        tuple: x,y,U,V (alle som 1D-array), I,J (ints) og ribbelista
    """
    with h5py.File(dataset, 'r') as f:
        Umx = np.array(f['Umx'])
        Vmx = np.array(f['Vmx'])
        (I,J) = (int(np.array(f['I'])),int(np.array(f['J'])))
        x = np.array(f['x'])
        y = np.array(f['y'])
        ribs = np.array(f['ribs'])
    
    return x,y,Umx,Vmx,I,J,ribs
        
def get_mean(dataset):
    _,_,Umx,Vmx,I,J,_ = get_xyuvijribs(dataset)
    
    u_bar = np.nanmean(Umx,0).reshape((J,I))
    v_bar = np.nanmean(Vmx,0).reshape((J,I))
    
    
    return u_bar, v_bar
    

# def lag_mindredatasett(case):
#     '''
#     Ein metode som tek inn eit case frå den store alle.hdf5-fila og tek ut berre dei viktige delane og lagrar i ei ny fil med maks kompresjon. Må laga ein tilsvarande metode for å henta fram data og laga alle dei bearbeida versjonane, på eit per case-basis.
#     '''
#     utfilnamn = "D:/Tonstad/utvalde/Q{}.hdf5".format(case)
#     print(utfilnamn)
        
    
#     with h5py.File(utfilnamn, 'a') as utfil:
#         for sett in ['nonanindex', 'x', 'y', 'nonanu', 'nonanv', 'nonancoords', 'vort', 'Umx', 'Vmx']:
#             utfil.create_dataset(sett, data=vass[case][sett], compression="gzip", compression_opts=9)
            
#         for sett in ['I', 'J', 'd_l', 'filnamn', 'flow_case', 'i', 'j']:
#             utfil.create_dataset(sett, data=vass[case][sett])
#     print("hei")

def vegglov(u_star, y, v):
    h= -5.9
    # nu = 1 # 1 mm²/s
    y = y - h
    ks = .0025
    return 1/0.4 * log(30 * y / ks) - v/u_star

def finn_u(y,v):
    u = np.zeros(127)
    
    for i in np.arange(0,67):
        u[i]= fsolve(vegglov, 2, args=(y[i],v[i]))
     
    return u

# def draw_rect(axes,color='red',new_setup=True):
    
#     if (new_setup):
#         axes.add_patch(Rectangle((-62.4,-9.56),50,8,linewidth=2,edgecolor=color,facecolor='none'))
#         axes.add_patch(Rectangle((37.6,-8.5),50,8,linewidth=2,edgecolor=color,facecolor='none'))

#     else:
#         axes.add_patch(Rectangle((-62.4,-9.56),50,8,linewidth=2,edgecolor=color,facecolor='none'))
#         axes.add_patch(Rectangle((37.6,-8.5),50,8,linewidth=2,edgecolor=color,facecolor='none'))

def draw_shade(axes, x0=0, x=430, color='red'):
    axes.add_patch(Rectangle((x0,-9.8),x,10.8,linewidth=2,edgecolor='none',facecolor='lightcoral'))

# def ranges():
#     y_range = np.s_[0:114]
#     x_range = np.s_[40:108]
    
#     piv_range = np.index_exp[y_range,x_range]
    
#     return piv_range

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
    
    Re_stressp = -1*up*vp*1e-3
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

# def calc_Re_stress(case):
#     up = case['up']
#     vp = case['vp']
    
#     Re_stressp=-1*np.array(up)*np.array(vp)*1e-3
    
#     Re_stressm = np.nanmean(Re_stressp,0)
    
#     Re_str_reshape1 = Re_stressm.reshape((127,126))
    
#     data = case['Re_str_reshape1']       # load the data
#     data[...] = Re_str_reshape1          # assign new values to data
#     rep = case['Re_stressp']
#     rep[...] = Re_stressp
#     rem = case['Re_stressm']
#     rem[...] = Re_stressm
#     fil.flush()
    
    
def calc_u_profile(case):

    
    piv_range = ranges()
    y_range = piv_range[0]
    
    # x_reshape1 = np.array(case['x_reshape1'])
    # x=x_reshape1[piv_range]
    y_reshape1 = np.array(case['y_reshape1'])
    y=y_reshape1[piv_range]


    u_reshape1 = np.array(case['u_reshape1'][piv_range])
    # v_reshape1 = np.array(case['v_reshape1'][piv_range])
    
    u_profile = np.nanmean(u_reshape1,1)
    
    gml_u_profile = np.array(case['u_profile'][y_range])
    
    myDPI = 300
    fig, axes = plt.subplots(figsize=(900/myDPI,900/myDPI),dpi=myDPI)
    
    axes.plot(u_profile, y[:,0], linewidth=.8, label="ny")
    axes.plot(gml_u_profile, y[:,0], linewidth=.8, label="gml")
    
    draw_shade(axes)
    # axes.set_title("Reynolds' turbulent shear stress")
    axes.set_xlabel(r'$u$ [mm/s]')
    axes.set_ylabel(r'$y$ [mm]')
    axes.legend()
    # axes[1].set_xlim(0,500)
    plt.tight_layout()
    
    filnamn = "u_nyoggamal.png"
    
    fig.savefig(filnamn)
    plt.close()
    
    
def fyllopp(discharges):
    cases = {}
    for q in discharges:
        print("byrja på Q", q)
        cases[q]=hentdata(q)
    return cases

def straumfelt(case):
    
    x,y,_,_ = get_reshape(case)
    u,v = get_mean(case)
    
    u_profile = np.nanmean(u,1)
    
    fig, axes = plt.subplots(1,2, figsize=(18,8))
    # ax.plot(x, y1, color="blue", label="x")
    # ax.plot(x, y2, color="red", label="y'(x)")
    # ax.plot(x, y3, color="green", label="y”(x)")
    
    
    p= axes[0].pcolor(x,y, u )
    axes[0].set_xlabel(r'$x$ [mm]', fontsize=18)
    axes[0].set_ylabel(r'$y$ [mm]', fontsize=18)
    
    cb = fig.colorbar(p, ax=axes[0])
    cb.set_label(r"$\overline{u}$ [mm/s]", fontsize=18)
       
    k = 5
    # https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/quiver_demo.html
    axes[0].quiver(x[::k, ::k], y[::k, ::k], u[::k, ::k], v[::k, ::k])
    # Kva med dette: Straumlinefelt: https://stackoverflow.com/questions/39619128/plotting-direction-field-in-python
     

    axes[0].axis('equal')
    fig.canvas.draw()
   
    
    axes[1].plot(u_profile,y[:,0])
    axes[1].set_xlabel(r'$x$ [mm/s]', fontsize=18)
    axes[1].set_ylabel(r'$y$ [mm]', fontsize=18)
    axes[1].set_ylim(axes[0].get_ylim())
    #axes[1].set_xlim(0,500)
    # https://matplotlib.org/gallery/images_contours_and_fields/plot_streamplot.html#sphx-glr-gallery-images-contours-and-fields-plot-streamplot-py
    
    draw_rect(axes[0])
    # draw_rect(axes[1])
    
    filnamn = "straumfeltQ20.png"
    
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
    
    
    p= axes[0].pcolor(x_reshape1,y_reshape1, u_reshape1,vmin=0, vmax=500 )
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
    

    draw_rect(axes[0])
    draw_shade(axes[1],x=500)
    
    filnamn = "straumfelt_normQ{}.png".format(re.split(r'/',case.name)[-1])
    
    fig.savefig(filnamn)
    plt.close()

def reynolds_plot(case):

    x_reshape1 = np.array(case['x_reshape1'])
    y_reshape1 = np.array(case['y_reshape1'])
    
    try:
        Re_str_reshape1 = np.array(case['Re_str_reshape1'])
    except RuntimeError:
        up = case['up']
        vp = case['vp']
        
        rep=-1*np.array(up)*np.array(vp)*1e-3
        
        rem = np.nanmean(rep,0)
        
        Re_str_reshape1 = rem.reshape((127,126))
    
    z = stats.zscore(Re_str_reshape1.flatten(),nan_policy='omit').reshape(Re_str_reshape1.shape)
    
    outliers = z > 10
    Re_str_reshape1[outliers] = np.nan
        
    vmin = np.nanmin(Re_str_reshape1)
    vmax = np.nanmax(Re_str_reshape1)
    
    norm = MidpointNormalize(vmin=vmin, vmax=vmax, midpoint=0)
    
    myDPI = 200
    fig, axes = plt.subplots(figsize=(1100/myDPI,900/myDPI),dpi=myDPI)
    
    p = axes.imshow(Re_str_reshape1, extent=[x_reshape1[0,0],x_reshape1[0,-1], y_reshape1[-1,0], y_reshape1[0,0]], cmap='RdGy', norm=norm)
    draw_rect(axes)
    
    cb = fig.colorbar(p, ax=axes)
    cb.set_label(r"$\tau_t$ [Pa]")
    
    axes.set_xlabel(r'$x$ [mm]')
    axes.set_ylabel(r'$y$ [mm]')
    
    # Re_profile = np.nanmean(Re_str_reshape1,1)
    # axes[1].plot(Re_profile,y_reshape1[:,0])
    # axes[1].set_xlabel(r'$Re$ [mm²/s²]')
    # axes[1].set_ylabel(r'$y$ [mm]')
    # axes[1].set_ylim(axes[0].get_ylim())
    # axes[1].set_xlim(0,500)
    plt.tight_layout()
    
    filnamn = "reynolds_stress_Q{}.png".format(re.split(r'/',case.name)[-1])
    
    fig.savefig(filnamn)
    plt.close()
    
def re_plot_all(cases):
    
    piv_range = ranges()
    
    # x_reshape1 = np.array(cases['40']['x_reshape1'])
    y_reshape1 = np.array(cases['40']['y_reshape1'])[piv_range]
    
    Re_profiles = {}
    
    vmin = np.inf
    vmax = np.NINF
    
    for q in cases:
        
        try:
            Re_str_reshape1 = np.array(cases[q]['Re_str_reshape1'])[piv_range]
        except RuntimeError:
            up = cases[q]['up']
            vp = cases[q]['vp']
            
            rep=-1*np.array(up)*np.array(vp)*1e-3
            
            rem = np.nanmean(rep,0)
            
            Re_str_reshape1 = rem.reshape((127,126))[piv_range]
        
        z = stats.zscore(Re_str_reshape1.flatten(),nan_policy='omit').reshape(Re_str_reshape1.shape)
        
        outliers = z > 10
        Re_str_reshape1[outliers] = np.nan
            
        vmin = np.min([np.nanmin(Re_str_reshape1), vmin])
        vmax = np.max([np.nanmax(Re_str_reshape1), vmax])
         
        
        Re_profiles[q] = np.nanmean(Re_str_reshape1,1)
    

    myDPI = 200
    fig, axes = plt.subplots(figsize=(900/myDPI,900/myDPI),dpi=myDPI)
    
    for case in Re_profiles:
        axes.plot(Re_profiles[case],y_reshape1[:,0], linewidth=.8, label="{} l/s".format(case))
        
    draw_shade(axes,x0=-.15,x=1.15)
    # axes.set_title("Reynolds' turbulent shear stress")
    axes.set_xlabel(r'$\langle\tau_t\rangle$ [Pa]')
    axes.set_ylabel(r'$y$ [mm]')
    axes.legend(frameon=False, loc='lower right', ncol=2, fontsize=9)
    # axes[1].set_xlim(0,500)
    plt.tight_layout()
    
    filnamn = "reynolds_profiles.png"
    
    fig.savefig(filnamn)
    plt.close()
    
    
def u_plot_all(cases):
    piv_range = ranges()
    y_reshape1 = np.array(cases['40']['y_reshape1'])[piv_range]
    
    u_profiles = {}
    
    for q in cases:
        u_profiles[q] = np.array(cases[q]['u_profile'])[piv_range[0]]
    
    myDPI = 200
    fig, axes = plt.subplots(figsize=(900/myDPI,900/myDPI),dpi=myDPI)
    
    for case in u_profiles:
        axes.plot(u_profiles[case],y_reshape1[:,0], linewidth=.8, label="{} l/s".format(case))
    
    draw_shade(axes)
    axes.set_xlabel(r'$\langle \bar{u}\rangle$ [mm/s]')
    axes.set_ylabel(r'$y$ [mm]')
    axes.legend(frameon=False, loc='lower right', ncol=2, fontsize=9)
    # axes[1].set_xlim(0,500)
    plt.tight_layout()
    
    filnamn = "u_profiles.png"
    
    fig.savefig(filnamn)
    plt.close()
    
def dbl_average(case):
    piv_range = ranges()
    # y_range = piv_range[0]
    
    x_reshape1 = np.array(case['x_reshape1'])
    x=x_reshape1[piv_range]
    y_reshape1 = np.array(case['y_reshape1'])
    y=y_reshape1[piv_range]


    u_profile = np.array(case['u_profile'][piv_range[0]])
    u_reshape1 = np.array(case['u_reshape1'][piv_range])
    v_reshape1 = np.array(case['v_reshape1'][piv_range])
    
    # up_sq_bar_reshape1 = np.array(case['up_sq_bar_reshape1'][piv_range])
    # vp_sq_bar_reshape1 = np.array(case['vp_sq_bar_reshape1'][piv_range])
        
    v_profile = np.nanmean(v_reshape1,1)
    
    u_tilde = (u_reshape1.T - u_profile).T
    
    v_tilde = (v_reshape1.T - v_profile).T
    
    

    # u_tilde_sq=u_tilde*u_tilde
    # v_tilde_sq=v_tilde*v_tilde
    
    # u_tilde_DA = np.nanmean(u_tilde,1)
    # v_tilde_DA = np.nanmean(v_tilde,1)

    # u_tilde_sq_DA = np.nanmean(u_tilde_sq,1)
    # v_tilde_sq_DA = np.nanmean(v_tilde_sq,1)
    
    form_stress = u_tilde*v_tilde*1e-3
    form_stress_DA = np.nanmean(form_stress,1)
    
    z = stats.zscore(u_tilde.flatten(),nan_policy='omit').reshape(u_tilde.shape)
    
    outliers = z > 10
    u_tilde[outliers] = np.nan
        
    vmin =  np.nanmin(form_stress)
    vmax = np.nanmax(form_stress)
    norm = MidpointNormalize(vmin=vmin, vmax=vmax, midpoint=0)
    myDPI = 300
    
    fig, axes = plt.subplots(1,2, figsize=(1800/myDPI,1000/myDPI),dpi=myDPI)
    
    p = axes[0].imshow(form_stress, extent=[x[0,0],x[0,-1], y[-1,0], y[0,0]], cmap='RdGy', norm=norm)
    draw_rect(axes[0])
    axes[0].set_xlabel(r'$x$ [mm]')
    axes[0].set_ylabel(r'$y$ [mm]')

    cb = fig.colorbar(p, ax=axes[0])
    cb.set_label(r"$-\rho\tilde{u}\tilde{w}$ [Pa]")    
    
    axes[1].plot(form_stress_DA,y[:,0])
    axes[1].set_xlabel(r'$-\rho\langle\tilde{u}\tilde{w}\rangle$ [Pa]')
    axes[1].set_ylabel(r'$y$ [mm]')
    axes[1].set_ylim(axes[0].get_ylim())
    plt.tight_layout()
    
    filnamn = "spatial_averaged_vel_Q{}.png".format(re.split(r'/',case.name)[-1])
    
    fig.savefig(filnamn)
    plt.close()
    
    
    
    fig, ax = plt.subplots(figsize=(2000/myDPI,830/myDPI),dpi=myDPI)
    
    ax.imshow(form_stress[50:80,:], extent=[x[0,0],x[0,-1], y[80,0], y[50,0]],cmap='RdGy', norm=norm)
    draw_rect(ax)
    
    cb = fig.colorbar(p, ax=ax)
    cb.set_label(r"$-\rho\langle\tilde{u}\tilde{w}\rangle$ [Pa]")
    
    ax.set_xlabel(r'$x$ [mm]')
    ax.set_ylabel(r'$y$ [mm]')
    plt.tight_layout()
    
    filnamn = "spatial_av_vel_near_Q{}.png".format(re.split(r'/',case.name)[-1])
    
    fig.savefig(filnamn)
    plt.close()
        
    
    fig, ax  = plt.subplots(figsize=(1000/myDPI,1000/myDPI),dpi=myDPI)
    
    ax.plot(u_tilde[61,:],v_tilde[61,:],"bo")
    ax.plot(u_tilde[58,:],v_tilde[58,:],"ro")
    ax.plot(u_tilde[55,:],v_tilde[55,:],"go")
    ax.plot(u_tilde[52,:],v_tilde[52,:],"co")
    # btw_rib = np.s_[15:48]
    # ax.plot(u_tilde[64,btw_rib],v_tilde[64,btw_rib],"bx")
    # ax.plot(u_tilde[66,btw_rib],v_tilde[66,btw_rib],"gx")
    ax.set_xlabel(r'$\tilde{u}$ [mm/s]')
    ax.set_ylabel(r'$\tilde{v}$ [mm/s]')
    ax.axis('equal')
    ax.axhline(y=0, color='k')
    ax.axvline(x=0, color='k')
    plt.tight_layout()
    
    filnamn = "spatial_quadrant_Q{}.png".format(re.split(r'/',case.name)[-1])
    fig.savefig(filnamn)
    plt.close()
    
    # k_bar= 0.5 * (up_sq_bar_reshape1 + vp_sq_bar_reshape1)
    
    # k_profile = 0.5 * np.nanmean((up_sq_bar_reshape1 + vp_sq_bar_reshape1),1)
    
    qua1 = np.logical_and(u_tilde > 0, v_tilde > 0)
    qua2 = np.logical_and(u_tilde < 0, v_tilde > 0)
    qua3 = np.logical_and(u_tilde < 0, v_tilde < 0)
    qua4 = np.logical_and(u_tilde > 0, v_tilde < 0)
    
    qua1 = np.ma.masked_where(qua1 == False, qua1)
    qua2 = np.ma.masked_where(qua2 == False, qua2)
    qua3 = np.ma.masked_where(qua3 == False, qua3)
    qua4 = np.ma.masked_where(qua4 == False, qua4)
    
    cmap1 = colors.ListedColormap(['green'])
    cmap2 = colors.ListedColormap(['red'])
    cmap3 = colors.ListedColormap(['yellow'])
    cmap4 = colors.ListedColormap(['blue'])
    
    # z = qua1[:-1, :-1]
    # levels = mpl.ticker.MaxNLocator(nbins=1).tick_values( z.max())
    # norm = colors.BoundaryNorm(levels, ncolors=cmap1.N, clip=True)
    
    
    fig, ax  = plt.subplots(figsize=(1000/myDPI,1000/myDPI),dpi=myDPI)
    
    ax.pcolormesh(x, y, qua1, shading='nearest', cmap=cmap1)
    ax.pcolormesh(x, y, qua2, shading='nearest', cmap=cmap2)
    ax.pcolormesh(x, y, qua3, shading='nearest', cmap=cmap3)
    ax.pcolormesh(x, y, qua4, shading='nearest', cmap=cmap4)
    
    ax.plot(np.array([-50]), np.array([7.25]),"ro")
    ax.plot(np.array([-50]), np.array([2.85]),"bo")
    ax.plot(np.array([-50]), np.array([11.66]),"go")
    ax.plot(np.array([-50]), np.array([16.07]),"co")
    
    draw_rect(ax,'black')
    
    ax.set_xlabel(r'$x$ [mm]')
    ax.set_ylabel(r'$y$ [mm]')

    plt.tight_layout()

    
    filnamn = "spatial_quadrant_map_Q{}.png".format(re.split(r'/',case.name)[-1])
    fig.savefig(filnamn)
    plt.close()
    
    print(x[0,0],x[0,-1],y[0,0],y[-1,0],x.shape)
    
def dbl_av_all(cases):
    u_tildes = {}
    v_tildes = {}
    

    piv_range = ranges()
    y_range = piv_range[0]

    for q in cases:
        case = cases[q]
        # x_reshape1 = np.array(case['x_reshape1'])
        # x=x_reshape1[piv_range]
        # y_reshape1 = np.array(case['y_reshape1'])
        # y=y_reshape1[piv_range]
    
        u_profile = np.array(case['u_profile'][y_range])
        u_reshape1 = np.array(case['u_reshape1'][piv_range])
        v_reshape1 = np.array(case['v_reshape1'][piv_range])
        v_profile = np.nanmean(v_reshape1,1)
        
        u_tilde = (u_reshape1.T - u_profile).T
        
        v_tilde = (v_reshape1.T - v_profile).T
        
        u_tildes[case] = u_tilde
        v_tildes[case] = v_tilde
    
    myDPI = 300
    fig, ax  = plt.subplots(figsize=(1000/myDPI,1000/myDPI),dpi=myDPI)
    
    for q in cases:
        case = cases[q]
        u_tilde = u_tildes[case]
        v_tilde = v_tildes[case]
        
        ax.plot(u_tilde[61,:],v_tilde[61,:],'-', label="{} l/s".format(q))
             
    ax.set_xlabel(r'$\tilde{u}$ [mm/s]')
    ax.set_ylabel(r'$\tilde{v}$ [mm/s]')
    ax.legend(bbox_to_anchor=(1.05, 1.0), loc='lower right', ncol=3, fontsize=9)
    ax.axhline(y=0, color='k')
    ax.axvline(x=0, color='k')

    ax.axis('equal')
    plt.tight_layout()
    
    filnamn = "quadrant_diagrams.png"
    
    fig.savefig(filnamn)
    plt.close()
    
def kontur(case):
    # piv_range = ranges()
    # y_range = piv_range[0]
    
    x_reshape1 = np.array(case['x_reshape1'])
    # x=x_reshape1[piv_range]
    y_reshape1 = np.array(case['y_reshape1'])
    # y=y_reshape1[piv_range]


    # u_profile = np.array(case['u_profile'][piv_range[0]])
    u_reshape1 = np.array(case['u_reshape1'])
    # v_reshape1 = np.array(case['v_reshape1'])
    # Re_str_reshape1 = np.array(case['Re_str_reshape1'])
    
  
    
    
    myDPI = 200
    fig, ax  = plt.subplots(figsize=(1000/myDPI,1000/myDPI),dpi=myDPI)
    
    ax.contour(x_reshape1, y_reshape1, u_reshape1, colors='black')
    draw_rect(ax)
    
    
def t_quadrant(case):
    '''Lag plott av kvadrantanalysen'''
    x_reshape1= np.array(case['x_reshape1'])
    y_reshape1 = np.array(case['y_reshape1'])
    up = np.array(case['up'])
    vp = np.array(case['vp'])
    
    # qua1 = np.logical_and(up > 0, vp > 0)
    # qua2 = np.logical_and(up < 0, vp > 0)
    # qua3 = np.logical_and(up < 0, vp < 0)
    # qua4 = np.logical_and(up > 0, vp < 0)
    
    # qua1num = qua1 * 1
    # qua2num = qua2 * 2
    # qua3num = qua3 * 3
    # qua4num = qua4 * 4
    
    # quanum = qua1num + qua2num + qua3num + qua4num
    
    # quanum_reshape = quanum.reshape((3600,127,126))
    
    myDPI = 200
    
    
    # fargar = ['green', 'red', 'yellow', 'blue', 'purple']
    # bounds = [0,1,2,3,4]

    # fargekart = colors.Colormap(colors)
    
    
    # fig, ax = plt.subplots(figsize=(1000/myDPI,1000/myDPI),dpi=myDPI)
    
    
    

    # norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    
    # field = ax.imshow(quanum_reshape[0,:,:], extent=[x_reshape1[0,0],x_reshape1[0,-1], y_reshape1[-1,0], y_reshape1[0,0]], cmap=fargar)
    
    # draw_rect(ax)
    
    # def nypkt(i):
    #     field.set_data(quanum_reshape[i,:,:])
        
    #     return field,
    
    # print("Skal byrja på filmen")
    # #ax.axis('equal')
    # ani = animation.FuncAnimation(fig, nypkt, frames=np.arange(1,300),interval=50)
    # plt.show()
    # print("ferdig med animasjon, skal lagra")
    
    # filnamn = "quadrant{}.mp4".format(re.split(r'/',case.name)[-1])
    # ani.save(filnamn)
    # plt.close()
    
    up_mean = np.nanmean(up,0)
    vp_mean = np.nanmean(vp,0)
    qua1_mean = np.logical_and(up_mean < 0, vp_mean > 0)
    qua2_mean = np.logical_and(up_mean > 0, vp_mean > 0)
    qua3_mean = np.logical_and(up_mean < 0, vp_mean < 0)
    qua4_mean = np.logical_and(up_mean > 0, vp_mean < 0)
    
    qua1num_mean = qua1_mean * 1
    qua2num_mean = qua2_mean * 2
    qua3num_mean = qua3_mean * 3
    qua4num_mean = qua4_mean * 4
    
    quanum_mean = qua1num_mean + qua2num_mean + qua3num_mean + qua4num_mean
    quanum_mean_reshape = quanum_mean.reshape((127,126))
    
    fig, axes = plt.subplots(figsize=(2000/myDPI,1400/myDPI),dpi=myDPI)
    
    p = axes.imshow(quanum_mean_reshape, extent=[x_reshape1[0,0],x_reshape1[0,-1], y_reshape1[-1,0], y_reshape1[0,0]], interpolation='none', cmap='viridis')
    
    axes.set_xlabel(r'$x$ [mm]', fontsize=18)
    axes.set_ylabel(r'$y$ [mm]', fontsize=18)
    cb = fig.colorbar(p, ax=axes)
    cb.set_label(r"$\overline{u'}$ [mm/s]", fontsize=18)
    
    draw_rect(axes)
    
    filnamn = "quadrant{}.png".format(re.split(r'/',case.name)[-1])
    
    fig.savefig(filnamn)
    plt.close()
    
def straumfelt_og_piler(case):
    ''' Straumfeltet og piler '''

    x_reshape1= np.array(case['x_reshape1'])
    y_reshape1 = np.array(case['y_reshape1'])
    u_reshape1 = np.array(case['u_reshape1'])
    v_reshape1 = np.array(case['v_reshape1'])
    v_bar_mag = np.array(case['v_bar_mag'])
    
    myDPI = 300
    fig, axes = plt.subplots(figsize=(2050/myDPI,1450/myDPI),dpi=myDPI)
    
    
    p= axes.pcolor(x_reshape1,y_reshape1, v_bar_mag )
    axes.set_xlabel(r'$x$ [mm]', fontsize=18)
    axes.set_ylabel(r'$y$ [mm]', fontsize=18)
    cb = fig.colorbar(p, ax=axes)
    cb.set_label(r"$\overline{u}$ [mm/s]", fontsize=18)
    
    k = 3
    axes.quiver(x_reshape1[::k, ::k], y_reshape1[::k, ::k], u_reshape1[::k, ::k], v_reshape1[::k, ::k])
    axes.set_xlim()
    
    axes.axis('equal')
    axes.axis([-91, 91, -75, 90])
    
    draw_rect(axes)
    
    filnamn = "straumfelt_hires_Q{}.png".format(re.split(r'/',case.name)[-1])
    
    fig.savefig(filnamn)
    plt.close()

def vortisiteten(case):
    ''' Vortisiteten i heile feltet '''
    
    x_reshape1= np.array(case['x_reshape1'])
    y_reshape1 = np.array(case['y_reshape1'])
    vort_bar = np.array(case['vort_bar'])
    
    myDPI = 300
    fig, axes = plt.subplots(figsize=(2050/myDPI,1450/myDPI),dpi=myDPI)
    
    
    p= axes.pcolor(x_reshape1,y_reshape1, vort_bar )
    axes.set_xlabel(r'$x$ [mm]')
    axes.set_ylabel(r'$y$ [mm]')
    cb = fig.colorbar(p, ax=axes)
    cb.set_label(r"Vorticity")
    
    axes.axis('equal')
    axes.axis([-91, 91, -75, 90])

    draw_rect(axes)    
    filnamn = "vorticityQ{}.png".format(re.split(r'/',case.name)[-1])
    
    fig.savefig(filnamn)
    plt.close()


def kvervel_naerbilete(case):
    '''Nærbilete av kvervelen ved ribba '''
    
    x_reshape1= np.array(case['x_reshape1'])
    y_reshape1 = np.array(case['y_reshape1'])
    u_reshape1 = np.array(case['u_reshape1'])
    v_reshape1 = np.array(case['v_reshape1'])
    v_bar_mag = np.array(case['v_bar_mag'])
    
    myDPI = 300
    fig, axes = plt.subplots(figsize=(2050/myDPI,1050/myDPI),dpi=myDPI)
    
    p= axes.pcolor(x_reshape1,y_reshape1, v_bar_mag )
    axes.set_xlabel(r'$x$ [mm]')
    axes.set_ylabel(r'$y$ [mm]')
    cb = fig.colorbar(p, ax=axes)
    cb.set_label(r"$\overline{u}$ [mm/s]")
    
    k = 1
    axes.quiver(x_reshape1[::k, ::k], y_reshape1[::k, ::k], u_reshape1[::k, ::k], v_reshape1[::k, ::k],scale=100)
    axes.set_xlim()
    
    axes.axis('equal')
    axes.axis([-25, 15, -20, 5])
    draw_rect(axes)
    plt.tight_layout()
    
    filnamn = "vortexQ{}.png".format(re.split(r'/',case.name)[-1])
    fig.savefig(filnamn)
    plt.close()

def film_fartogpiler(datafil):
    ''' Lagar ein film av området nedstraums for ribba, med absoluttverdien av farten i bakgrunnen, og piler for farten oppå.'''
    
    # x_reshape1= np.array(case['x_reshape1'])
    # y_reshape1 = np.array(case['y_reshape1'])
    # V_mag_reshape = np.array(case['V_mag_reshape'])
    # Umx_reshape = np.array(case['Umx_reshape'])
    # Vmx_reshape = np.array(case['Vmx_reshape'])
    
    x,y,u,v, ribs = get_reshape(datafil)
    x1 = x[0,:]
    y1 = y[:,0]
    v_mag = np.hypot(u,v)

    myDPI = 200
    fig, ax = plt.subplots(figsize=(1000/myDPI,1000/myDPI),dpi=myDPI)
    
    range = np.index_exp[0:1,60:75,80:95]

    field = ax.imshow(v_mag[0,range[1],range[2]], extent=[x1[np.r_[range[2]][0]],x1[np.r_[range[2]][-1]], y1[np.r_[range[1]][-1]], y1[np.r_[range[1]][0]]])
    pil = ax.quiver(x[range[1:]], y[range[1:]], u[0,range[1],range[2]], v[0,range[1],range[2]], scale=1000)
    # draw_rect(ax, ribs )
        
    def nypkt(i):
        field.set_data(v_mag[i,range[1],range[2]])
        pil.set_UVC(u[i,range[1],range[2]], v[i,range[1],range[2]])
        return field,pil
    
    print("Skal byrja på filmen")
    #ax.axis('equal')
    ani = animation.FuncAnimation(fig, nypkt, frames=np.arange(1,600),interval=50)
    # plt.show()
    print("ferdig med animasjon, skal lagra")
    
    filnamn = "ribbe.mp4"
    ani.save(filnamn)
    # plt.close()




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
    draw_rect(ax)
    
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

def lagra(dataset):
    f = h5py.File('alle2.hdf5','w')
    
    vassf = f.create_group("vassføringar")
    
    for q in dataset:
        gr = vassf.create_group(str(q))
        for k in dataset[q]:
            gr.create_dataset(k, data=dataset[q][k], compression="gzip", compression_opts=9)
    f.close()


# lag straumfelt-bilete
def plottingar(cases):
    '''kall med plottingar(fil['vassføringar'])'''
    
    maxmin(cases)
    for q in cases:
        straumfelt_normalisert(cases[str(q)])
        straumfelt(cases[str(q)])
        reynolds_plot(cases[str(q)])
        straumfelt_og_piler(cases[str(q)])
        vortisiteten(cases[str(q)])
        kvervel_naerbilete(cases[str(q)])
        #film_fartogpiler(cases[str(q)])
        #film_vortisitetogpiler(cases[str(q)])
        


def maxmin(case):
    '''Skal finna maks og min for over, midt på og under ribbene'''
    maxmin = {}
    for q in case:
        maxmin[q] = dict(over=np.max(case[q]['u_profile'][0:62]), 
                         under= np.max(case[q]['u_profile'][66:113]),
                         midt=np.min(case[q]['u_profile'][60:100])
                         )
        
    over, under, midt = (np.zeros(7),np.zeros(7),np.zeros(7))
 
    for i,q in enumerate(discharges):  
        over[i] = maxmin[str(q)]['over']
        under[i] = maxmin[str(q)]['under']
        midt[i] = maxmin[str(q)]['midt']
        
    maxmin = dict(over=over, under=under, midt=midt)
   
    myDPI = 200
    fig, axes = plt.subplots(figsize=(900/myDPI,900/myDPI),dpi=myDPI)
    
    
    axes.plot(discharges,over,'r-', label=r'$u_{max,over}$')
    axes.plot(discharges, under,'b-', label=r'$u_{max,under}$')
    axes.plot(discharges,midt,'g-', label=r'$u_{min}$')
    # axes.set_yscale('log')
    # axes.set_title('Maximum and minimum velocities for discharges')
    axes.set_xlabel(r'$Q$ [l/s]')
    axes.set_ylabel(r'$u$ [mm/s]')
    # axes.grid(b=True, which='both')
    axes.legend(frameon=False)

    
    filnamn = "max_and_min_velocities.png"
    
    fig.savefig(filnamn)
    
    axes.set_yscale('log')
    fig.savefig("max_and_min_velocities_log.png")
    
    plt.close()
    
    #return maxmin
        


  
  
def runsTest(l, l_median): 
  
    runs, n1, n2 = 0, 0, 0
      
    # Checking for start of new run 
    for i in range(len(l)): 
          
        # no. of runs 
        if (l[i] >= l_median and l[i-1] < l_median) or (l[i] < l_median and l[i-1] >= l_median): 
            runs += 1  
          
        # no. of positive values 
        if(l[i]) >= l_median: 
            n1 += 1   
          
        # no. of negative values 
        else: 
            n2 += 1   
  
    runs_exp = ((2*n1*n2)/(n1+n2))+1
    stan_dev = sqrt((2*n1*n2*(2*n1*n2-n1-n2))/(((n1+n2)**2)*(n1+n2-1))) 
  
    z = (runs-runs_exp)/stan_dev 
  
    return z 



 
