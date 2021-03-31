# -*- coding: utf-8 -*-
'''køyr funksjonar som plottingar(fil['vassføringar'])'''

import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "STIXGeneral"
plt.rcParams['mathtext.fontset'] = 'stix'
from matplotlib import animation, colors
import matplotlib as mpl
from matplotlib.patches import Rectangle

import numpy as np
from scipy import interpolate
from scipy.integrate import solve_ivp  # https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html#r179348322575-1
from scipy.optimize import fsolve
import h5py
import re
import scipy.stats as stats

import scipy.interpolate as spint
import scipy.spatial.qhull as qhull
import itertools
from scipy.spatial import cKDTree

# from IPython.display import clear_output


from math import ceil, floor, log, sqrt
# import os.path.join as pjoin

# fil = h5py.File("D:/Tonstad/alle.hdf5", 'a')
# vass = fil['vassføringar']


discharges = [20,40,60,80,100,120,140]

h= -5.9

def reshape(dataset = h5py.File("c:/Users/havrevol/Q40.hdf5", 'r')):
    '''
    Ein metode som tek inn eit datasett og gjer alle reshapings-tinga for x og y, u og v og Re.

    '''
    
    (I,J)=(int(np.array(dataset['I'])),int(np.array(dataset['J'])))
    t_max = 30
    steps = t_max * 20
    
    Umx = np.array(dataset['Umx'])[0:t_max,:]
    Umx_reshape = Umx.reshape((len(Umx),J,I))[:,1:114,1:125]
    # Vmx = np.array(dataset['Vmx'])[0:t_max,:]
    # Vmx_reshape = Vmx.reshape((len(Vmx),J,I))[:,1:114,1:125]
    
    x = np.array(dataset['x'])
    y = np.array(dataset['y'])
    x_reshape = x.reshape(J,I)[1:114,1:125]
    y_reshape = y.reshape(J,I)[1:114,1:125]
    
    # nonanUmx = np.invert(np.isnan(Umx))
    # nonanUmx_reshape = np.invert(np.isnan(Umx_reshape))
    # nonanVmx = np.invert(np.isnan(Vmx))
    # nonanVmx_reshape = np.invert(np.isnan(Vmx_reshape))
    
    t_3d,y_3d,x_3d = np.meshgrid(np.arange(t_max, step=0.05),y_reshape[:,0],x_reshape[0,:],indexing='ij')
    
    nonan = np.invert(np.isnan(Umx_reshape.ravel()))
    
    # Umx_lang = Umx_reshape.ravel()[nonan]
    # Vmx_lang = Vmx_reshape.ravel()[nonan]
    t_lang = t_3d.ravel()[nonan]
    x_lang = x_3d.ravel()[nonan]
    y_lang = y_3d.ravel()[nonan]
    
    # uvw = (0,-88.5,87)
    txy = np.vstack((t_lang,x_lang,y_lang)).T

    # interpolate.griddata((t_lang, x_lang, y_lang), Umx_lang, uvw, method='linear')
        
    # tri = qhull.Delaunay(txy)
    # simplex = tri.find_simplex(uvw)
    # vertices = np.take(tri.simplices, simplex, axis=0)
    # temp = np.take(tri.transform, simplex, axis=0)
    # delta = uvw - temp[3, :]
    # bary = np.einsum('njk,nk->nj', temp[:, :d, :], delta)
    # wts = np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True))
                    
    # interpolate = np.einsum('nj,nj->n', np.take(values, vtx), wts)
    
    tree = cKDTree(txy)
    # Umx_lang[tree.query(uvw)[1]]
    # dist, i = tree.query(uvw)
    
    return tree


def vegglov(u_star, y, v):
    # nu = 1 # 1 mm²/s
    y = y - h
    ks = .0025
    return 1/0.4 * log(30 * y / ks) - v/u_star

def finn_u(y,v):
    u = np.zeros(127)
    
    for i in np.arange(0,67):
        u[i]= fsolve(vegglov, 2, args=(y[i],v[i]))
     
    return u

def draw_rect(axes,color='red'):
    axes.add_patch(Rectangle((-62.4,-9.56),50,8,linewidth=2,edgecolor=color,facecolor='none'))
    axes.add_patch(Rectangle((37.6,-8.5),50,8,linewidth=2,edgecolor=color,facecolor='none'))

def draw_shade(axes, x0=0, x=430, color='red'):
    axes.add_patch(Rectangle((x0,-9.8),x,10.8,linewidth=2,edgecolor='none',facecolor='lightcoral'))

def ranges():
    y_range = np.s_[0:114]
    x_range = np.s_[40:108]
    
    piv_range = np.index_exp[y_range,x_range]
    
    return piv_range


# https://stackoverflow.com/questions/20915502/speedup-scipy-griddata-for-multiple-interpolations-between-two-irregular-grids

def interp_weights(tri, uv):
   
    simplex = tri.find_simplex(uv)
    vertices = np.take(tri.simplices, simplex, axis=0)
    temp = np.take(tri.transform, simplex, axis=0)
    delta = uv - temp[:, 2]
    bary = np.einsum('njk,nk->nj', temp[:2, :], delta)
    wts = np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))
    ret = np.einsum('nj,nj->n', np.take(uv, vertices), wts)

def interpol(coords, values, yn):
    
    #ret[np.any(wts < 0, axis=1)] = fill_value
    return ret



def interp_lin_near(coords,values, yn):
    new = interpolate.griddata(coords, values, yn, method='linear')
    if np.isnan(new):
        return interpolate.griddata(coords, values, yn, method='nearest')
    else:
        return new

def lag_ft(case, t_start, t_end, fps=20):
    ''' Funksjon for å laga eit kontinuerleg vektorfelt '''
    
    nonanxindex = np.array(case['nonanxindex'])
    nonanyindex = np.array(case['nonanyindex'])
    Umx = np.array(case['Umx'])
    Vmx = np.array(case['Vmx'])
    x = np.array(case['x'])
    y = np.array(case['y'])
    
    nonanx={}
    nonany={}
    nonanu={}
    nonanv={}
    trix={}
    triy={}
    
    for t in np.arange(t_start*fps, t_end*fps):
        
        nonanx[t]=np.vstack((x[nonanxindex[t]],y[nonanxindex[t]])).T
        nonany[t]=np.vstack((x[nonanyindex[t]],y[nonanyindex[t]])).T
        nonanu[t]=Umx[t,nonanxindex[t]]
        nonanv[t]=Vmx[t,nonanyindex[t]]
        trix[t] = qhull.Delaunay(nonanx[t])
        triy[t] = qhull.Delaunay(nonany[t])
        print(t, end = '')
        print(' ', end = '')
        
    def f_t(t, yn):
        
        if yn[0] > 100:
            return np.hstack([0,0])
        
        t_0 = floor(t)
        t_1 = ceil(t)
        
        if t_0 == t_1:
            u_0 = interp_lin_near((nonanx[t_0], nonany[t_0]), nonanu[t_0], yn, tri) #interpolate.griddata((x[nonanxindex[t_0]], y[nonanxindex[t_0]]), Umx[t_0,nonanxindex[t_0,:]], yn)
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
    
def rk(t0, y0, L, f, h=0.02):
    ''' Heimelaga Runge-Kutta-metode '''
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

class Particle:
    def __init__(self, position, diameter, density=2600, velocity=0 ):
        self.position = position
        self.diameter= diameter
        self.density = density
        self.velocity = velocity
    
    
    
    

def lag_sti(case, t_start,t_end,fps=20):
    

    
    f_t = lag_ft(case, t_start,t_end,fps=20)
    
    p_x,p_y = np.meshgrid([-90,-200],[85,75,65,55,45,35,25,15,5,0,-20,-30,-40,-50,-60])
    
    p_x = p_x.T.reshape(-1)
    p_y= p_y.T.reshape(-1)
    
    sti = []
    
 
    for par in np.column_stack((p_x,p_y)):
        sti.append(solve_ivp(f_t, [t_start,t_end*fps], par, t_eval=np.arange(t_start, t_end*fps, 1)))
        
    sti_ny=[]
    
    for el in sti:
        sti_ny.append(el.y.T)
    
    return np.array(sti_ny)
    

def sti_animasjon(case):
    
    x_reshape1= np.array(case['x_reshape1'])
    y_reshape1 = np.array(case['y_reshape1'])
    V_mag_reshape = np.array(case['V_mag_reshape'])
    sti = np.array(case['sti'])
    
    fig, ax = plt.subplots()
    
    field = ax.imshow(V_mag_reshape[0,:,:], extent=[x_reshape1[0,0],x_reshape1[0,-1], y_reshape1[-1,0], y_reshape1[0,0]])
    particle, =ax.plot(sti[:,0,0], sti[:,0,1], 'ro')
    ax.set_xlim([x_reshape1[0,0],x_reshape1[0,-1]])
    draw_rect(ax)
    
    def nypkt(i):
        field.set_data(V_mag_reshape[i,:,:])
        particle.set_data(sti[:,i,0], sti[:,i,1])
        return field,particle
    
    print("Skal byrja på filmen")
    #ax.axis('equal')
    ani = animation.FuncAnimation(fig, nypkt, frames=np.arange(1,600),interval=50)
    plt.show()
    print("ferdig med animasjon, skal lagra")
    
    filnamn = "stiQ{}.mp4".format(re.split(r'/',case.name)[-1])
    ani.save(filnamn)
    plt.close()


def lagra(dataset):
    f = h5py.File('alle2.hdf5','w')
    
    vassf = f.create_group("vassføringar")
    
    for q in dataset:
        gr = vassf.create_group(str(q))
        for k in dataset[q]:
            gr.create_dataset(k, data=dataset[q][k], compression="gzip", compression_opts=9)
    f.close()
  
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



 
