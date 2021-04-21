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
import pickle
import re
import scipy.stats as stats

import scipy.interpolate as spint
import scipy.spatial.qhull as qhull
import itertools
from scipy.spatial import cKDTree

import os.path

from math import ceil, floor, log, sqrt, pi, hypot

# import os.path.join as pjoin

if os.path.isfile("D:/Tonstad/utvalde/Q40.hdf5"):
    filnamn = "D:/Tonstad/utvalde/Q40.hdf5"
elif os.path.isfile("C:/Users/havrevol/Q40.hdf5"):
    filnamn = "C:/Users/havrevol/Q40.hdf5"
else:
    filnamn ="D:/Tonstad/Q40.hdf5"
    
if os.path.isfile("D:/Tonstad/Q40_20s.pickle"):
    pickle_fil = "D:/Tonstad/Q40_20s.pickle"
elif os.path.isfile("C:/Users/havrevol/Q40_2s.pickle"):
    pickle_fil = "C:/Users/havrevol/Q40_2s.pickle"
else:
    pickle_fil ="D:/Tonstad/Q40_2s.pickle"

# fil = h5py.File("D:/Tonstad/alle.hdf5", 'a')
# vass = fil['vassføringar']


discharges = [20,40,60,80,100,120,140]

h= -5.9


def lag_tre(t_max=1, dataset = h5py.File(filnamn, 'r'), nearest=True):
    '''
    Ein metode som tek inn eit datasett og gjer alle reshapings-tinga for x og y, u og v og Re.

    '''
    
    (I,J)=(int(np.array(dataset['I'])),int(np.array(dataset['J'])))

    steps = t_max * 20
    
    Umx = np.array(dataset['Umx'])[0:steps,:]
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

    # interpolate.griddata((t_lang, x_lang, y_lang), Umx_lang, uvw[0], method='linear')

    import time

    start = time.time()        
    if (nearest):
        tree = cKDTree(txy)
    else:
        tree = qhull.Delaunay(txy)
    
    end = time.time()
    print(end - start)
    
    # simplex = tri.find_simplex(uvw)
    # vertices = np.take(tri.simplices, simplex, axis=0)
    # temp = np.take(tri.transform, simplex, axis=0)
    # delta = uvw - temp[3, :]
    # bary = np.einsum('njk,nk->nj', temp[:, :d, :], delta)
    # wts = np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True))
                    
    # interpolate = np.einsum('nj,nj->n', np.take(values, vtx), wts)    
    
    # Umx_lang[tree.query(uvw)[1]]
    # dist, i = tree.query(uvw)
    
    return tree

def hent_tre(fil=pickle_fil):
    with open(fil, 'rb') as f:
        tri = pickle.load(f)
 
    return tri

def get_velocity_data(t_max=1):
    steps = t_max * 20
    
    with h5py.File(filnamn, 'r') as f:
        Umx = np.array(f['Umx'])[0:steps,:]
        Vmx = np.array(f['Vmx'])[0:steps,:]
        (I,J) = (int(np.array(f['I'])),int(np.array(f['J'])))
    
    Umx_reshape = Umx.reshape((len(Umx),J,I))[:,1:114,1:125].ravel()
    Vmx_reshape = Vmx.reshape((len(Vmx),J,I))[:,1:114,1:125].ravel()
    
    nonan = np.invert(np.isnan(Umx_reshape))
        
    return Umx_reshape[nonan], Vmx_reshape[nonan]

Umx_lang, Vmx_lang = get_velocity_data(2)
tri = hent_tre()

# Så dette er funksjonen som skal analyserast av runge-kutta-operasjonen. Må ha t som fyrste og y som andre parameter.
def U(t, x):
    '''
    https://stackoverflow.com/questions/20915502/speedup-scipy-griddata-for-multiple-interpolations-between-two-irregular-grids    

    Parameters
    ----------
    tri : spatial.qhull.Delaunay
        Eit tre med data.
    Umx_lang : Array of float64
        Fartsdata i ei lang remse med same storleik som tri.
    x : Array of float64
        Eit punkt i tid og rom som du vil finna farten i.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    x = np.concatenate(([t], x))
    d=3
    simplex = tri.find_simplex(x)
    vertices = np.take(tri.simplices, simplex, axis=0)
    temp = np.take(tri.transform, simplex, axis=0)
    delta = x - temp[d]
    bary = np.einsum('jk,k->j', temp[:d, :], delta)
    wts = np.hstack((bary, 1 - bary.sum(axis=0, keepdims=True)))
                
    return np.einsum('j,j->', np.take(Umx_lang, vertices), wts), np.einsum('j,j->', np.take(Vmx_lang, vertices), wts)
    

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


g = 9.81
nu = 1e-6
dt = 1
rho = 1000

class Particle:
    #Lag ein tabell med tidspunkt og posisjon for kvar einskild partikkel.
    def __init__(self, initPosition, diameter, density=1600, velocity=0 ):
        self.diameter= diameter
        self.density = density
        self.force = 0
        self.timeposvel = np.array([0,initPosition,velocity])
        
    def get_mass(self):
        # V = 4/3 πr³
        return self.diameter**3 * pi * 1/6
    
    mass = property(get_mass)
    
    def get_radius(self):
        return self.diameter/2
    
    radius = property(get_radius)
    
    def get_abs_vel(self):
        return hypot(self.velocity)
    
    abs_vel = property(get_abs_vel)    
        
    def moveObject(self):
        # ball.pos2D = ball.pos2D.addScaled(ball.velo2D,dt);
        self.position += self.velocity * dt
                
    def updateAccel(self):    
        pass
        
    def updateVelo(self):
        pass
    
    def move(self):
        self.moveObject(self)
        self.calcForce(self)
        self.updateAccel(self)
        self.updateVelo(self) 
        
    def calcForce(self):
        # var dr = (yball-yLevel)/rball;
        # var drag = ball.velo2D.multiply(-ratio*k*ball.velo2D.length());
        # force = Forces.add([gravity, upthrust, drag]);
        
        gravity = self.mass * g
        
        #drag = D = Cd * A * .5 * r * V²
       
        R = self.velocity * self.diameter / nu
        
        cd = 24 / R
        
        drag = 3/4 * cd/self.diameter * rho/self.density * self.velocity**2
    
        
    def f(self, t,x):
        '''
        Sjølve differensiallikninga med t som x, og x som y (jf. Kreyszig)
        Så x er ein vektor med to element, nemleg x[0] = posisjon og x[1] = fart
    
        Parameters
        ----------
        t : TYPE
            DESCRIPTION.
        x : TYPE
            DESCRIPTION.
    
        Returns
        -------
        None.
    
        '''
        dx_dt = x[1]
        
        R = self.velocity * self.diameter / nu
        cd = 24 / R
        
        du_dt= 3/4 * cd / self.diameter * rho / self.density * abs(U(t,x) - x[1])*(U(t,x) - x[1]) + (rho / self.density - 1)* g
        
        return dx_dt,du_dt
    

def lag_sti(case, t_start,t_end,fps=20):
    # f_t = lag_ft(case, t_start,t_end,fps=20)
    
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


def draw_rect(axes,color='red'):
    axes.add_patch(Rectangle((-62.4,-9.56),50,8,linewidth=2,edgecolor=color,facecolor='none'))
    axes.add_patch(Rectangle((37.6,-8.5),50,8,linewidth=2,edgecolor=color,facecolor='none'))

def ranges():
    y_range = np.s_[0:114]
    x_range = np.s_[40:108]
    
    piv_range = np.index_exp[y_range,x_range]
    
    return piv_range



 
