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
    
if os.path.isfile("D:/Tonstad/Q40_6s.pickle"):
    pickle_fil = "D:/Tonstad/Q40_6s.pickle"
elif os.path.isfile("C:/Users/havrevol/Q40_2s.pickle"):
    pickle_fil = "C:/Users/havrevol/Q40_2s.pickle"
else:
    pickle_fil ="D:/Tonstad/Q40_2s.pickle"

# fil = h5py.File("D:/Tonstad/alle.hdf5", 'a')
# vass = fil['vassføringar']


discharges = [20,40,60,80,100,120,140]

h= -5.9

def norm(v):
    return v / (v**2).sum()**0.5

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

def lagra_tre(tre, fil):
    with open(fil, 'wb') as f:
        pickle.dump(tre, f)

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
    
    # u_bar = np.nanmean(Umx,0)
    # v_bar = np.nanmean(Vmx,0)

    # u_reshape = u_bar.reshape((J,I))[1:114,1:125]
    # v_reshape = v_bar.reshape((J,I))[1:114,1:125]
    
    nonan = np.invert(np.isnan(Umx_reshape))
        
    return Umx_reshape[nonan], Vmx_reshape[nonan] #, u_reshape, v_reshape


# Så dette er funksjonen som skal analyserast av runge-kutta-operasjonen. Må ha t som fyrste og y som andre parameter.
def U(t, x, tri, ckdtre, Umx_lang, Vmx_lang):
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
    if (simplex==-1):
        return Umx_lang[ckdtre.query(x)[1]], Vmx_lang[ckdtre.query(x)[1]]
        # Her skal eg altså leggja inn å sjekka eit lite nearest-tre for næraste snittverdi.
        # raise Exception("Coordinates outside the complex hull")
        
    vertices = np.take(tri.simplices, simplex, axis=0)
    temp = np.take(tri.transform, simplex, axis=0)
    delta = x - temp[d]
    bary = np.einsum('jk,k->j', temp[:d, :], delta)
    wts = np.hstack((bary, 1 - bary.sum(axis=0, keepdims=True)))
                
    return np.einsum('j,j->', np.take(Umx_lang, vertices), wts), np.einsum('j,j->', np.take(Vmx_lang, vertices), wts)
    
cd = interpolate.interp1d(np.array([0.001,0.01,0.1,1,10,20,40,60,80,100,200,400,600,800,1000,2000,4000,6000,8000,10000,100000]), np.array([2.70E+04,2.40E+03,2.50E+02,2.70E+01,4.40E+00,2.80E+00,1.80E+00,1.45E+00,1.25E+00,1.12E+00,8.00E-01,6.20E-01,5.50E-01,5.00E-01,4.70E-01,4.20E-01,4.10E-01,4.15E-01,4.30E-01,4.38E-01,5.40E-01,]))
    

    
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

def rk_2(f, y0, L, h, tri, Umx_lang, Vmx_lang):
    ''' Heimelaga Runge-Kutta-metode '''
    t0, t1 = L
    N=int((t1-t0)/h)

    t=[0]*N # initialize lists
    y=[0]*N # initialize lists
    
    t[0] = t0
    y[0] = y0
    
    for n in range(0, N-1):
        #print(n,t[n], y[n], f(t[n],y[n]))
        k1 = h*f(t[n], y[n], tri, Umx_lang, Vmx_lang)
        k2 = h*f(t[n] + 0.5 * h, y[n] + 0.5 * k1, tri, Umx_lang, Vmx_lang)
        k3 = h*f(t[n] + 0.5 * h, y[n] + 0.5 * k2, tri, Umx_lang, Vmx_lang)
        k4 = h*f(t[n] + h, y[n] + k3, tri, Umx_lang, Vmx_lang)
        
        if (np.isnan(k4+k3+k2+k1).any()):
            #print(k1,k2,k3,k4)
            return t,y
        
        t[n+1] = t[n] + h
        y[n+1] = y[n] + 1/6 * (k1 + 2*k2 + 2*k3 + k4)
        
    return t, y

def rk_3 (f, t, y0):
    resultat = solve_ivp(f, t, y0,  t_eval = [t[1]], args=(tri, ckdtre, Umx_lang, Vmx_lang))
    
    return np.concatenate((resultat.t, resultat.y.T[0]))

g = np.array([0, 9.81e3]) # mm/s^2 = 9.81 m/s^2
nu = 1 # 1 mm^2/s = 1e-6 m^2/s
dt = 1 # s
rho = 1e-6  # kg/mm^3 = 1000 kg/m^3 

class Particle:
    #Lag ein tabell med tidspunkt og posisjon for kvar einskild partikkel.
    def __init__(self, initPosition, diameter, density=2.65e-6, velocity=0 ):
        self.diameter= diameter
        self.density = density
        self.force = 0
        self.timeposvel = np.array([0,initPosition[0],initPosition[1],velocity])
        
    def get_mass(self):
        # V = 4/3 πr³
        return self.diameter**3 * pi * 1/6
    
    mass = property(get_mass)
    
    def get_radius(self):
        return self.diameter/2
    
    radius = property(get_radius)
    
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
            
        
    def f(self, t, x, tri, ckdtre, Umx_lang, Vmx_lang):
        """
        Sjølve differensiallikninga med t som x, og x som y (jf. Kreyszig)
        Så x er ein vektor med to element, nemleg x[0] = posisjon og x[1] = fart.
        Men for at solve_ivp skal fungera, må x vera 1-dimensjonal. Altså. 
        x= [x,y,u,v], ikkje x = [[x,y], [u,v]].
        f (t, x, y) = [dx/dt, du/dt]

        Parameters
        ----------
        t : double
            Tidspunktet for funksjonen.
        x : tuple
            Ein tuple med koordinatane og farten, altså (x0, y0, u0, v0).
        tri : spatial.qhull.Delaunay
            Samling av triangulerte data.
        Umx_lang : Array of float64
            Ein nd-array med horisontale fartsdata.
        Vmx_lang : Array of float64
            Ein nd-array med vertikale fartsdata.

        Returns
        -------
        tuple
             Ein tuple med [dx/dt, du/dt]

        """
        dx_dt = np.array([x[2], x[3]])
        # vel = np.array([100,0]) - dx_dt # relativ snøggleik
        vel = np.array(U(t,np.array([x[0],x[1]]),tri, ckdtre, Umx_lang, Vmx_lang)) - dx_dt # relativ snøggleik
        
        Re = hypot(vel[0],vel[1]) * self.diameter / nu 
        
        if (Re<1000):
            try:
                cd = 24 / Re * (1+0.15*Re**0.687)
                # Cheng (1997) skildrar Cd for kantete og runde steinar. Dette 
                # er kanskje den viktigaste grunnen til at eg bør gjera dette?
                # Ferguson og Church (2004) gjev nokre liknande bidrag, men viser til Cheng.
            except ZeroDivisionError:
                cd = 2e4
        else:
            cd = 0.44
        
        # print("Re = ", Re," cd= ", cd)
        drag_component =  3/4 * cd / self.diameter * rho / self.density * abs(vel)*vel
        gravity_component = (rho / self.density - 1) * g
        
        # print("drag_component =",drag_component,", gravity_component = ",gravity_component)
        
        # du_dt= 3/4 * cd / self.diameter * rho / self.density * abs(np.array(U(t,x[0])) - np.array(x[1]))*(np.array(U(t,x[0])) - np.array(x[1])) + (rho / self.density - 1)* g
        du_dt = drag_component + gravity_component
        
        return np.concatenate((dx_dt,du_dt))
    
    def checkCollision(self, data, rib):
        """
        Sjekkar kollisjonar mellom ein partikkel med ein posisjon og ei ribbe.

        Parameters
        ----------
        data : tuple
            Ein tuple (x,y,u,v) som gjev koordinatane og farten til senter av partikkelen.
        rib : Rib
            Den aktuelle ribba, altså eit rektangel.

        Returns
        -------
        tuple
            Ein tuple med fylgjande data: (boolean, collisionInfo, rib). 
            CollisionInfo er ein tuple med (collision depth, rib normal, punkt 
                                            på partikkelen som kolliderer, relativ fart).

        """
        
        position = data[0:2]        
        inside = True
        bestDistance = -99999
        nearestEdge = 0
        
        collisionInfo = (-1,np.array([-1,-1]), np.array([-1,-1]), -1)
        
        #Step A - compute nearest edge
        vertices = rib.vertices
        normals = rib.normals
        
        for i in range(len(vertices)):
            v = position - vertices[i]
            projection = np.dot(v, normals[i])
            if (projection > 0):
                # if the center of circle is outside of rectangle
                bestDistance = projection
                nearestEdge = i
                inside = False
                break
            
            if (projection > bestDistance):
                # If the center of the circle is inside the rectangle
                bestDistance = projection
                nearestEdge = i
                
        
        if (not inside):            
            #  Step B1: If center is in Region R1
            # the center of circle is in corner region of mVertex[nearestEdge]

            # //v1 is from left vertex of face to center of circle
            # //v2 is from left vertex of face to right vertex of face
            v1 = position - vertices[nearestEdge]
            v2 = vertices[(nearestEdge + 1) % 4] - vertices[nearestEdge]
            
            dot = np.dot(v1, v2)
            
            if (dot < 0): #region R1
                dis = np.sqrt(v1.dot(v1))
                
                if (dis > self.radius):
                    return (False, collisionInfo, rib) # må vel endra til (bool, depth, normal, start)
                
                normal = norm(v1)
                
                radiusVec = normal*self.radius*(-1)
                
                # sender informasjon til collisioninfo:                    
                collisionInfo = (self.radius - dis, normal, position + radiusVec)
                
            else:
                # //the center of circle is in corner region of mVertex[nearestEdge+1]
        
                #         //v1 is from right vertex of face to center of circle 
                #         //v2 is from right vertex of face to left vertex of face
                v1 = position - vertices[(nearestEdge +1) % 4]
                v2 = (-1) * v2
                dot = v1.dot(v2)
                
                if (dot < 0):
                    dis = np.sqrt(v1.dot(v1))
                                        
                    # //compare the distance with radium to decide collision
            
                    if (dis > self.radius):
                        return (False, collisionInfo, rib)
                    
                    normal = norm(v1)
                    radiusVec = normal * self.radius*(-1)
                    
                    collisionInfo = (self.radius - dis, normal, position + radiusVec)
                else:
                    #//the center of circle is in face region of face[nearestEdge]
                    if (bestDistance < self.radius):
                        radiusVec = normals[nearestEdge] * self.radius
                        collisionInfo = (self.radius - bestDistance, normals[nearestEdge], position - radiusVec)
                    else:
                        return (False, collisionInfo, rib)
        else:
            #     //the center of circle is inside of rectangle
            radiusVec = normals[nearestEdge] * self.radius
            collisionInfo = (self.radius - bestDistance, normals[nearestEdge], position - radiusVec, -1)
            
            return (True, collisionInfo, rib) 
            # Måtte laga denne returen så han ikkje byrja å rekna ut relativ fart når partikkelen uansett er midt inne i ribba.
 
        # Rekna ut relativ fart, jamfør Baraff (2001) formel 8-3.
        n = collisionInfo[1]
        v = np.array(data[2:])
        v_rel = np.dot(n,v)
        collisionInfo = (collisionInfo[0], collisionInfo[1],collisionInfo[2], v_rel)
                
        if (v_rel < 0): # Sjekk om partikkelen er på veg vekk frå veggen.
            is_collision = True
        else:
            is_collision = False
        
        return (is_collision, collisionInfo, rib)
    
class Rib:
    def __init__(self, origin, width, height):
        # Bør kanskje ha informasjon om elastisiteten ved kollisjonar òg?
        self.origin = np.array(origin)
        self.width = width
        self.height = height
        
    def get_vertices(self): # Går mot klokka
        return [self.origin,
                self.origin + np.array([self.width,0]),
                self.origin + np.array([self.width, self.height]),
                self.origin + np.array([0, self.height])]
    
    vertices = property(get_vertices)
    
    def get_face_normal(self):
        vertices = self.vertices
        # 0 - botn, 1 -- høgre, 2 -- topp, 3 -- venstre
        return [norm(vertices[1]-vertices[2]), 
                norm(vertices[2]-vertices[3]),
                norm(vertices[3]-vertices[0]),
                norm(vertices[0]-vertices[1])]
    
    normals = property(get_face_normal)
        
       
# p_x,p_y = np.meshgrid([-90,-200],[85,75,65,55,45,35,25,15,5,0,-20,-30,-40,-50,-60])
    
# p_x = p_x.T.reshape(-1)
# p_y= p_y.T.reshape(-1)

def lag_sti(part, x0, t_span,fps=20):
    
    # stien må innehalda posisjon, fart og tid.
    sti = []
    sti_komplett = []
    
    step_old = np.concatenate(([t_span[0]], x0))
    # Step_old og step_new er ein array med [t, x, y, u, v]. 
    
    sti.append(step_old)
    sti_komplett.append(step_old)
    
    # finn neste steg med rk_2 og standard tidssteg.
    # sjekk kollisjon. Dersom ikkje kollisjon, bruk resultat frå rk_2 og gå til neste steg
    # Dersom kollisjon: halver tidssteget, sjekk kollisjon. Dersom ikkje kollisjon

    t = t_span[0]
    t_main = t
    dt_main = 1/fps
    dt = dt_main
    eps = 0.001
    rest = 0.5
    
    while (t < t_span[1]):
        # step_new = rk_2(part.f, step_old, (t, t+dt), 0.01, tri, Umx_lang, Vmx_lang)
        step_new = rk_3(part.f, (t,t+dt), step_old[1:])
        
        for rib in ribs:
            collision_info = part.checkCollision(step_new[1:], rib)
            if (collision_info[0]):
                break
           
        if (collision_info[0]):
            if (collision_info[1][0] < eps):
                #Gjer alt som skal til for å endra retningen på partikkelen
                n = collision_info[1][1]
                v = step_new[3:]
                v_rel = collision_info[1][3]
                v_new = v - (rest + 1) * v_rel * n
                step_old = np.copy(step_new)
                step_old[3:] = v_new
                
                #Fullfør rørsla fram til hovud-steget, vonleg med rett retning 
                # og fart, så ein kjem inn i rett framerate igjen.
                dt = t_main + dt_main - t
                                
                # step_new = rk_3(part.f, (t, t_main+dt_main), step_old[1:])
                
                if (abs(v_rel) < 0.1):
                    sti_komplett.append(step_new)
                    sti.append(step_new)
                    break
                
                # t = t_main + dt_main
                
            else:
                dt = dt/2
                continue
        else:
            sti_komplett.append(step_new)
            step_old = step_new
            
            t = t + dt
            
            if (round(step_new[0]*10000) % round(dt_main*10000) == 0):
                sti.append(step_new)
                t_main = step_new[0]
                t = t_main
                dt = dt_main
                
            
                


    
    # for par in np.column_stack((p_x,p_y)):
    #     sti.append(solve_ivp(particle.f, [t_start,t_end*fps], par, t_eval=np.arange(t_start, t_end*fps, 1)))
        
    # sti_ny=[]
    
    # for el in sti:
    #     sti_ny.append(el.y.T)
    
    return np.array(sti)
            
# #%% Førebu

# %timeit U(random.randint(0,20), [random.uniform(-88,88), random.uniform(-70,88)], tri, Umx_lang, Vmx_lang)
Umx_lang, Vmx_lang = get_velocity_data(6)
tri = hent_tre()
ckdtre = lag_tre(t_max=6)



# f_retur = stein.f(0,[-88.5,87,100,-155], tri, Umx_lang, Vmx_lang)


# #%% Test løysing av difflikning

# svar_profft = solve_ivp(stein.f,(0.375,0.4), np.array([-88.5,87,0,0]), args=(tri, Umx_lang, Vmx_lang))

# svar_profft2 = rk_3(stein.f, (0.375,0.4), np.array([-88.5,87,0,0]))

# svar = rk_2(stein.f, np.array([-88.5,87,0,0]), (0,0.4), 0.01, tri, Umx_lang, Vmx_lang)

# #%% Test kollisjon
# stein2 = Particle([-80,50],3)
# koll = stein2.checkCollision([-63,-1], ribs[0]) #R2
# koll2 = stein2.checkCollision([-40,-1], ribs[0]) #R3 (midten av flata)


#%% Test å¨laga sti
# #%% Utfør

ribs = [Rib((-62.4,-9.56),50,8), 
        Rib((37.6,-8.5), 50, 8), 
        Rib((-100,-74.3), 200, -10)]

stein = Particle([-88.5,87],1) #Partikkel med koordinatar og 1 mm diameter

stein2 = Particle([-85,60], 0.5) #Partikkel med koordinatar og 0.5 mm diameter

stien1 = lag_sti(stein, [-88.5,87,0,0],(0,5))
stien2 = lag_sti(stein2, [-85,60,0,0],(0,5))


#%% ikkje bruk

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


# Her er ein funksjon for fritt fall av ein 1 mm partikkel i vatn.
# d u_p/dt = u_p(t,y) der y er vertikal fart. Altså berre modellert drag og gravitasjon.
def u_p(t,y):
    if(y==0):
        cd=1e4
    else:
        cd=24/abs(0-y)*(1+0.15*abs(0-y)**0.687)
        
    return 3/4*(0-y)*abs(0-y)*cd*1/2.65-9810*1.65/2.65
 
