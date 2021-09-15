# -*- coding: utf-8 -*-
'''køyr funksjonar som plottingar(fil['vassføringar'])'''
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# from cycler import cycler
plt.rcParams["font.family"] = "STIXGeneral"
plt.rcParams['mathtext.fontset'] = 'stix'
from matplotlib import animation

import numpy as np
# from scipy import interpolate
from scipy.integrate import solve_ivp  # https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html#r179348322575-1

import h5py
import ray
# from numba import jit

from math import pi, hypot #, atan2


# fil = h5py.File("D:/Tonstad/alle.hdf5", 'a')
# x = np.array(h5py.File(filnamn, 'r')['x']).reshape(127,126)[ranges()]
# vass = fil['vassføringar']


from hjelpefunksjonar import norm, draw_rect, ranges, finn_fil, sortClockwise


filnamn = "../Q40.hdf5" #finn_fil(["D:/Tonstad/utvalde/Q40.hdf5", "C:/Users/havrevol/Q40.hdf5", "D:/Tonstad/Q40.hdf5"])

t_max_global = 20
t_min_global = 0

nullfart = np.zeros(2)

# Så dette er funksjonen som skal analyserast av runge-kutta-operasjonen. Må ha t som fyrste og y som andre parameter.
# @jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
def get_u(t, x_inn, radius, tre_plasma, linear=True, lift = True, addedmass = True):
    '''
    https://stackoverflow.com/questions/20915502/speedup-scipy-griddata-for-multiple-interpolations-between-two-irregular-grids    

    Parameters
    ----------
    tri : spatial.qhull.Delaunay
        Eit tre med data.
    U : Tuple
        Fartsdata i ei lang remse med same storleik som tri. Det er (U,V,dU/dt,dV/dt), altså gradienten i kvart punkt og kvar tid.
    x : Array of float64
        Eit punkt i tid og rom som du vil finna farten i.
    linear : Bool
        Skal det interpolerast lineært eller næraste nabo?

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''    
    tx = np.concatenate(([t], x_inn[:2]))
    U_p = x_inn[2:]
        
    dt, dx, dy = 0.01, 0.1, 0.1
    
    tre_samla = ray.get(tre_plasma)

    U_del = tre_samla.get_U(tx)
    tri = tre_samla.get_tri(tx)
    
    x = np.vstack((tx,tx + np.array([dt,0,0]), tx + np.array([0,dx,0]), tx +np.array([0,0,dy])))
        
    kdtre = tre_samla.kdtre
    U_kd = tre_samla.U_kd
    
    get_u.counter +=1
    
    # if (get_u.counter > 2000000):
    #     raise Exception("Alt for mange iterasjonar")
    
    if(linear):
        d=3
        # simplex = tri.find_simplex(x)
        simplex = np.tile(tri.find_simplex(x[0]), 4)
        
        if (np.any(simplex==-1)):
            get_u.utanfor += 1
            while True:
                try:
                    U_kd[:,kdtre.query(x[0])[1]]
                    break
                except IndexError:
                    x[np.abs(x)>1e100] /= 1e10
                    
                
            return U_kd[:,kdtre.query(x[0])[1]], nullfart, np.vstack((nullfart,nullfart))
            # Gjer added mass og lyftekrafta lik null, sidan den 
          
        vertices = np.take(tri.simplices, simplex, axis=0)
        temp = np.take(tri.transform, simplex, axis=0)
                
        delta = x - temp[:,d]
        bary = np.einsum('njk,nk->nj', temp[:,:d, :], delta)
        wts = np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))
        
        # U_f = np.einsum('nij,ni->nj', np.take(np.column_stack(U_del),vertices,axis=0),wts)
        # U_f = np.einsum('ijn,ij->n', np.take(U_del, vertices, axis=0), wts)
        U_f = np.einsum('jni,ni->jn', np.take(U_del,vertices,axis=1),wts)
        
        if (addedmass):
            dUdt = (U_f[:,1] - U_f[:,0]) / dt # Fyrste verdien er dU/dt og andre er dV/dt
            dUdx = (U_f[:,2] - U_f[:,0]) / dx # Fyrste verdien er dU/dx og andre er dV/dy
            dUdy = (U_f[:,3] - U_f[:,0]) / dt # Fyrste verdien er dU/dy og andre er dV/dy
    
            # skal finna gradienten i t, u og v-retning for å bruka på added mass.
            # DU/Dt = dU/dt + u * dU/dx + v*dU/dy
            
            dudt_material = dUdt + U_f[0,0] * dUdx + U_f[1,0] * dUdy
        else:
            dudt_material = nullfart

        if (lift):
            # skal finna farten i passande punkt over og under partikkelen for lyftekraft
            U_rel = U_f[:, 0] - U_p
            
            particle_top =    x_inn[0:2] + radius * norm(U_rel) @ np.array([[0, 1],[-1, 0]])
            particle_bottom = x_inn[0:2] + radius * norm(U_rel) @ np.array([[0, -1],[1, 0]])
            
            part = np.array([[t, particle_top[0], particle_top[1]], [t, particle_bottom[0], particle_bottom[1]] ])
            
            
            part_simplex = simplex[:2]
            part_vertices = vertices[:2]
            part_temp = temp[:2]
                
            while (True):
                part_delta = part - part_temp[:,d] #avstanden til referansepunktet i simpleksen.
                part_bary = np.einsum('njk,nk->nj', part_temp[:,:d, :], part_delta) 
                part_wts = np.hstack((part_bary, 1 - part_bary.sum(axis=1, keepdims=True)))
            
                if (np.any(part_wts < -0.02)):
                    part_simplex = tri.find_simplex(part)
                    part_vertices = np.take(tri.simplices, part_simplex, axis=0)
                    part_temp = np.take(tri.transform, part_simplex, axis=0)
                    get_u.simplex_prob += 1
                else:
                    get_u.simplex_prob = 0
                    break
                    
            U_top_bottom = np.einsum('jni,ni->jn', np.take(U_del, part_vertices, axis=1), part_wts)
        else:
            U_top_bottom = np.vstack((nullfart,nullfart))
                    
        return (U_f[:,0], dudt_material, U_top_bottom)
        # return np.einsum('j,j->', np.take(U_del[0], vertices), wts), np.einsum('j,j->', np.take(U_del[1], vertices), wts),  np.einsum('j,j->', np.take(U_del[2], vertices), wts),  np.einsum('j,j->', np.take(U_del[3], vertices), wts)
    else:
        kd_index = kdtre.query(x)[1]
        
        return U_kd[:,kd_index], nullfart, nullfart
        # return U[0][kd_index], U[1][kd_index], U[2][kd_index], U[3][kd_index]
  
# cd = interpolate.interp1d(np.array([0.001,0.01,0.1,1,10,20,40,60,80,100,200,400,600,800,1000,2000,4000,6000,8000,10000,100000]), np.array([2.70E+04,2.40E+03,2.50E+02,2.70E+01,4.40E+00,2.80E+00,1.80E+00,1.45E+00,1.25E+00,1.12E+00,8.00E-01,6.20E-01,5.50E-01,5.00E-01,4.70E-01,4.20E-01,4.10E-01,4.15E-01,4.30E-01,4.38E-01,5.40E-01,]))
    
get_u.counter = 0
get_u.utanfor = 0
get_u.simplex_prob = 0
    


def rk_2(f, L, y0, h, tri, U):
    ''' Heimelaga Runge-Kutta-metode '''
    t0, t1 = L
    N=int((t1-t0)/h)

    t=[0]*N # initialize lists
    y=[0]*N # initialize lists
    
    t[0] = t0
    y[0] = y0
    
    for n in range(0, N-1):
        #print(n,t[n], y[n], f(t[n],y[n]))
        k1 = h*f(t[n], y[n], tri, U)
        k2 = h*f(t[n] + 0.5 * h, y[n] + 0.5 * k1, tri, U)
        k3 = h*f(t[n] + 0.5 * h, y[n] + 0.5 * k2, tri, U)
        k4 = h*f(t[n] + h, y[n] + k3, tri, U)
        
        if (np.isnan(k4+k3+k2+k1).any()):
            #print(k1,k2,k3,k4)
            return t,y
        
        t[n+1] = t[n] + h
        y[n+1] = y[n] + 1/6 * (k1 + 2*k2 + 2*k3 + k4)
        
    return t, y

def rk_3 (f, t, y0, solver_args):
    resultat = solve_ivp(f, t, y0,   t_eval = [t[1]], **solver_args)
    # har teke ut max_ste=0.02, for det vart aldri aktuelt, ser det ut til.  method=solver_args['method'], args=solver_args['args'],
    
    return np.concatenate((resultat.t, resultat.y[:,0]))


def sti_animasjon(partiklar, t_span, dataset = h5py.File(filnamn, 'r'), utfilnamn="stiQ40.mp4",  fps=20 ):
    
    # piv_range = ranges()
    
    # with h5py.File(filnamn, mode='r') as f:
    #     x, y = np.array(f['x']), np.array(f['y'])
        
    #     I, J = int(np.array(f['I'])),int(np.array(f['J']))
               
    #     x_reshape = x.reshape((127,126))[piv_range]
    #     y_reshape = y.reshape((127,126))[piv_range]
    
    (I,J)=(int(np.array(dataset['I'])),int(np.array(dataset['J'])))

    t_min = t_span[0]
    t_max = t_span[1]

    steps = (t_max-t_min) * fps
    
    t_list = np.arange(t_min*fps,t_max*fps)/fps
    
    piv_range = ranges()
    
    Umx = np.array(dataset['Umx'])[t_min*fps:t_max*fps,:]
    Umx_reshape = Umx.reshape((len(Umx),J,I))[:,piv_range[0],piv_range[1]]
    Vmx = np.array(dataset['Vmx'])[t_min*fps:t_max*fps,:]
    Vmx_reshape = Vmx.reshape((len(Vmx),J,I))[:,piv_range[0],piv_range[1]]
    
    x = np.array(dataset['x'])
    y = np.array(dataset['y'])
    x_reshape = x.reshape(J,I)[piv_range]
    y_reshape = y.reshape(J,I)[piv_range]
            
    V_mag_reshape = np.hypot(Umx_reshape, Vmx_reshape)        
    # V_mag_reshape = np.hypot(U[2], U[3])
       
    myDPI = 300
    fig, ax = plt.subplots(figsize=(1190/myDPI,1080/myDPI),dpi=myDPI)
    
    field = ax.imshow(V_mag_reshape[0,:,:], extent=[x_reshape[0,0],x_reshape[0,-1], y_reshape[-1,0], y_reshape[0,0]], interpolation='none')
    # particle, =ax.plot(sti[:,0,1], sti[:,0,2], color='black', marker='o', linestyle=' ', markersize=1)
    
    # https://stackoverflow.com/questions/9215658/plot-a-circle-with-pyplot
    # particle = 
    
    # sti = np.array([part.sti for part in partiklar])    
    
    # prop_cycle = plt.rcParams['axes.prop_cycle']
    # colors = prop_cycle.by_key()['color']
    
    # F8B195   F67280   C06C84   6C5B7B   355C7D 
    # A7226E   EC2049   F26B38   F7DB4F   2F9599
    # A8A7A7   CC527A   E8175D   474747   363636 
    # E5FCC2   9DE0AD   45ADA8   547980   594F4F
    # Palettar kan finnast her: https://digitalsynopsis.com/design/minimal-web-color-palettes-combination-hex-code/
    # https://digitalsynopsis.com/design/how-to-make-color-palettes-from-one-color/

    #Denne oppskrifta lagar eit nytt fargesett: 
    # https://matplotlib.org/stable/gallery/lines_bars_and_markers/markevery_prop_cycle.html#sphx-glr-gallery-lines-bars-and-markers-markevery-prop-cycle-py
    
    
    colors = ['#1f77b4',
          '#ff7f0e',
          '#2ca02c',
          '#d62728',
          '#9467bd',
          '#8c564b',
          '#e377c2',
          '#7f7f7f',
          '#bcbd22',
          '#17becf',
          '#1a55FF']
    
    colors = [ '#F8B195', '#F67280', '#C06C84', '#6C5B7B', '#355C7D', '#A7226E', '#EC2049', '#F26B38', '#F7DB4F', '#2F9599', '#A8A7A7', '#CC527A', '#E8175D', '#474747', '#363636', '#E5FCC2', '#9DE0AD', '#44AA57', '#547980', '#594F4F']
    # matplotlib.rcParams['axes.prop_cycle'] = cycler(color=colors)

    
    
    for part,color in zip(partiklar, colors[:len(partiklar)]):
        circle = plt.Circle((part.sti[0,1], part.sti[0,2]), part.radius, color=color)
        ax.add_patch(circle)
        part.circle = circle
        part.annotation  = ax.annotate("Partikkel", xy=(np.interp(0,part.sti[:,0],part.sti[:,1]), np.interp(0,part.sti[:,0],part.sti[:,2])), xycoords="data",
                        xytext=(-50, 20), fontsize=10, #textcoords="offset points",
                        arrowprops=dict(arrowstyle="->", connectionstyle="arc3, rad=.5", facecolor=color))
        
    ax.set_xlim([x_reshape[0,0],x_reshape[0,-1]])
    draw_rect(ax)
    
    def nypkt(i):
        field.set_data(V_mag_reshape[i,:,:])
        # particle.set_data(sti[:,i,1], sti[:,i,2])
        t = t_list[i]
        # https://stackoverflow.com/questions/16527930/matplotlib-update-position-of-patches-or-set-xy-for-circles
        for part in partiklar:
            part.circle.center = np.interp(t,part.sti[:,0],part.sti[:,1]), np.interp(t,part.sti[:,0],part.sti[:,2])
            # part.circle.center = part.sti[i,1], part.sti[i,2]
            part.annotation.xy = (np.interp(t,part.sti[:,0],part.sti[:,1]), np.interp(t,part.sti[:,0],part.sti[:,2]))
        
        return 1 #field,particle
    
    print("Skal byrja på filmen")
    #ax.axis('equal')
    # ani = animation.FuncAnimation(fig, nypkt, frames=np.arange(1,steps),interval=50)
    ani = animation.FuncAnimation(fig, nypkt, frames=np.arange(0,steps),interval=int(1000/fps))
    plt.show()
    print("ferdig med animasjon, skal lagra")
    
    ani.save(utfilnamn)
    plt.close()

# def createParticle(diameter, init_position, density=2.65e-6):
#     volume = diameter**3 * pi * 1/6
#     mass = volume*density
#     radius = diameter/2
#     return {'diameter': diameter, 'init_position':init_position}

class Particle:
    #Lag ein tabell med tidspunkt og posisjon for kvar einskild partikkel.
    def __init__(self, diameter, init_position, density=2.65e-6 ):
        self.diameter= diameter
        self.init_position = init_position
        self.density = density
        self.volume = self.diameter**3 * pi * 1/6
        self.mass = self.volume * self.density
        self.radius = self.diameter/2

def particle_copy(pa):
    return Particle(pa.diameter, pa.init_position, pa.density)
        
def f(t, x, particle, tri, linear=True, lift=False, addedmass=True):
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
    U : Tuple
        Ein tuple av dei to fartsvektor-arrayane.

    Returns
    -------
    tuple
            Ein tuple med [dx/dt, du/dt]

    """
    
    g = np.array([0, 9.81e3]) # mm/s^2 = 9.81 m/s^2
    nu = 1 # 1 mm^2/s = 1e-6 m^2/s
    rho = 1e-6  # kg/mm^3 = 1000 kg/m^3 
    
    dx_dt = x[2:]
    # vel = np.array([100,0]) - dx_dt # relativ snøggleik
    # U_f = np.array(get_u(t,np.array([x[0],x[1]]),tri, linear))
    
    U_f, dudt_material, U_top_bottom = get_u(t, x, particle.radius, tri, linear=linear, lift=lift, addedmass=addedmass)
    
    # if (np.isnan(U_f[2])):
    #     U_f[2] = 0
    #     # U_f[2] = dudt_mean[0][ckdtre.query(np.concatenate(([t], np.array([x[0],x[1]]))))[1]]
    # if (np.isnan(U_f[3])):
    #     U_f[3] = 0
    #     # U_f[3] = dudt_mean[1][ckdtre.query(np.concatenate(([t], np.array([x[0],x[1]]))))[1]]
        
        
    vel = U_f - dx_dt # relativ snøggleik
    # vel_ang = atan2(vel[1], vel[0])
    
    
    Re = hypot(vel[0],vel[1]) * particle.diameter / nu 
    
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
    rho_self_density = rho / particle.density
    
    drag_component =  3/4 * cd / particle.diameter * rho_self_density * abs(vel)*vel
    gravity_component = (rho_self_density - 1) * g

    added_mass_component = 0.5 * rho_self_density * dudt_material
    
    lift_component = 3/4 * 0.5 / particle.diameter * rho_self_density * (U_top_bottom[:,0]*U_top_bottom[:,0] - U_top_bottom[:,1]*U_top_bottom[:,1])
    
    divisor = 1 + 0.5 * rho_self_density * addedmass
    # divisoren trengst for akselerasjonen av partikkel kjem fram i added 
    # mass på høgre sida, så den må bli flytta over til venstre og delt på resten.
    
    # print("drag_component =",drag_component,", gravity_component = ",gravity_component)        

    du_dt = (drag_component + gravity_component + added_mass_component + lift_component ) / divisor
    
    # if (np.any(np.isnan(du_dt))):
    #     print("her er nan! og t, x og dudt er dette:", t,x, du_dt)
    
    return np.concatenate((dx_dt,du_dt))
    
def checkCollision(particle, data, rib):
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
            
            if (dis > particle.radius):
                return (False, collisionInfo, rib) # må vel endra til (bool, depth, normal, start)
            
            normal = norm(v1)
            
            radiusVec = normal*particle.radius*(-1)
            
            # sender informasjon til collisioninfo:                    
            collisionInfo = (particle.radius - dis, normal, position + radiusVec)
            
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
        
                if (dis > particle.radius):
                    return (False, collisionInfo, rib)
                
                normal = norm(v1)
                radiusVec = normal * particle.radius*(-1)
                
                collisionInfo = (particle.radius - dis, normal, position + radiusVec)
            else:
                #//the center of circle is in face region of face[nearestEdge]
                if (bestDistance < particle.radius):
                    radiusVec = normals[nearestEdge] * particle.radius
                    collisionInfo = (particle.radius - bestDistance, normals[nearestEdge], position - radiusVec)
                else:
                    return (False, collisionInfo, rib)
    else:
        #     //the center of circle is inside of rectangle
        radiusVec = normals[nearestEdge] * particle.radius
        collisionInfo = (particle.radius - bestDistance, normals[nearestEdge], position - radiusVec, -1)
        
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

@ray.remote
def lag_sti(ribs, t_span, solver_args, fps=20, wraparound = False):

    # stien må innehalda posisjon, fart og tid.
    sti = []
    sti_komplett = []

    particle = solver_args['args'][0]
    
    step_old = np.concatenate(([t_span[0]], particle.init_position))
    # Step_old og step_new er ein array med [t, x, y, u, v]. 
    
    sti.append(step_old)
    sti_komplett.append(step_old)
    
    # finn neste steg med rk_2 og standard tidssteg.
    # sjekk kollisjon. Dersom ikkje kollisjon, bruk resultat frå rk_2 og gå til neste steg
    # Dersom kollisjon: halver tidssteget, sjekk kollisjon. Dersom ikkje kollisjon

    t = t_span[0]
    t_max = t_span[1]
    t_main = t
    dt_main = 1/fps
    dt = dt_main
    eps = 0.001
    rest = 1
    
    while (t < t_span[1]):
        
        step_new = rk_3(f, (t,t+dt), step_old[1:], solver_args)

        
        if (step_new[1] > 67 and wraparound):
            step_new[1] -= 100
        elif(step_new[1] > 95):
            break
            
        
        for rib in ribs:
            collision_info = checkCollision(particle, step_new[1:], rib)
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
                    # sti.append(step_new)
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
            
    if (len(sti) < t_max*fps):
        sti = np.pad(sti, ((0,t_max*fps - len(sti)),(0,0)),'edge')
        sti[int(t_main*fps):,0] = np.arange(t_main, t_max, 1/fps)
    
    if (len(sti) > t_max*fps):
        sti = sti[0:t_max*fps]
        
    return np.array(sti)


class Rib:
    def __init__(self, coords):
        self.vertices = sortClockwise(np.array(coords))
        
        self.normals = [np.cross(self.vertices[1]-self.vertices[0],np.array([0,0,-1]))[:2],
                        np.cross(self.vertices[2]-self.vertices[1],np.array([0,0,-1]))[:2], 
                        np.cross(self.vertices[3]-self.vertices[2], np.array([0,0,-1]))[:2],
                        np.cross(self.vertices[0]-self.vertices[3],np.array([0,0,-1]))[:2] ]
        
                        # Må sjekka om punkta skal gå mot eller med klokka. 
                        # Nett no går dei MED klokka. Normals skal peika UT.
        
        
    # def __init__(self, origin, width, height):
    #     # Bør kanskje ha informasjon om elastisiteten ved kollisjonar òg?
    #     self.origin = np.array(origin)
    #     self.width = width
    #     self.height = height


#Typisk ribbe:
# a = [[897,1011], [895,926  ], [353,1014 ], [351,929  ]]
# pkt_ny = sortClockwise(pkt)   

#%%
# Her er ein funksjon for fritt fall av ein 1 mm partikkel i vatn.
# d u_p/dt = u_p(t,y) der y er vertikal fart. Altså berre modellert drag og gravitasjon.
def u_p(t,y):
    if(y==0):
        cd=1e4
    else:
        cd=24/abs(0-y)*(1+0.15*abs(0-y)**0.687)
        
    return 3/4*(0-y)*abs(0-y)*cd*1/2.65-9810*1.65/2.65
 
