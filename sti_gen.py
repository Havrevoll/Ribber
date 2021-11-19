# -*- coding: utf-8 -*-
'''køyr funksjonar som plottingar(fil['vassføringar'])'''

import matplotlib
matplotlib.use("Agg")
from lag_video import sti_animasjon
from kornfordeling import get_PSD_part
import numpy as np
# from scipy import interpolate
from scipy.integrate import solve_ivp  # https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html#r179348322575-1
from hjelpefunksjonar import norm, sortClockwise, finn_fil

import ray

import datetime
from math import floor, pi, hypot
import random
from scipy.sparse.construct import rand #, atan2

# fil = h5py.File("D:/Tonstad/alle.hdf5", 'a')
# x = np.array(h5py.File(filnamn, 'r')['x']).reshape(127,126)[ranges()]
# vass = fil['vassføringar']


t_max_global = 20
t_min_global = 0
vel_limit = 0.1
eps = 0.01
rest = 0.2 # collision restitution

nullfart = np.zeros(2)

def simulering(tal, rnd_seed, tre,  fps = 20, t_span = (0,59), linear = True,  lift = True, addedmass = True, wrap_max = 50, 
      atol = 1e-1, rtol = 1e-1, method = 'RK23', tre_fil = finn_fil(["../Q40_0_60.pickle", "../Q40_0_10.pickle"]), laga_film = False, verbose=True, collision_correction=True):

    start = datetime.datetime.now()


    ray.init() 
    tre_plasma = ray.put(tre)
    
    
    random.seed(rnd_seed)
    diameters = get_PSD_part(tal, rnd_seed=rnd_seed)
    # diameters[0]=0.34

    # diameters = [0.08]
    particle_list = [Particle(d, [-35.81,random.uniform(0,90), 0, 0], random.uniform(0,50)) for d in diameters]
    # particle_list = [Particle(d, [-90,43.43268, 0, 0], 3.65) for d in diameters]

    for i, p in enumerate(particle_list):
        p.atol , p.rtol = atol, rtol
        p.method = method
        p.linear, p.lift, p.addedmass = linear, lift, addedmass
        p.init_time = floor(p.init_time *fps)/fps #rettar opp init_tid slik at det blir eit tal som finst i datasettet.
        p.index = i

    print("Har laga tre_objekt, skal putta")

    ribs = [Rib(rib) for rib in tre.ribs]
        
    for pa in particle_list:
        pa.job_id = lag_sti.remote(ribs, t_span, particle=pa, tre=tre_plasma, fps = fps, wrap_max=wrap_max, verbose=verbose, collision_correction=collision_correction)

    fanga = 0

    for pa in particle_list:
        pa.sti = ray.get(pa.job_id)
        if pa.sti[-1][2] < -10:
            fanga += 1

    


    print("Brukte {} s fram til filmlaging.".format(datetime.datetime.now()-start))
    print(f"Av {len(particle_list)} partiklar vart {fanga} fanga, altså {100* fanga/len(particle_list)}%")

    # while (True):
    #     ready_refs, remaining_refs = ray.wait(object_refs, num_returns=1, timeout=None)
    #     if remaining_refs == 0:
    #         break

    if laga_film:
        start_film = datetime.datetime.now()
        film_fil = f"sti_{method}_{len(particle_list)}_{atol:.0e}_noaddedmass_nolift_ray.mp4"
        sti_animasjon(particle_list,t_span=t_span, utfilnamn=film_fil, fps=fps)
        print("Brukte  {} s på å laga film".format(datetime.datetime.now() - start_film))

    ray.shutdown()
    print("Brukte totalt {} s på heile operasjonen".format(datetime.datetime.now() - start))

    return particle_list

def event_check(t, x, particle, tre, ribs):
    for rib in ribs:
        collision = checkCollision(particle, x,rib)
        if collision['is_collision'] and not collision['is_resting_contact']:
            return 0.0
    
    return 1.0

event_check.terminal = True

# @ray.remote
def lag_sti(ribs, t_span, particle, tre, fps=20, wrap_max = 50, verbose=True, collision_correction=True):
    # stien må innehalda posisjon, fart og tid.
    sti = []
    sti_komplett = []
    # print(type(tre))
    # tre = ray.get(tre)
    print(f"Partikkel nr. {particle.index} byrja, starta på {particle.init_position}, og {particle.init_time}")
    
    solver_args = dict(atol = particle.atol, rtol= particle.rtol, method=particle.method, args = (particle, tre, ribs), events = event_check)
 
    step_old = np.concatenate(([particle.init_time], particle.init_position))
    # Step_old og step_new er ein array med [t, x, y, u, v]. 
    
    sti.append(step_old)
    sti_komplett.append(step_old)
    
    # finn neste steg med rk_2 og standard tidssteg.
    # sjekk kollisjon. Dersom ikkje kollisjon, bruk resultat frå rk_2 og gå til neste steg
    # Dersom kollisjon: halver tidssteget, sjekk kollisjon. Dersom ikkje kollisjon

    t = particle.init_time
    t_max = t_span[1]
    t_main = t
    dt_main = 1/fps
    dt = dt_main
    wrap_counter = 0

    left_edge = ribs[0].get_rib_middle()
    right_edge = ribs[1].get_rib_middle()
    
    starttid = datetime.datetime.now()

    while True:
        # style = random.randint(0,4)
        text_color = random.randint(30,38)+ 60*random.randint(0,1)
        background = random.randint(40,48) 
        combined = (text_color, background)
        bad = [(30,40),(30,48),(31,41), (32,42), (33,43), (34,44),(35,44),(35,45),(36,46),(37,47),(35,41),(36,42),(37,43),(31,45),(32,46),(33,47),(96,41),(97,43)]
        if (combined not in bad):
            break
    del bad, combined
    status_col = f"{str()};{str(text_color)};{str(background)}"
    

    while (t < t_max):
        
        step_new, collision = rk_3(f, (t,t+dt), step_old[1:], solver_args)

        if verbose:
            status_msg = "{tid} pa nr {i}, {d:.2f} mm startpos. [{pos[0]:.3f}, {pos[1]:.3f}] ferdig med t={t:.4f}, pos=[{x:.2f},{y:.2f}] U=[{u:.2f},{v:.2f}]".format(tid=(datetime.datetime.now()-starttid), i=particle.index, d=particle.diameter, pos=particle.init_position[:2], t=step_new[0], x=step_new[1], y=step_new[2], u=step_new[3], v=step_new[4])
            status_wrap = f"\x1b[{status_col}m {status_msg} \x1b[0m"
            print(status_wrap)

        #.strftime('%X.%f')
        if (step_new[1] > right_edge and wrap_counter <= wrap_max):
            step_new[1] = left_edge
            wrap_counter += 1
        elif(step_new[1] > right_edge):
            break
            
            
        if (collision):
            for rib in ribs:
                collision_info = checkCollision(particle, step_new[1:], rib)
                if (collision_info['is_collision']):
                    break

            if not collision_correction:
                break

            assert collision_info['collision_depth'] < eps
            if (collision_info['collision_depth'] < eps):
                if verbose:
                    print("kolliderte")
                #Gjer alt som skal til for å endra retningen på partikkelen
                n = collision_info['rib_normal']
                v = step_new[3:]
                v_rel = collision_info['relative_velocity'] 
                # v_rel er relativ fart i normalkomponentretning, jf. formel 8-3 i baraff ("notesg.pdf")
                v_new = v - (rest + 1) * v_rel * n
                step_old = np.copy(step_new)
                step_old[3:] = v_new
                
                #Fullfør rørsla fram til hovud-steget, vonleg med rett retning 
                # og fart, så ein kjem inn i rett framerate igjen.
                # dt = t_main + dt_main - t
                if step_old[0] != t+dt:
                    dt = t + dt - step_old[0]
                else:
                    sti.append(step_old)
                    t_main = step_old[0]
                t = step_old[0]

                # step_new = rk_3(part.f, (t, t_main+dt_main), step_old[1:])
                
                # if (np.hypot(v[0], v[1]) < 0.01):
                #     sti_komplett.append(step_new)
                #     # sti.append(step_new)
                #     break
                
                # t = t_main + dt_main
                
            else:
                dt = dt/2
                if verbose:
                    print("må finjustera kollisjon, skulle ikkje vore her, HJELP!!!!!")
                continue
        else:
            sti_komplett.append(step_new)
            step_old = step_new
            
            t = round((t + dt) * 10000) / 10000

        if (round(step_new[0]*10000) % round(dt_main*10000) == 0):
            sti.append(step_new)
            t_main = step_new[0]
            t = t_main
            dt = dt_main
        
    # if (len(sti) < t_max*fps):
    #     sti = np.pad(sti, ((0,t_max*fps - len(sti)),(0,0)),'edge')
    #     sti[int(t_main*fps):,0] = np.arange(t_main, t_max, 1/fps)
    
    # if (len(sti) > t_max*fps):
    #     sti = sti[0:t_max*fps]

    print(f"Partikkel nr. {particle.index} brukte {datetime.datetime.now()-starttid}")    
    return np.array(sti)


def rk_3 (f, t, y0, solver_args):
    assert t[1] - t[0] > 0
    resultat = solve_ivp(f, t, y0, dense_output=True,   **solver_args) # t_eval = [t[1]],
    # har teke ut max_ste=0.02, for det vart aldri aktuelt, ser det ut til.  method=solver_args['method'], args=solver_args['args'],
    assert resultat.success == True

    if (resultat.message == "A termination event occurred."):
        return np.concatenate((resultat.t_events[0], resultat.y_events[0][0])), True
    else:
        return np.concatenate(([resultat.t[-1]], resultat.y[:,-1])), False


# def createParticle(diameter, init_position, density=2.65e-6):
#     volume = diameter**3 * pi * 1/6
#     mass = volume*density
#     radius = diameter/2
#     return {'diameter': diameter, 'init_position':init_position}

def f(t, x, particle, tri, ribs):
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
    

 
    mu = 0.5 # friksjonskoeffisenten
    addedmass = particle.addedmass
    
    dxdt = x[2:]
    # vel = np.array([100,0]) - dxdt # relativ snøggleik
    # U_f = np.array(get_u(t,np.array([x[0],x[1]]),tri, linear))

    
    for rib in ribs:
        collision = checkCollision(particle, x, rib)
                    
        if collision['is_collision'] or collision['is_resting_contact']:
            # print("Kollisjonsdjup er {} og det er {}over grenseverdien".format(collision['collision_depth'], "ikkje " if collision['collision_depth'] < eps else ""))

            # n = collision['rib_normal']
            # v = x[2:]
            # v_rel = collision['relative_velocity'] 
            # # v_rel er relativ fart i normalkomponentretning, jf. formel 8-3 i baraff ("notesg.pdf")
            # v_new = v - (rest + 1) * v_rel * n
            # x[2:] = v_new
        
            break

    U_f, dudt_material, U_top_bottom = get_u(t, x, particle, tri, collision= collision)
    
        # if (np.isnan(U_f[2])):
    #     U_f[2] = 0
    #     # U_f[2] = dudt_mean[0][ckdtre.query(np.concatenate(([t], np.array([x[0],x[1]]))))[1]]
    # if (np.isnan(U_f[3])):
    #     U_f[3] = 0
    #     # U_f[3] = dudt_mean[1][ckdtre.query(np.concatenate(([t], np.array([x[0],x[1]]))))[1]]
        
        
    vel = U_f - dxdt # relativ snøggleik
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
    dudt = (drag_component + gravity_component + added_mass_component + lift_component ) / divisor

    try:
        if collision['is_resting_contact']:
            #akselerasjonen, dudt
            dudt_n = collision['rib_normal'] * np.dot(collision['rib_normal'],dudt) # projeksjon av dudt på normalvektoren
            dxdt_n = collision['rib_normal'] * np.dot(collision['rib_normal'],dxdt)
            
            dudt_t = dudt - dudt_n
            dxdt_t = dxdt - dxdt_n

            if hypot(dxdt_t[0],dxdt_t[1]) < vel_limit:
                dudt_friction = -norm(dudt_t)*hypot(dudt_n[0],dudt_n[1])*mu
                if (hypot(dudt_t[0],dudt_t[1]) > hypot(dudt_friction[0],dudt_friction[1])):
                    dudt = dudt_t + dudt_friction #
                else:
                    dudt = np.zeros(2)
            else:
                dudt_friction = -norm(dxdt_t)*hypot(dudt_n[0],dudt_n[1])*mu
                dudt = dudt_t + dudt_friction
                dxdt = dxdt_t
                
    except KeyError:
        pass

    return np.concatenate((dxdt,dudt))

# Så dette er funksjonen som skal analyserast av runge-kutta-operasjonen. Må ha t som fyrste og y som andre parameter.
# @jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
def get_u(t, x_inn, particle, tre_samla, collision):
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
    Tuple
        DESCRIPTION.

    '''    
    radius = particle.radius
    lift, addedmass, linear = particle.lift, particle.addedmass, particle.linear

    tx = np.concatenate(([t], x_inn[:2]))
    U_p = x_inn[2:]
        
    dt, dx, dy = 0.01, 0.1, 0.1
    
    # tre_samla = ray.get(tre_plasma)

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
            addedmass = False
            linear = False
            lift = False

            while True:
                try:
                    U_kd[:,kdtre.query(x[0])[1]]
                    break
                except IndexError:
                    x[np.abs(x)>1e100] /= 1e10
                    
                
            U_f = U_kd[:,kdtre.query(x[0])[1]] #, nullfart, np.zeros((2,2))
            # Gjer added mass og lyftekrafta lik null, sidan den ikkje er viktig her.
        else:  
            vertices = np.take(tri.simplices, simplex, axis=0)
            temp = np.take(tri.transform, simplex, axis=0)
                    
            delta = x - temp[:,d]
            bary = np.einsum('njk,nk->nj', temp[:,:d, :], delta)
            wts = np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))
            
            # U_f = np.einsum('nij,ni->nj', np.take(np.column_stack(U_del),vertices,axis=0),wts)
            # U_f = np.einsum('ijn,ij->n', np.take(U_del, vertices, axis=0), wts)
            U_f = np.einsum('jni,ni->jn', np.take(U_del,vertices,axis=1),wts)
    else:
        U_f = U_kd[:,kdtre.query(x[0])[1]]#, nullfart, np.zeros((2,2))
        addedmass = False
        linear = False
        lift = False
        # return np.einsum('j,j->', np.take(U_del[0], vertices), wts), np.einsum('j,j->', np.take(U_del[1], vertices), wts),  np.einsum('j,j->', np.take(U_del[2], vertices), wts),  np.einsum('j,j->', np.take(U_del[3], vertices), wts)
        # return U[0][kd_index], U[1][kd_index], U[2][kd_index], U[3][kd_index]

    try:
        if (collision['is_resting_contact']):
            U_f = U_f - np.array([collision['rib_normal']]).T * np.dot(collision['rib_normal'],U_f) # projeksjon av dudt på normalvektoren
    except KeyError:
        pass
                
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
            break
        
            # if (np.any(part_wts < -0.02)):
            #     part_simplex = tri.find_simplex(part)
            #     if np.any(part.simplex == -1):
            #         break
            #     part_vertices = np.take(tri.simplices, part_simplex, axis=0)
            #     part_temp = np.take(tri.transform, part_simplex, axis=0)
            #     get_u.simplex_prob += 1
            #     if get_u.simplex_prob > 20:
            #         print("Går i loop i part_simplex og der!")
            # else:
            #     get_u.simplex_prob = 0
            #     break
                
        U_top_bottom = np.einsum('jni,ni->jn', np.take(U_del, part_vertices, axis=1), part_wts)
    else:
        U_top_bottom = np.zeros((2,2))
    
    if (linear):
        U_f = U_f[:,0]

    return (U_f, dudt_material, U_top_bottom)

  
# cd = interpolate.interp1d(np.array([0.001,0.01,0.1,1,10,20,40,60,80,100,200,400,600,800,1000,2000,4000,6000,8000,10000,100000]), np.array([2.70E+04,2.40E+03,2.50E+02,2.70E+01,4.40E+00,2.80E+00,1.80E+00,1.45E+00,1.25E+00,1.12E+00,8.00E-01,6.20E-01,5.50E-01,5.00E-01,4.70E-01,4.20E-01,4.10E-01,4.15E-01,4.30E-01,4.38E-01,5.40E-01,]))
    
get_u.counter = 0
get_u.utanfor = 0
get_u.simplex_prob = 0
    


# def rk_2(f, L, y0, h, tri, U):
#     ''' Heimelaga Runge-Kutta-metode '''
#     t0, t1 = L
#     N=int((t1-t0)/h)

#     t=[0]*N # initialize lists
#     y=[0]*N # initialize lists
    
#     t[0] = t0
#     y[0] = y0
    
#     for n in range(0, N-1):
#         #print(n,t[n], y[n], f(t[n],y[n]))
#         k1 = h*f(t[n], y[n], tri, U)
#         k2 = h*f(t[n] + 0.5 * h, y[n] + 0.5 * k1, tri, U)
#         k3 = h*f(t[n] + 0.5 * h, y[n] + 0.5 * k2, tri, U)
#         k4 = h*f(t[n] + h, y[n] + k3, tri, U)
        
#         if (np.isnan(k4+k3+k2+k1).any()):
#             #print(k1,k2,k3,k4)
#             return t,y
        
#         t[n+1] = t[n] + h
#         y[n+1] = y[n] + 1/6 * (k1 + 2*k2 + 2*k3 + k4)
        
#     return t, y

class Particle:
    #Lag ein tabell med tidspunkt og posisjon for kvar einskild partikkel.
    def __init__(self, diameter, init_position, init_time=0, density=2.65e-6 ):
        self.diameter= diameter
        self.init_position = init_position
        self.init_time = init_time
        self.density = density
        self.volume = self.diameter**3 * pi * 1/6
        self.mass = self.volume * self.density
        self.radius = self.diameter/2
        self.index = 0
        self.atol = 1e-6
        self.rtol = 1e-3
        self.method = 'RK45'
        self.linear = True
        self.lift = True
        self.addedmass = True


def particle_copy(pa):
    return Particle(pa.diameter, pa.init_position, pa.density)


    
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
    
    # collisionInfo = (-1,np.array([-1,-1]), np.array([-1,-1]), -1)
    # collisionInfo = {}
    
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
                return {'is_collision':False, 'is_resting_contact':False}#, 'rib':rib}
                # (False, collisionInfo, rib) # må vel endra til (bool, depth, normal, start)
            
            normal = norm(v1)
            
            radiusVec = normal*particle.radius*(-1)
            
            # sender informasjon til collisioninfo:                    
            collision_info = dict(collision_depth=particle.radius - dis, rib_normal=normal, particle_collision_point = position + radiusVec)
            
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
                    return {'is_collision':False, 'collision_depth': 0, 'is_resting_contact':False}#, 'rib':rib}

                normal = norm(v1)
                radiusVec = normal * particle.radius*(-1)
                
                collision_info = dict(collision_depth=particle.radius - dis, rib_normal = normal, particle_collision_point = position + radiusVec)
            else:
                #//the center of circle is in face region of face[nearestEdge]
                if (bestDistance < particle.radius):
                    radiusVec = normals[nearestEdge] * particle.radius
                    collision_info = dict(collision_depth = particle.radius - bestDistance, rib_normal = normals[nearestEdge], particle_collision_point = position - radiusVec)
                else:
                    return dict(is_collision =  False, collision_depth = 0, is_resting_contact = False)
    else:
        #     //the center of circle is inside of rectangle
        radiusVec = normals[nearestEdge] * particle.radius

        return dict(is_collision = True, is_resting_contact = False, rib = rib, collision_depth = particle.radius - bestDistance, rib_normal = normals[nearestEdge], particle_collision_point = position - radiusVec, inside = True)
        # Måtte laga denne returen så han ikkje byrja å rekna ut relativ fart når partikkelen uansett er midt inne i ribba.

    # Rekna ut relativ fart i retning av normalkomponenten, jamfør Baraff (2001) formel 8-3.
    n = collision_info['rib_normal']
    v = np.array(data[2:])
    v_rel = np.dot(n,v)
    collision_info['relative_velocity'] = v_rel
    collision_info['rib'] = rib

    if (abs(v_rel) < vel_limit):
        collision_info['is_resting_contact'] = True
    else:
        collision_info['is_resting_contact'] = False

    if (v_rel < -vel_limit): # Sjekk om partikkelen er på veg vekk frå veggen.
        collision_info['is_collision'] = True
    else:
        collision_info['is_collision'] = False
    
    return collision_info


class Rib:
    def __init__(self, coords):
        self.vertices = sortClockwise(np.array(coords))
        
        self.normals = [norm(np.cross(self.vertices[1]-self.vertices[0],np.array([0,0,-1]))[:2]),
                        norm(np.cross(self.vertices[2]-self.vertices[1],np.array([0,0,-1]))[:2]), 
                        norm(np.cross(self.vertices[3]-self.vertices[2], np.array([0,0,-1]))[:2]),
                        norm(np.cross(self.vertices[0]-self.vertices[3],np.array([0,0,-1]))[:2]) ]
        
                        # Må sjekka om punkta skal gå mot eller med klokka. 
                        # Nett no går dei MED klokka. Normals skal peika UT.
        
    def get_rib_middle(self):
        centerPoint = np.sum(self.vertices, axis=0)/len(self.vertices)
        return centerPoint[0]

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
# def u_p(t,y):
#     if(y==0):
#         cd=1e4
#     else:
#         cd=24/abs(0-y)*(1+0.15*abs(0-y)**0.687)
        
#     return 3/4*(0-y)*abs(0-y)*cd*1/2.65-9810*1.65/2.65
 
