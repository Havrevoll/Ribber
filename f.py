# import imp
import numpy as np
from math import hypot
from hjelpefunksjonar import norm, t2f
from constants import g, ρ_p, ρ, ν
g = np.array([0, g]) # mm/s^2 = 9.81 m/s^2
# import ray

nullfart = np.zeros(2)

def f(t, x, particle, tri, ribs, fps):
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
    
    addedmass = particle.addedmass
    collision = particle.collision 
    try:
        µ = collision['rib'].µ # friksjonskoeffisenten
    except KeyError:
        µ = 0.5
    
    dxdt = x[2:]

    U_f, dudt_material, U_top_bottom = get_u(t, x, particle, tri, collision= collision, fps=fps)
    
    vel = U_f - dxdt # relativ snøggleik
    # vel_ang = atan2(vel[1], vel[0])
    
    Re = hypot(vel[0],vel[1]) * particle.diameter / ν
    
    # if (Re<1000):
    try:
            # with np.errstate(divide='raise'):
                # cd = 24 / Re * (1+0.15*Re**0.687)
            cd = ( (32 / Re)**(1/1.5) + 1)**1.5
            # Cheng (1997) skildrar Cd for kantete og runde steinar. Dette 
            # er kanskje den viktigaste grunnen til at eg bør gjera dette?
            # Ferguson og Church (2004) gjev nokre liknande bidrag, men viser til Cheng.
    except ZeroDivisionError:
            cd = 2e4
    # else:
    #     cd = 0.44
    
    # print("Re = ", Re," cd= ", cd)
    rho_self_density = ρ / particle.density
    
    drag_component =  3/4 * cd / particle.diameter * rho_self_density * abs(vel)*vel
    gravity_component = (rho_self_density - 1) * g

    added_mass_component = 0.5 * rho_self_density * dudt_material 

    pressure_component = rho_self_density * dudt_material
    
    if dudt_material[0] == 0.0 and dudt_material[1] == 0.0:
        addedmass = False

    lift_component = 3/4 * 0.5 / particle.diameter * rho_self_density * (U_top_bottom[:,0]*U_top_bottom[:,0] - U_top_bottom[:,1]*U_top_bottom[:,1]) * norm(drag_component) @ np.array([[0, 1],[-1, 0]])
    
    divisor = 1 + 0.5 * rho_self_density * addedmass
    # divisoren trengst for akselerasjonen av partikkel kjem fram i added 
    # mass på høgre sida, så den må bli flytta over til venstre og delt på resten.
    
    dudt = (drag_component + gravity_component + added_mass_component + pressure_component + lift_component ) / divisor

    # print(f"{t};{x};{drag_component};{gravity_component};{added_mass_component - 0.5 * rho_self_density * dudt};{lift_component};{pressure_component};{dudt}",)

    try:
        if (collision['is_resting_contact'] and particle.resting):# and np.dot(collision['rib_normal'],dudt) <= 0: #Kan ikkje sjekka om partikkelen skal ut frå flata midt i berekninga. Må ha ein event til alt slikt.
            #akselerasjonen, dudt
            dudt_n = collision['rib_normal'] * np.dot(collision['rib_normal'],dudt) # projeksjon av dudt på normalvektoren
            dxdt_n = collision['rib_normal'] * np.dot(collision['rib_normal'],dxdt)
            
            dudt_t = dudt - dudt_n
            dxdt_t = dxdt - dxdt_n
            
            if hypot(dxdt_t[0],dxdt_t[1]) == 0: # < vel_limit -- Tilfellet kvilefriksjon:
                dudt_friction = -norm(dudt_t) * hypot(dudt_n[0],dudt_n[1]) * µ
                if (hypot(dudt_t[0],dudt_t[1]) > hypot(dudt_friction[0],dudt_friction[1])):
                    dudt = dudt_t + dudt_friction # Må ordna: Viss normalkomponenten peikar oppover, ikkje null ut den då.
                else:
                    dudt = np.zeros(2)
                    # dxdt = np.zeros(2)
            else: # Tilfellet glidefriksjon:
                dudt_friction = -norm(dxdt_t)*hypot(dudt_n[0],dudt_n[1]) * µ
                dudt = dudt_t + dudt_friction

    except KeyError:
        pass

    return np.concatenate((dxdt,dudt))

# Så dette er funksjonen som skal analyserast av runge-kutta-operasjonen. Må ha t som fyrste og y som andre parameter.
# @jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
def get_u(t, x_inn, particle, tre_samla, collision, fps):
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

    tx = np.concatenate(([t2f(t, fps)], x_inn[:2]))
    U_p = x_inn[2:]
        
    # dt, dx, dy = 0.01, 0.1, 0.1
    Δ = 0.01
    
    # U_del = tre_samla.get_U(tx)
    # tri = tre_samla.get_tri(tx)
    tri, U_del = tre_samla.get_tri_og_U(tx)

    x = np.vstack((tx,tx + np.array([Δ,0,0]), tx + np.array([0,Δ,0]), tx +np.array([0,0,Δ])))
        
    kdtre = tre_samla.kdtre
    U_kd = tre_samla.U_kd
    
    get_u.counter +=1
    
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
            if U_f.shape == (2,):
                U_f = U_f - collision['rib_normal'] * np.dot(collision['rib_normal'],U_f) # projeksjon av dudt på normalvektoren
            else:
                U_f = U_f - np.array([collision['rib_normal']]).T * np.dot(collision['rib_normal'],U_f) # projeksjon av dudt på normalvektoren
    except KeyError:
        pass
                
    if (addedmass):
        dUdt = (U_f[:,1] - U_f[:,0]) / Δ # Fyrste verdien er dU/dt og andre er dV/dt
        dUdx = (U_f[:,2] - U_f[:,0]) / Δ # Fyrste verdien er dU/dx og andre er dV/dy
        dUdy = (U_f[:,3] - U_f[:,0]) / Δ # Fyrste verdien er dU/dy og andre er dV/dy

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

        simplex_prob = 0
        while (True):
            part_delta = part - part_temp[:,d] #avstanden til referansepunktet i simpleksen.
            part_bary = np.einsum('njk,nk->nj', part_temp[:,:d, :], part_delta) 
            part_wts = np.hstack((part_bary, 1 - part_bary.sum(axis=1, keepdims=True)))
        
            if (np.any(part_wts < -0.02)):
                part_simplex = tri.find_simplex(part)
                if np.any(part_simplex == -1):
                    break
                part_vertices = np.take(tri.simplices, part_simplex, axis=0)
                part_temp = np.take(tri.transform, part_simplex, axis=0)
                simplex_prob += 1
                if simplex_prob > 20:
                    print("Går i loop i part_simplex og der!")
                    break
            else:
                break
                
        U_top_bottom = np.einsum('jni,ni->jn', np.take(U_del, part_vertices, axis=1), part_wts)
    else:
        U_top_bottom = np.zeros((2,2))
    
    if (linear):
        U_f = U_f[:,0]

    return (U_f, dudt_material, U_top_bottom)

  
get_u.counter = 0
get_u.utanfor = 0
get_u.simplex_prob = 0