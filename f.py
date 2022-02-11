import numpy as np
from math import hypot
from hjelpefunksjonar import norm

from get_u import get_u_simple as get_u

def f(t, x, particle, ribs):
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
 
    addedmass = particle.addedmass
    collision = particle.collision 
    try:
        µ = collision['rib'].µ # friksjonskoeffisenten
    except KeyError:
        µ = 0.5
    
    dxdt = x[2:]

    U_f, dudt_material, U_top_bottom = get_u(t, x, particle, collision= collision)
    
    vel = U_f - dxdt # relativ snøggleik
    # vel_ang = atan2(vel[1], vel[0])
    
    Re = hypot(vel[0],vel[1]) * particle.diameter / nu 
    
    if (Re<1000):
        try:
            # with np.errstate(divide='raise'):
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

    pressure_component = rho_self_density * dudt_material
    
    if dudt_material[0] == 0.0 and dudt_material[1] == 0.0:
        addedmass = False

    lift_component = 3/4 * 0.5 / particle.diameter * rho_self_density * (U_top_bottom[:,0]*U_top_bottom[:,0] - U_top_bottom[:,1]*U_top_bottom[:,1]) * norm(drag_component) @ np.array([[0, 1],[-1, 0]])
    
    divisor = 1 + 0.5 * rho_self_density * addedmass
    # divisoren trengst for akselerasjonen av partikkel kjem fram i added 
    # mass på høgre sida, så den må bli flytta over til venstre og delt på resten.
    
    # print("drag_component =",drag_component,", gravity_component = ",gravity_component)        
    dudt = (drag_component + gravity_component + added_mass_component + pressure_component + lift_component ) / divisor

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

