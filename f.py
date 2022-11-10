import numpy as np
from hjelpefunksjonar import norm
from constants import g, ρ, ν
g = np.array([[0], [g]]) # mm/s^2 = 9.81 m/s^2
# import ray

def f(t, x, particle, tri, ribs, skalering, get_u, separated = False):
    """
    Sjølve differensiallikninga med t som x, og x som y (jf. Kreyszig)
    Så x er ein vektor med to element, nemleg x[0] = posisjon og x[1] = fart.
    Men for at solve_ivp skal fungera, må x vera 1-dimensjonal. Altså. 
    x= [x,y,u,v], ikkje x = [[x,y], [u,v]].
    f (t, x, y) = [dx/dt, du/dt]

    Parameters
    ----------
    t : double
        Tidspunktet for funksjonen. Ikkje frame, men tid. Må vera slik sidan tida vert multiplisert med posisjonen ute i solve_ivp-funksjonen.
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
    number_of_vectors = x.shape[1]
    try:
        µ = collision['rib'].µ # friksjonskoeffisenten
    except KeyError:
        µ = 0.5
    
    dxdt = x[2:]

    U_f, dudt_material, U_top_bottom = get_u(t, x, particle, tri, ribs, collision= collision, skalering=skalering)
    
    vel = U_f - dxdt # relativ snøggleik
    # vel_ang = atan2(vel[1], vel[0])
    assert len(vel) == 2
    Re = np.hypot(vel[0],vel[1]) * particle.diameter / ν
    
    # if (Re<1000):
    try:
        # with np.errstate(divide='raise'):
            cd = ( (32 / (Re+.001))**(1/1.5) + 1)**1.5
            # Cheng (1997) skildrar Cd for kantete og runde steinar. Dette 
            # er kanskje den viktigaste grunnen til at eg bør gjera dette?
            # Ferguson og Church (2004) gjev nokre liknande bidrag, men viser til Cheng.
    except (ZeroDivisionError, FloatingPointError):
            cd = 2e4
    # else:
    #     cd = 0.44
    
    # print("Re = ", Re," cd= ", cd)
    rho_self_density = ρ / particle.density
    
    drag_component =  3/4 * cd / particle.diameter * rho_self_density * np.linalg.norm(vel,axis=0)*vel
    gravity_component = (rho_self_density - 1) * g

    added_mass_component = 0.5 * rho_self_density * dudt_material 

    pressure_component = rho_self_density * dudt_material
    
    if np.all(dudt_material == 0.0 ):
        addedmass = False

    # U_top_bottom = U_top_bottom[ :,np.flipud(np.argsort(np.squeeze(np.linalg.norm(U_top_bottom, axis=0)))) ]

    # lift_component = np.array([[0, -1],[1, 0]]) @ (np.einsum('ij,ij->j', 3/4 * 0.2 / particle.diameter * rho_self_density * -np.diff(np.square(U_top_bottom- x[2:,None]), axis=1).reshape(2,number_of_vectors), norm(drag_component)) * norm(drag_component))
    if np.all(U_top_bottom == 0.0):
        lift_component = np.zeros((2,number_of_vectors))
    else:
        rotation_matrix_sign = np.sign(np.diff(np.linalg.norm(U_top_bottom- x[2:,None],axis=0),axis=0)).item(0) # Rotasjonsmatrisa skal vera [[0,-1],[1,0]] om U_top > U_bottom, og [[0,1],[-1,0]] om U_top < U_bottom. rotation_matrix_sign > 0 om U_top < U_bottom.

        lift_component = 3/4 * 0.2 / particle.diameter * rho_self_density * -np.diff(np.square(np.hypot((U_top_bottom- x[2:,None])[0], (U_top_bottom- x[2:,None])[1])),axis=0) * (np.array([[0, rotation_matrix_sign],[-rotation_matrix_sign, 0]]) @ norm(drag_component) )
    
    divisor = 1 + 0.5 * rho_self_density * addedmass
    # divisoren trengst for akselerasjonen av partikkel kjem fram i added 
    # mass på høgre sida, så den må bli flytta over til venstre og delt på resten.
    
    dudt = (drag_component + gravity_component + added_mass_component + pressure_component + lift_component ) / divisor

    # print(f"{t};{x};{drag_component};{gravity_component};{added_mass_component - 0.5 * rho_self_density * dudt};{lift_component};{pressure_component};{dudt}",)

    try:
        if (collision['is_resting_contact'] and particle.resting):# and np.dot(collision['rib_normal'],dudt) <= 0: #Kan ikkje sjekka om partikkelen skal ut frå flata midt i berekninga. Må ha ein event til alt slikt.
            #akselerasjonen, dudt
            normal = collision['rib_normal']
            dudt_n = normal * np.dot(normal.T,dudt) # projeksjon av dudt på normalvektoren
            dxdt_n = normal * np.dot(normal.T,dxdt)
            
            dudt_t = dudt - dudt_n
            dxdt_t = dxdt - dxdt_n
            
            if np.any(np.hypot(dxdt_t[0],dxdt_t[1]) == 0): # < vel_limit -- Tilfellet kvilefriksjon:
                dudt_friction = -norm(dudt_t) * np.hypot(dudt_n[0],dudt_n[1]) * µ
                if np.any(np.hypot(dudt_t[0],dudt_t[1]) > np.hypot(dudt_friction[0],dudt_friction[1])):
                    dudt = dudt_t + dudt_friction # Må ordna: Viss normalkomponenten peikar oppover, ikkje null ut den då.
                else:
                    dudt = np.zeros_like(dudt_t)
                    # dxdt = np.zeros(2)
            else: # Tilfellet glidefriksjon:
                dudt_friction = -norm(dxdt_t)* np.linalg.norm(dudt_n,axis=0) * µ #np.hypot(dudt_n[0],dudt_n[1])
                dudt = dudt_t + dudt_friction

    except KeyError:
        pass
    
    if not separated:
        return np.concatenate((dxdt,dudt))
    else:
        return dict(drag = drag_component, gravity = gravity_component, added_mass = added_mass_component - 0.5 * rho_self_density * dudt, pressure = pressure_component, lift = lift_component, dudt = dudt, dxdt=dxdt)
