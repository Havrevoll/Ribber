import numpy as np
from math import hypot
from hjelpefunksjonar import norm, t2f
from constants import g, ρ_p, ρ, ν
g = np.array([[0], [g]]) # mm/s^2 = 9.81 m/s^2
# import ray

# class RealisticInfoArray(np.ndarray):

#     def __new__(cls, input_array, delkrefter=None):
#         # Input array is an already formed ndarray instance
#         # We first cast to be our class type
#         obj = np.asarray(input_array).view(cls)
#         # add the new attribute to the created instance
#         obj.delkrefter = delkrefter
#         # Finally, we must return the newly created object:
#         return obj

#     def __array_finalize__(self, obj):
#         # see InfoArray.__array_finalize__ for comments
#         if obj is None: return
#         self.delkrefter = getattr(obj, 'info', None)

def f(t, x, particle, tri, ribs, skalering):
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
    number_of_vectors = x.shape[1]
    try:
        µ = collision['rib'].µ # friksjonskoeffisenten
    except KeyError:
        µ = 0.5
    
    dxdt = x[2:]

    U_f, dudt_material, U_top_bottom = get_u(t, x, particle, tri, collision= collision, skalering=skalering)
    
    vel = U_f - dxdt # relativ snøggleik
    # vel_ang = atan2(vel[1], vel[0])
    assert len(vel) == 2
    Re = np.hypot(vel[0],vel[1]) * particle.diameter / ν
    
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
    
    if np.all(dudt_material == 0.0 ):
        addedmass = False

    lift_component = np.array([[0, -1],[1, 0]]) @ ( 3/4 * 0.5 / particle.diameter * rho_self_density * np.diff(np.square(U_top_bottom), axis=1).reshape(2,number_of_vectors) * norm(drag_component) )
    
    divisor = 1 + 0.5 * rho_self_density * addedmass
    # divisoren trengst for akselerasjonen av partikkel kjem fram i added 
    # mass på høgre sida, så den må bli flytta over til venstre og delt på resten.
    
    dudt = (drag_component + gravity_component + added_mass_component + pressure_component + lift_component ) / divisor

    # print(f"{t};{x};{drag_component};{gravity_component};{added_mass_component - 0.5 * rho_self_density * dudt};{lift_component};{pressure_component};{dudt}",)

    try:
        if (collision['is_resting_contact'] and particle.resting):# and np.dot(collision['rib_normal'],dudt) <= 0: #Kan ikkje sjekka om partikkelen skal ut frå flata midt i berekninga. Må ha ein event til alt slikt.
            #akselerasjonen, dudt
            dudt_n = collision['rib_normal'] * np.dot(collision['rib_normal'][:,0],dudt) # projeksjon av dudt på normalvektoren
            dxdt_n = collision['rib_normal'] * np.dot(collision['rib_normal'][:,0],dxdt)
            
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
def get_u(t, x_inn, particle, tre_samla, collision, skalering):
    '''
    get_u skal i praksis vera ein funksjon: [t,x,y]→ ( [u,v], [dudt_material,dvdt_material], [[u_top, u_bottom],[v_top, v_bottom]] ) Så når x_inn er ein vektor med fleire koordinatar: 
    tx = np.array([f₀, f₁, f₂, f₃],
                  [x₀, x₁, x₂, x₃], 
                  [y₀, y₁, y₂, y₃]])
    så må U_f = np.array([[u₀, u₁, u₂, u₃], 
                          [v₀, v₁, v₂, v₃]])
    https://stackoverflow.com/questions/20915502/speedup-scipy-griddata-for-multiple-interpolations-between-two-irregular-grids    

    Parameters
    ----------
    t : float
        tida i tid, ikkje frames (vert gjort om inne her)
    x_inn : Array of float 64
        heile arrayen med posisjon og fart: [x,y,u,v] presentert i ein vertikal vektor med ein eller fleire vektorar bortover.
    particle : Particle.particle
        Den gjeldande partikkelen.
    tre_samla : datagenerering.tre_objekt
        Eit tre med ein dict med delaunay-trea (ein for kvart tidssteg) og tilhøyrande fartsdata, eit kd-tre og tilhøyrande fartsdata, samt ribber.
    collision : dict
        kollisjonsdata
    skalering : int
        skalering, til dømes 1, 20, 40, 100 eller til og med 1000

    Returns
    -------
    Tuple
        U_f.shape = (), dudt_material, U_top_bottom

    '''    
  
    lift, addedmass, linear = particle.lift, particle.addedmass, particle.linear
    frame = t2f(t,skalering)
    number_of_vectors = x_inn.shape[-1] #Dette er talet på vektorar pga vectorized i solve_ivp. Vanlegvis 1, men kan vera fleire, t.d. 4.

    tx = np.concatenate((np.broadcast_to([frame],(1,number_of_vectors)), x_inn[:2]),axis=-2)
        
    # dt, dx, dy = 0.01, 0.1, 0.1
    Δ = 0.01
    
    # U_del = tre_samla.get_U(tx)
    # tri = tre_samla.get_tri(tx)
    tri, U_del = tre_samla.get_tri_og_U(frame)

    x = np.stack((tx,tx + np.asarray([[Δ],[0],[0]]), tx + np.asarray([[0],[Δ],[0]]), tx +np.asarray([[0],[0],[Δ]])))
        
    kdtre = tre_samla.kdtre
    U_kd = tre_samla.U_kd
    
    get_u.counter +=1
    
    if (linear and np.all(tx >= tri.min_bound.reshape(3,1)) and np.all(tx <= tri.max_bound.reshape(3,1))):
        d=3
        # simplex = tri.find_simplex(x)
        # simplex = np.tile(tri.find_simplex(np.swapaxes(tx)), 4)
        # simplex = np.broadcast_to(tri.find_simplex(np.swapaxes(tx, -2,-1)).reshape(tx.shape[-1],1),(4,tx.shape[-1],1))
        simplex = np.broadcast_to(tri.find_simplex(np.swapaxes(tx, -2,-1)),(4,number_of_vectors)) #kan kanskje ta vekk reshape? Sjekk når det er fleire vektorar i ein.
        
        if (np.any(simplex==-1)):
            get_u.utanfor += 1
            addedmass = False
            linear = False
            lift = False
                
            U_f = tre_samla.get_kd_U(tx)
            # Gjer added mass og lyftekrafta lik null, sidan den ikkje er viktig her.
        else:  
            vertices = np.take(tri.simplices, simplex, axis=0)
            temp = np.take(tri.transform, simplex, axis=0)
                    # https://en.wikipedia.org/wiki/Barycentric_coordinate_system#Barycentric_coordinates_on_tetrahedra

            delta = x - temp[:,:,d,:].swapaxes(1,2)
            bary = np.einsum('pnjk,pkn->pjn', temp[:,:,:d,:], delta)
            wts = np.concatenate((bary, 1 - bary.sum(axis=1, keepdims=True)),axis=1)
            
            # U_f = np.einsum('nij,ni->nj', np.take(np.column_stack(U_del),vertices,axis=0),wts)
            # U_f = np.einsum('ijn,ij->n', np.take(U_del, vertices, axis=0), wts)
            U_f = np.einsum('vpnj,pjn->vpn', np.take(U_del,vertices,axis=1),wts)
    else:
        U_f = tre_samla.get_kd_U(tx) # U_f har shape (2,1) dersom tx.shape == (3,1)
        addedmass = False
        linear = False
        lift = False
        # return np.einsum('j,j->', np.take(U_del[0], vertices), wts), np.einsum('j,j->', np.take(U_del[1], vertices), wts),  np.einsum('j,j->', np.take(U_del[2], vertices), wts),  np.einsum('j,j->', np.take(U_del[3], vertices), wts)
        # return U[0][kd_index], U[1][kd_index], U[2][kd_index], U[3][kd_index]

    try:
        if (collision['is_resting_contact']):
            if U_f.shape == (2,1): # Skal sjekka om det er frå delaunay eller kd-tre. Er det frå kd-tre, er U_f.shape == (2,1) og er det lineær interpolasjon er U_f.shape == (2,4,1)
                U_f = U_f - collision['rib_normal']* np.einsum('ij,in->n',collision['rib_normal'], U_f) # tangentialkomponenten er lik U_f - normalkomponenten. Normalkomponenten er lik n * dot(U_f,n), for dot(U_f,n) = |U_f|cos(α), som er lik projeksjonen av U_f på normalvektoren, der projeksjonen er hosliggjande katet og U_f er hypotenusen.
            else:
                U_f = U_f - collision['rib_normal'][:,None] * np.einsum('ij,ipn->pn',collision['rib_normal'], U_f) # 
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
        dudt_material = np.zeros((2,number_of_vectors))

    if (lift):
        # skal finna farten i passande punkt over og under partikkelen for lyftekraft
        U_rel = U_f[:, 0] - x_inn[2:]
        
        # particle_top =    x_inn[0:2] + np.asarray([[0, -1],[1, 0]]) @ (particle.radius * norm(U_rel) )
        # particle_bottom = x_inn[0:2] + np.asarray([[0, 1],[-1, 0]]) @ (particle.radius * norm(U_rel) )
        
        particle_top_and_bottom = x_inn[0:2] + np.stack( (np.asarray([[0, -1],[1, 0]]) @ (particle.radius * norm(U_rel) ) ,  #Topp 
                                                            np.asarray([[0, 1],[-1, 0]]) @ (particle.radius * norm(U_rel) )) ) # botn

        part = np.concatenate((np.broadcast_to([frame],(2,1,number_of_vectors)), particle_top_and_bottom),axis=1)
        
        part_simplex = simplex[:2]
        part_vertices = vertices[:2]
        part_temp = temp[:2]

        simplex_prob = 0
        while (True):
            part_delta = part - part_temp[:,:,d,:].swapaxes(1,2) #avstanden til referansepunktet i simpleksen.
            part_bary = np.einsum('pnjk,pkn->pjn', part_temp[:,:,:d, :], part_delta) 
            part_wts = np.concatenate((part_bary, 1 - part_bary.sum(axis=1, keepdims=True)),axis=1)
        
            if (np.any(part_wts < -0.02) and  np.all(part >= tri.min_bound.reshape(3,1)) and np.all(part <= tri.max_bound.reshape(3,1))):
                part_simplex = tri.find_simplex(part.swapaxes(-2,-1))
                if np.any(part_simplex == -1):
                    print("denne staden skulle eg aldri ha kome til, for eg har alt sjekka om eg er utanfor med å ta part > tri.min_bound....")
                    break
                part_vertices = np.take(tri.simplices, part_simplex, axis=0)
                part_temp = np.take(tri.transform, part_simplex, axis=0)
                simplex_prob += 1
                if simplex_prob > 20:
                    print("Går i loop i part_simplex og der!")
                    break
            else:
                break
                
        U_top_bottom = np.einsum('vpnj,pjn->vpn', np.take(U_del, part_vertices, axis=1), part_wts)
    else:
        U_top_bottom = np.zeros((2,2,number_of_vectors))
    
    if (linear):
        U_f = U_f[:,0]

    return (U_f, dudt_material, U_top_bottom)

  
get_u.counter = 0
get_u.utanfor = 0
get_u.simplex_prob = 0