import numpy as np
from hjelpefunksjonar import norm

nullfart = np.zeros(2)

def get_u_simple(t, x_inn, particle, collision):
     return (np.array([100,0]), nullfart, np.zeros((2,2)))

def get_u_log(t, x_inn, particle, collision):
     return (np.array([100,0]), nullfart, np.zeros((2,2)))


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
    
    U_del = tre_samla.get_U(tx)
    tri = tre_samla.get_tri(tx)
    
    x = np.vstack((tx,tx + np.array([dt,0,0]), tx + np.array([0,dx,0]), tx +np.array([0,0,dy])))
        
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