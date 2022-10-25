import numpy as np
from hjelpefunksjonar import norm, t2f
from constants import g
g = np.array([[0], [g]]) # mm/s^2 = 9.81 m/s^2


# Så dette er funksjonen som skal analyserast av runge-kutta-operasjonen. Må ha t som fyrste og y som andre parameter.
# @jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
def get_u(t, x, particle, tre_samla, ribs, collision, skalering):
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
    get_u.counter +=1
  
    lift, addedmass, linear = particle.lift, particle.addedmass, particle.linear
    frame = t2f(t,skalering)
    number_of_vectors = x.shape[-1] #Dette er talet på vektorar pga vectorized i solve_ivp. Vanlegvis 1, men kan vera fleire, t.d. 4.
    field_width = ribs[1].get_rib_middle()[0] - ribs[0].get_rib_middle()[0]
    rib_width = ribs[1].get_rib_dimensions()[1]
    tx = np.concatenate((np.broadcast_to([frame],(1,number_of_vectors)), x[:2]),axis=-2)
    # dt, dx, dy = 0.01, 0.1, 0.1
    tx_avstand = (tx[1] - particle.init_position[0]) % field_width
    tx[1] = particle.init_position[0] + tx_avstand
    
    if linear:
        try:
            tri, U_del = tre_samla.get_tri_og_U(frame)
        except KeyError:
            tri, U_del = tre_samla.get_max_tri_og_U()

        innanfor = np.all(tx >= tri.min_bound.reshape(3,1)) and np.all(tx <= tri.max_bound.reshape(3,1))
    else:
        innanfor = False

    # assert not np.isnan(np.sum(tx))
    orig_num_vec = number_of_vectors
    if np.any(tx_avstand < (rib_width *.4)):  # or (real_tx[1] > (field_width - rib_width  *.4) ):
        vekting_av_venstre = tx_avstand * 2 / rib_width
        
        tx_bortanfor = np.copy(tx)
        tx_bortanfor[1] = tx_bortanfor[1] + field_width
        tx = np.hstack((tx,tx_bortanfor))
        number_of_vectors += number_of_vectors

    else:
        vekting_av_venstre = 1
    

    # kdtre = tre_samla.kdtre
    # U_kd = tre_samla.U_kd

    if (linear and innanfor):
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
            # https://en.wikipedia.org/wiki/Barycentric_coordinate_system#Barycentric_coordinates_on_tetrahedra

            # kodane her må vera:
            # p = dei ulike forskyvingane til bruk i material derivative/added mass 
            # n = tal på vektorar
            # v = u og v, horisontal og vertikal-komponentane
            # j = vertices i kvar simplex
            # k = tid, x og y (har eg meint komponentar?)

            vertices = np.take(tri.simplices, simplex, axis=0) # (p,n,j)
            temp = np.take(tri.transform, simplex, axis=0) # (p,n,j,k)
            Δ = 0.001
            delta = np.stack((tx,tx + np.asarray([[Δ],[0],[0]]), tx + np.asarray([[0],[Δ],[0]]), tx +np.asarray([[0],[0],[Δ]]))) - temp[:,:,d,:].swapaxes(1,2) # (p,k,n)
            bary = np.einsum('pnjk,pkn->pjn', temp[:,:,:d,:], delta) # (p,j,n) men den siste rada manglar frå j, den blir lagt til i wts.
            wts = np.concatenate((bary, 1 - bary.sum(axis=1, keepdims=True)),axis=1) # (p,j,n)
            
            # U_f = np.einsum('nij,ni->nj', np.take(np.column_stack(U_del),vertices,axis=0),wts)
            # U_f = np.einsum('ijn,ij->n', np.take(U_del, vertices, axis=0), wts)
            U_f = np.einsum('vpnj,pjn->vpn', np.take(U_del,vertices,axis=1),wts) # (v,p,n)
    else:
        U_f = tre_samla.get_kd_U(tx) # U_f har shape (2,1) dersom tx.shape == (3,1)
        addedmass = False
        linear = False
        lift = False
        # return np.einsum('j,j->', np.take(U_del[0], vertices), wts), np.einsum('j,j->', np.take(U_del[1], vertices), wts),  np.einsum('j,j->', np.take(U_del[2], vertices), wts),  np.einsum('j,j->', np.take(U_del[3], vertices), wts)
        # return U[0][kd_index], U[1][kd_index], U[2][kd_index], U[3][kd_index]

    try:
        if (collision['is_resting_contact']):
            if U_f.shape == (2,number_of_vectors): # Skal sjekka om det er frå delaunay eller kd-tre. Er det frå kd-tre, er U_f.shape == (2,1) og er det lineær interpolasjon er U_f.shape == (2,4,1)
                U_f = U_f - collision['rib_normal'] * np.einsum('ij,in->n',collision['rib_normal'], U_f) # tangentialkomponenten er lik U_f - normalkomponenten. Normalkomponenten er lik ň * dot(U_f,ň), for dot(U_f,ň) = |U_f|cos(α), som er lik projeksjonen av U_f på normalvektoren, der projeksjonen er hosliggjande katet og U_f er hypotenusen. ň er kollisjonsnormalen og |ň|=1.
            else:
                U_f = U_f - collision['rib_normal'][:,None] * np.einsum('ij,ipn->pn',collision['rib_normal'], U_f) #
    except KeyError:
        pass
                
    if (addedmass):
        dUdt = (U_f[:,1] - U_f[:,0]) / Δ # Fyrste verdien er dU/dt og andre er dV/dt
        dUdx = (U_f[:,2] - U_f[:,0]) / Δ # Fyrste verdien er dU/dx og andre er dV/dx
        dUdy = (U_f[:,3] - U_f[:,0]) / Δ # Fyrste verdien er dU/dy og andre er dV/dy

        # skal finna gradienten i t, u og v-retning for å bruka på added mass.
        # DU/Dt = dU/dt + u * dU/dx + v*dU/dy
        
        dudt_material = dUdt + U_f[0,0] * dUdx + U_f[1,0] * dUdy
        U_f = U_f[:,0] # (v,n)
    else:
        dudt_material = np.zeros((2,number_of_vectors))

    if (lift):
        # skal finna farten i passande punkt over og under partikkelen for lyftekraft
        U_rel = U_f - x[2:]
        
        # particle_top =    x_inn[0:2] + np.asarray([[0, -1],[1, 0]]) @ (particle.radius * norm(U_rel) )
        # particle_bottom = x_inn[0:2] + np.asarray([[0, 1],[-1, 0]]) @ (particle.radius * norm(U_rel) )
        
        particle_top_and_bottom = tx[1:] + np.stack( (np.asarray([[0, -1],[1, 0]]) @ (particle.radius * norm(U_rel) ) ,  #Topp 
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
    
    if np.any(vekting_av_venstre != 1):
        
        # U_f_ny = np.empty((2,orig_num_vec))
        U_f[:,:orig_num_vec] = (U_f[:,:orig_num_vec]*vekting_av_venstre + U_f[:,orig_num_vec:] * (1- vekting_av_venstre) )
        dudt_material[:,:orig_num_vec] = (dudt_material[:,:orig_num_vec]*vekting_av_venstre + dudt_material[:,orig_num_vec:] * (1- vekting_av_venstre) )
        U_top_bottom[:,:,:orig_num_vec] = (U_top_bottom[:,:,:orig_num_vec]*vekting_av_venstre + U_top_bottom[:,:,orig_num_vec:] * (1- vekting_av_venstre) )

    return (U_f[:,:orig_num_vec], dudt_material[:,:orig_num_vec], U_top_bottom[:,:,:orig_num_vec])

  
get_u.counter = 0
get_u.utanfor = 0
get_u.simplex_prob = 0