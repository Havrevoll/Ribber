import numpy as np
from math import log
from constants import g, ρ_p, ρ
from hjelpefunksjonar import norm

def get_u(t, x_inn, particle, collision, skalering):
    #x_inn = x,y,u,v
    number_of_vectors = x_inn.shape[-1] #Dette er talet på vektorar pga vectorized i solve_ivp. Vanlegvis 1, men kan vera fleire, t.d. 4.

    c1 = 300
    c2 =  1
    x_kvervel, kvervel = kvervelfunksjon(x_inn, c1, c2)
    # horisontal = 100

    J = 0.0001;
    h = 100
    γ = g * ρ
    τ0 = γ * h * J
    u_star = (τ0/ρ)**0.5
    k_s = 10

    u_log = np.zeros_like(x_inn[:2])
    u_log[0] = (2.5*np.log(x_inn[1]/k_s) + 8.5) * u_star
    U_f =  kvervel + u_log

    

    try:
        if (collision['is_resting_contact']):
            if U_f.shape == (2,number_of_vectors): # Skal sjekka om det er frå delaunay eller kd-tre. Er det frå kd-tre, er U_f.shape == (2,1) og er det lineær interpolasjon er U_f.shape == (2,4,1)
                U_f = U_f - collision['rib_normal'] * np.einsum('ij,in->n',collision['rib_normal'], U_f) # tangentialkomponenten er lik U_f - normalkomponenten. Normalkomponenten er lik ň * dot(U_f,ň), for dot(U_f,ň) = |U_f|cos(α), som er lik projeksjonen av U_f på normalvektoren, der projeksjonen er hosliggjande katet og U_f er hypotenusen. ň er kollisjonsnormalen og |ň|=1.
            else:
                U_f = U_f - collision['rib_normal'][:,None] * np.einsum('ij,ipn->pn',collision['rib_normal'], U_f) #
    except KeyError:
        pass

    vel = U_f - x_inn[2:]
    particle_top_and_bottom = x_inn[0:2] + np.stack( (np.asarray([[0, -1],[1, 0]]) @ (particle.radius * norm(vel) ) ,  #Topp 
                                                            np.asarray([[0, 1],[-1, 0]]) @ (particle.radius * norm(vel) )) ) # botn

    _, U_kvervel_top_bottom = kvervelfunksjon(particle_top_and_bottom, c1, c2)

    U_log_top_bottom = np.zeros_like(particle_top_and_bottom)
    U_log_top_bottom[:,0] = (2.5*np.log(particle_top_and_bottom[:,1,:]/k_s) + 8.5) * u_star
    U_top_bottom = U_kvervel_top_bottom + U_log_top_bottom

    #dudt_material 
    x = x_kvervel[0]
    y = x_kvervel[1]

    du_constant = c1 / (1+c2*(x**2+y**2))**2
    dudx = -c2 * x * y  * du_constant
    dudy = (1 + c2 * (x**2 - y**2)) * du_constant
    dvdx = - dudy
    dvdy = - dudx
    dudt_material = np.asarray([ [(y  * dudx - x * dudy ) / (1+x**2+y**2)] + 2.5 * u_star/ x, 
                        [(y  * dvdx - x * dvdy ) / (1+x**2+y**2)]]) 
    
    return kvervel + u_log, U_top_bottom, dudt_material

def kvervelfunksjon(x_inn, c1,c2):
    x_kvervel = x_inn[:2] - np.array([[0],[50]]) 
    kvervel = c1*np.flipud(x_kvervel)*np.array([[1],[-1]])/(1+c2*((np.square(x_kvervel)).sum(axis=-2)))
    return x_kvervel, kvervel