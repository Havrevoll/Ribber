import numpy as np
from math import log
from constants import g, ρ_p, ρ,ν
from hjelpefunksjonar import norm

π = np.pi
c1 = 2 * 50 * π * 200 # sirkulasjon, dvs. ved radius 50 mm skal farten vera 200 mm/s.
J = 0.0001
h = 300.
γ = g * ρ
τ0 = γ * h * J
u_star = (τ0/ρ)**0.5
k_s = 20.
hev = 100.


def get_u_log(x):
    # with np.errstate(divide='raise'):
    u_log = np.zeros_like(x)
    u_log[0] = np.clip((2.5*np.log(np.clip(x[1]/k_s,a_min=0.00001,a_max=None)) + 8.5) * u_star,a_min=0.0,a_max=None)
    return u_log

u_log_max = get_u_log(np.asarray([[0], [h]]))[0]


def get_u(t, x_inn, particle, collision, skalering):
    #x_inn = x,y,u,v

    t= 50
    number_of_vectors = x_inn.shape[-1] #Dette er talet på vektorar pga vectorized i solve_ivp. Vanlegvis 1, men kan vera fleire, t.d. 4.

    kvervel = kvervelfunksjon(x_inn, c1,t)
    u_log = get_u_log(x_inn[:2])
    U_f =  kvervel * u_log[0]/u_log_max + u_log

    try:
        if (collision['is_resting_contact']):
            if U_f.shape == (2,number_of_vectors): # Skal sjekka om det er frå delaunay eller kd-tre. Er det frå kd-tre, er U_f.shape == (2,1) og er det lineær interpolasjon er U_f.shape == (2,4,1)
                U_f = U_f - collision['rib_normal'] * np.einsum('ij,in->n',collision['rib_normal'], U_f) # tangentialkomponenten er lik U_f - normalkomponenten. Normalkomponenten er lik ň * dot(U_f,ň), for dot(U_f,ň) = |U_f|cos(α), som er lik projeksjonen av U_f på normalvektoren, der projeksjonen er hosliggjande katet og U_f er hypotenusen. ň er kollisjonsnormalen og |ň|=1.
            else:
                U_f = U_f - collision['rib_normal'][:,None] * np.einsum('ij,ipn->pn',collision['rib_normal'], U_f) #
    except KeyError:
        pass
    
    u_top_bottom = get_u_top_bottom(t, x_inn, particle, U_f)
    dudt_material = get_dudt_material(t, x_inn, U_f)
    
    return U_f, u_top_bottom, dudt_material

def kvervelfunksjon(x_inn, c1, t):
    x_kvervel = x_inn[:2] - np.array([[0],[hev]])
    r =np.square(x_kvervel).sum(axis=-2)
    # kvervel = -np.expm1(-r/(4*ν))*c1*np.flip(x_kvervel, axis=-2)*np.array([[1],[-1]])/(r*4*np.pi)
    kvervel = np.divide(-np.expm1(-r/(4*ν*t))*c1*np.flip(x_kvervel, axis=-2)*np.array([[1],[-1]]) ,(r*2*np.pi), out=np.zeros_like(x_kvervel), where=r!=0)
    return kvervel

def get_u_top_bottom(t, x_inn, particle, U_f):
    vel = U_f - x_inn[2:]
    particle_top_and_bottom = x_inn[0:2] + np.stack( (np.asarray([[0, -1],[1, 0]]) @ (particle.radius * norm(vel) ) ,  #Topp 
                                                            np.asarray([[0, 1],[-1, 0]]) @ (particle.radius * norm(vel) )) ) # botn
    
    U_kvervel_top = kvervelfunksjon(particle_top_and_bottom[0], c1, t)
    U_kvervel_bottom = kvervelfunksjon(particle_top_and_bottom[1], c1, t)
    U_kvervel_top_bottom = np.swapaxes(np.stack((U_kvervel_top,U_kvervel_bottom)),0,1)
    U_log_top_bottom = get_u_log(np.swapaxes(particle_top_and_bottom,0,1))
    # U_log_top_bottom = np.zeros_like(particle_top_and_bottom)
    # U_log_top_bottom[:,0] = (2.5*np.log(np.clip(particle_top_and_bottom[:,1,:]/k_s,a_min=0.05,a_max=None)) + 8.5) * u_star
    return U_kvervel_top_bottom  * (U_log_top_bottom[0]/u_log_max) + U_log_top_bottom

def get_dudt_material(t, x_inn, U_f ):
    
    #dudt_material 
    # x = x_kvervel[0]
    # y = x_kvervel[1]
    x = x_inn[0]
    y = x_inn[1]
    u = U_f[0]
    v = U_f[1]
    x_hev = x_inn[:2] - np.array([[0],[hev]]) 
    y_hev = x_hev[1]

    r =np.square(x_hev).sum(axis=-2)
    
    # em1_ledd = (- np.expm1(np.divide(-x**2 - y**2,4*t*ν,out=np.full_like(x,-np.inf),where=t!=0)))
    em1_ledd = -np.expm1((-r)/(4*t*ν))
    e_ledd = np.exp((-r)/(4*t*ν))
    du_constant1 = 1/(4*t*ν*π*np.square(r))
    u_log= get_u_log(x_inn[:2])[0]


    # dudx = -c1*du_constant1*u_log*u_star*x*y_hev*(-e_ledd*r + 4*em1_ledd*t*ν)/u_log_max
    # dudy = du_constant1*u_star*(c1*e_ledd*r*u_log*y*y_hev**2 + 2*c1*em1_ledd*r*t*u_log*y*ν + 5.0*c1*em1_ledd*r*t*y_hev*ν - 4*c1*em1_ledd*t*u_log*y*y_hev**2*ν + 10.0*r**2*t*u_log_max*ν*π)/(u_log_max*y)
    # dvdx = c1*du_constant1*(-e_ledd*r*x**2 - 2*em1_ledd*r*t*ν + 4*em1_ledd*t*x**2*ν)
    # dvdy = -c1*du_constant1*x*y_hev*(e_ledd*r - 4*em1_ledd*t*ν)
    # dudt = 0
    # dvdt = 0

    dudx = -c1*du_constant1*u_log*x*y_hev*(-e_ledd*r + 4*em1_ledd*t*ν)/u_log_max
    dudy = du_constant1*u_star*(c1*e_ledd*r*u_log*y*y_hev**2/u_star + 2*c1*em1_ledd*r*t*u_log*y*ν/u_star + 5.0*c1*em1_ledd*r*t*y_hev*ν - 4*c1*em1_ledd*t*u_log*y*y_hev**2*ν/u_star + 10.0*r**2*t*u_log_max*ν*π)/(u_log_max*y)
    dvdx = c1*du_constant1*u_log*(-e_ledd*r*x**2 - 2*em1_ledd*r*t*ν + 4*em1_ledd*t*x**2*ν)/u_log_max
    dvdy = c1*du_constant1*u_star*x*(-e_ledd*r*u_log*y*y_hev/u_star - 5.0*em1_ledd*r*t*ν + 4*em1_ledd*t*u_log*y*y_hev*ν/u_star)/(u_log_max*y)
    dudt = 0
    dvdt = 0

    # dudx = c1*u_star*x*(hev - y)*(4*t*ν*(1 - np.exp(-(x**2 + (-hev + y)**2)/(4*t*ν))) - (x**2 + (hev - y)**2)*np.exp(-(x**2 + (hev - y)**2)/(4*t*ν)))*u_log/(4*t*u_log_max*ν*π*(x**2 + (hev - y)**2)**2)
    # dudy = u_star*(-4*c1*t*y*ν*(1 - np.exp(-(x**2 + (hev - y)**2)/(4*t*ν)))*(hev - y)**2*u_log + 2*c1*t*y*ν*(1 - np.exp(-(x**2 + (hev - y)**2)/(4*t*ν)))*(x**2 + (hev - y)**2)*u_log - 5.0*c1*t*ν*(1 - np.exp(-(x**2 + (hev - y)**2)/(4*t*ν)))*(hev - y)*(x**2 + (hev - y)**2) + c1*y*(hev - y)**2*(x**2 + (hev - y)**2)*u_log*np.exp(-(x**2 + (hev - y)**2)/(4*t*ν)) + 10.0*t*u_log_max*ν*π*(x**2 + (hev - y)**2)**2)/(4*t*u_log_max*y*ν*π*(x**2 + (hev - y)**2)**2)
    # dvdx = c1*(4*t*x**2*ν*(1 - np.exp(-(x**2 + (-hev + y)**2)/(4*t*ν))) - 2*t*ν*(1 - np.exp(-(x**2 + (-hev + y)**2)/(4*t*ν)))*(x**2 + (hev - y)**2) - x**2*(x**2 + (hev - y)**2)*np.exp(-(x**2 + (hev - y)**2)/(4*t*ν)))/(4*t*ν*π*(x**2 + (hev - y)**2)**2)
    # dvdy = c1*x*(hev - y)*(-4*t*ν*(1 - np.exp(-(x**2 + (-hev + y)**2)/(4*t*ν))) + (x**2 + (hev - y)**2)*np.exp(-(x**2 + (hev - y)**2)/(4*t*ν)))/(4*t*ν*π*(x**2 + (hev - y)**2)**2)
    # dudt = 0
    # dvdt = 0

    dudt_material = np.asarray([ dudt +  u * dudx + v * dudy , dvdt +  u * dvdx + v * dvdy ]) 
    return dudt_material
