import numpy as np

def get_u(t, x_inn, particle, tre_samla, collision, skalering):
    c1 = 300
    c2 =  .5
    kvervel = c1*np.flip(x_inn[:2]*[1,-1])/(1+c2*(np.square(x_inn[:2])+np.square(np.flip(x_inn[:2]))))
    horisontal = 100

    return kvervel+horisontal