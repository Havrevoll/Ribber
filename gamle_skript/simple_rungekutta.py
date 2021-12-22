import numpy as np

from math import hypot

def f(t,y):
    """Returner fart og akselerasjon

    Args:
        t (float): tid
        y (array): [x,y,u,v]

    Returns:
        array: [u,v,a_x,a_y]
    """
      
    Re = hypot(y[2],y[3]) * 0.06
    
    cd = 24 / Re * (1+0.15*Re**0.687)

    drag =  3/4* cd /0.06 /2.65 * abs(-y[2:]) * (-y[2:])
    gravity =  np.array([0,(1/2.65-1)*9810])


    return np.hstack([y[2:], drag + gravity])


y0 = np.array([13.11481, -73.7378, 0.114264 , 0.3707])
t0 = 0


def rk(f, L, y0, h):
    ''' Heimelaga Runge-Kutta-metode '''
    t0, t1 = L
    N=int((t1-t0)/h)

    t=[0]*N # initialize lists
    y=[0]*N # initialize lists
    
    t[0] = t0
    y[0] = y0
    
    for n in range(0, N-1):
        #print(n,t[n], y[n], f(t[n],y[n]))
        k1 = h*f(t[n], y[n])
        k2 = h*f(t[n] + 0.5 * h, y[n] + 0.5 * k1)
        k3 = h*f(t[n] + 0.5 * h, y[n] + 0.5 * k2)
        k4 = h*f(t[n] + h, y[n] + k3)
                
        t[n+1] = t[n] + h
        y[n+1] = y[n] + 1/6 * (k1 + 2*k2 + 2*k3 + k4)
        
    return t, y

hei = rk(f, (0,0.005), y0, 0.0001)