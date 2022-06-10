# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 12:31:23 2021

@author: havrevol
"""
import random
import numpy as np
# import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle

from pathlib import Path
from scipy.optimize import root_scalar

from copy import deepcopy

def dobla(a, tal=1):
    b =  (a[1:] + a[:-1]) / 2
    c = np.empty((a.size + b.size,))
    c[0::2] = a
    c[1::2] = b
    
    if(tal == 1):    
        return c
    else:
        return dobla(c,tal-1)

def norm(v):
    return np.divide(v , (v**2).sum()**0.5, out=np.zeros_like(v), where=v!=0)

def draw_rect(axes, ribs, color='red', fill = True):
    
        # axes.add_patch(Rectangle((-61.07,-8.816),50.2,7.8,linewidth=2,edgecolor='none',facecolor=color))
        # axes.add_patch(Rectangle((37.6,-8.5),50,7.8,linewidth=2,edgecolor=color,facecolor='none'))
    for rib in ribs:
        if fill:
            axes.add_patch(Polygon(rib.vertices, facecolor=color))
        else:
            axes.add_patch(Polygon(rib.vertices, facecolor='none', edgecolor = color, linewidth = .7))


def ranges(kutt=False):
    # Dette var dei eg brukte for å laga kvadrantanalysen.
    # y_range = np.s_[0:114]
    # x_range = np.s_[40:108]
    
    # Her får me med heile biletet med data, og kuttar ut berre ytste kolonne og golvet:    
    if kutt:
        y_range = np.s_[1:114]
        x_range = np.s_[1:125]    
    else:
        y_range = np.s_[:]
        x_range = np.s_[:]
    
    piv_range = np.index_exp[y_range,x_range]
    
    return piv_range

def finn_fil(kandidatar):
    for fil in kandidatar:
        if fil.exists():
            return fil
    raise Exception("Fila finst ikkje på dei stadene du leita")


def sortClockwise(A):
    ''' Insertion sort from Introduction to Algorithms '''
    centerPoint = np.sum(A, axis=0)/len(A)
    
    for j in range(1,len(A)):
        key = deepcopy(A[j])
        i = j-1
        while i >= 0 and getIsLess(key , A[i], centerPoint):
            A[i+1] = A[i]
            i = i - 1
        A[i+1] = key

    return A

def getIsLess(a, b, center):
    ''' Comparison method from https://stackoverflow.com/questions/6989100/sort-points-in-clockwise-order '''
    if a[0]-center[0] >= 0 and b[0]-center[0] < 0:
        return True
    if a[0] - center[0] < 0 and b[0] - center[0] >= 0:
        return False
    if a[0] - center[0] == 0 and b[0] - center[0] == 0:
        if a[1] - center[1] >= 0 or b[1] - center[1] >= 0:
            return a[1] > b[1]
        return b[1] > a[1]
    
    # compute the cross product of vectors (center -> a) x (center -> b)
    # det = (a[0] - center[0]) * (b[1] - center[1]) - (b[0] - center[0]) * (a[1] - center[1])
    det = np.cross(a-center, b-center)
    # print("det = ",det, "det2 =", det2)
    if det < 0:
        return True
    elif det > 0:
        return False

    # points a and b are on the same line from the center
    # check which point is closer to the center
    d1 = (a[0] - center[0]) * (a[0] - center[0]) + (a[1] - center[1]) * (a[1] - center[1])
    d2 = (b[0] - center[0]) * (b[0] - center[0]) + (b[1] - center[1]) * (b[1] - center[1])
    return d1 > d2

def create_bins(in_list):
    in_list = np.asarray(in_list)
    return np.vstack((in_list[:-1],in_list[1:])).T 

def term_vel(d, Δ=1.65, ν=1.5674, c1=20, c2=1.1, g=9810):
    """Ferguson & Church 2004

    Args:
        d (float): partikkeldiameter i mm.
        Δ (float): dykka eigenvekt [-]. Defaults to 1.65.
        ν (float): kinematisk viskositet i mm²/s. Defaults to 1.5674.
        c1 (int, optional): 1. parameter for kornigheit. Defaults to 20.
        c2 (float, optional): 2. parameter for kornigheit. Defaults to 1.1.
        g (int, optional): Gravitasjon i mm/s². Defaults to 9810.

    Returns:
        float: fallsnøggleik i vatn for ein mineralpartikkel
    """    
    return Δ * g * d**2/(c1 * ν + (0.75 * c2 * Δ * g * d**3)**0.5)
    
def diff(d,u):
    return (term_vel(d) - u )

def scale_bins(bins,factor):
    u_m = term_vel(bins)
    u_p = u_m * factor**0.5

    bins_p = np.zeros(bins.shape)

    for i,(d,u) in enumerate(zip(bins,u_p)):
        res = root_scalar(diff,bracket=[d,d*1.1*factor],args=(u))
        bins_p[i] = res.root
    
    return bins_p

def f2t(f,scale=1):
    """Gjer om frame til tid

    Args:
        f (int): Datasett-nummer (0-3599)
        scale (int, optional): Skaleringsfaktor, til dømes 20 eller 40? Defaults to 1.
    
    Returns:
        float: tid som tilsvarer frame. Til dømes i skala 1: 1000 -> 50
    """    
    return f*0.05*scale**0.5

def t2f(t,scale=1):
    """Gjer om tid til frame

    Args:
        t (float): Tidspunkt
        scale (int, optional): Skaleringsfaktor, til dømes 20 eller 40? Defaults to 1.

    Returns:
        int: Datasett-nummer (0-3599)
    """
    return t*20./scale**0.5

def status_colors():
    while True:
        # style = random.randint(0,4)
        text_color = random.randint(30,38)+ 60*random.randint(0,1)
        background = random.randint(40,48) 
        combined = (text_color, background)
        bad = [(30,40),(30,48),(31,41), (32,42), (33,43), (34,44),(35,44),(35,45),(36,46),(37,47),(35,41),(36,42),(37,43),(31,45),(32,46),(33,47),(90,44),(92,42),(93,43),(93,47),(93,47),(96,41),(96,42),(96,45),(97,43),(97,47),(98,43)]
        if (combined not in bad):
            break
    return f"{str()};{str(text_color)};{str(background)}"