# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 12:31:23 2021

@author: havrevol
"""
import numpy as np
# import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import os.path

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
    return v / (v**2).sum()**0.5

def draw_rect(axes,color='red'):
    axes.add_patch(Rectangle((-61.07,-8.816),50.2,7.8,linewidth=2,edgecolor='none',facecolor=color))
    axes.add_patch(Rectangle((37.6,-8.5),50,7.8,linewidth=2,edgecolor=color,facecolor='none'))

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
        if os.path.isfile(fil):
            return fil
    return Exception("Fila finst ikkje på dei stadene du leita")


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
