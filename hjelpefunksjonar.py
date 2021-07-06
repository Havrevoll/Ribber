# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 12:31:23 2021

@author: havrevol
"""
import numpy as np
# import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import os.path

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
    axes.add_patch(Rectangle((-61.07,-8.816),50.2,7.8,linewidth=2,edgecolor=color,facecolor='none'))
    axes.add_patch(Rectangle((37.6,-8.5),50,7.8,linewidth=2,edgecolor=color,facecolor='none'))

def ranges():
    # Dette var dei eg brukte for å laga kvadrantanalysen.
    # y_range = np.s_[0:114]
    # x_range = np.s_[40:108]
    
    y_range = np.s_[1:114]
    x_range = np.s_[1:125]    
    
    piv_range = np.index_exp[y_range,x_range]
    
    return piv_range

def finn_fil(kandidatar):
    for fil in kandidatar:
        if os.path.isfile(fil):
            return fil
    return Exception("Fila finst ikkje på dei stadene du leita")