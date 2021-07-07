# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 11:42:08 2021

@author: havrevol
"""

import multiprocessing

from datagenerering import lagra_tre, lag_tre

def lag_tre_multi(t_span, filnamn):
    
    a_pool = multiprocessing.Pool()
    
    t_min = t_span[0]
    t_max = t_span[1]
    
    i = [(i/10,(i+1.5)/10) for i in range(t_min*10,t_max*10)]
    
    result = a_pool.map(lag_tre, i)
    
    
    i_0 =  range(t_min*10,t_max*10)
    
    trees = dict(zip(i_0, result))

    lagra_tre(trees, filnamn)
    