# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 10:23:53 2021

@author: havrevol
"""
    
def e_alg(m, n):
    
    while (True):
        r = (m % n)
        
        if (r==0):
            return n
        else:
            m = n
            n = r
            
            