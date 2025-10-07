# -*- coding: utf-8 -*-
"""
Edgar Ivan CASTRO CEDENO
2025/10/xx
E-mail: edgar.castro@cinvestav.mx
"""

#%% Imports
import numpy as np




#%%#############################################################################
"""
Rate laws for different reaction types (power series model)
"""

def rate(k, cA, order=1):
    """
    Power series rate law, considering a single species A.

    Parameters
    ----------
    k : float
        reaction rate constant, with appropiate units...
    
    cA : float
        concentration of limiting reactant A, [mol.m-3]

    order : float
        reaction order: a positive number
        
    Returns
    -------
        array
        specific rate of change of amount of substance [mol.m-3.s-1]    
    """
    if order>=0: return k*np.power(cA, order)
    else: raise ValueError("power series considers only positive values")



def rateRP(k, cA, cR, oA=1.0, oR=0.0):
    """
    Power series rate law, considering reactant (R) and product (P)
    for autocatalytic reactions

    Parameters
    ----------
    k : float
        reaction rate constant, with appropiate units...
    
    cA : float
        concentration of reactant A, [mol.m-3]

    cR : float
        concentration of catalytic product R, [mol.m-3]

    oA : float
        reaction order: a positive number

    oR : float
        reaction order: a positive number
        
    Returns
    -------
        array
        specific rate of change of amount of substance [mol.m-3.s-1]    
    """
    return k * np.power(cA, oA) * np.power(cR, oR)
    

