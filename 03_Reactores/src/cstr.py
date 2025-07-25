# -*- coding: utf-8 -*-
"""
Edgar Ivan CASTRO CEDENO
2025/07/xx
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

    order : int
        reaction order: a positive number
        
    Returns
    -------
        array
        specific rate of change of amount of substance [mol.m-3.s-1]    
    """
    if order>=0: return k*(cA)**(order)
    else: raise ValueError("power series considers only positive values")



#%%#############################################################################
"""
Continuosly-Stirred Tank Reactor (CSTR) model for a general reaction,
in terms of rate of change of amount of all species.
"""

def cstrQVX(X, Q, cA0, k, order=1):
    """
    Algebraic equation for volume of steady-state CSTR,  
    required to achieve a given conversion, X

    Parameters
    ----------
    X : float
        conversion, [-]

    Q : float
        steady-state regime volumetric flow rate, [m3.s-1]
    
    cA0 : float
        initial concentration of limiting reactant A, [mol.m-3]

    k : float
        reaction rate constant, with appropiate units...
    
    V : 

    order : int
        reaction order: 0, 1, 2
        
    Returns
    -------
        float
        volume of CSTR operating at steady-state regime, [m3]
    """

    # error catching
    errors =[
        "a value of conversion larger than 1 was input"
    ]
    if (X > 1).any(): raise ValueError(errors[0])

    # estimation of rates (power series rate law)
    cA = cA0*(1-X)
    rA = -1.0*rate(k, cA, order)
    return (Q*cA0*X)/(-rA)

