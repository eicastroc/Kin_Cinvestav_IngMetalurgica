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
    if order>=0: return k*np.power(cA, order)
    else: raise ValueError("power series considers only positive values")




#%%#############################################################################
"""
Batch reactor model for a general reaction,
in terms of rate of change of amount of all species.
"""


def batch(t, n, s, k, V, order=1):
    """
    Differential equation system (batch reactor)
    for a general reaction: aA + bB + ... --> cC + dD + ...

    Parameters
    ----------
    t : float
        reaction time, [s]
    
    n : array
        amount of substance [nA, nB, nC, nD], [mol]

    s : array
        stoichiometric coefficients [-a, -b, c, d], [-]
        reactants have negative sign, products have positive sign.

    k : float
        reaction rate constant, with appropiate units...
    
    V : float
        volume of reactor, [m3]

    order : int
        reaction order: a positive number
        
    Returns
    -------
        array
        rate of change of amount of substance [mol.s-1]
    """

    # error catching
    errors=[
        "at least two species need to be considered: length of array n must be >= 2",
        "input arrays [n] and [s] are not of equal length",
        "no reactants considered: at least one value in array [s] needs to be negative",
        "no products considered: at least one value in array [s] needs to be positive",
        "sign of s[0] is not negative (s[0] is assumed to be limiting reactant)",
    ]
    if len(n) < 2: raise ValueError(errors[0])
    if len(n) != len(s): raise ValueError(errors[1])
    if len(np.where(s<0)[0])<1: raise ValueError(errors[2])
    if len(np.where(s>0)[0])<1: raise ValueError(errors[3])
    if (np.sign(s[0]) > 0) is np.True_: raise ValueError(errors[4])

    # estimation of rates (power series rate law)
    cA = n[0]/V
    rA = -1.0 * rate(k, cA, order)

    # construction of differential equation system
    dN_dt = (s/s[0]) * rA * V
    return np.array(dN_dt)


def batchEnd(t, n, s, k, V, order=1):
    """
    End-point condition for differential equation.
        Takes same input as batch().
    End-point is when one of the reactants is fully consumed.
    """
    ridx = np.where(s<0)[0]
    return min(n[ridx])
batchEnd.terminal = True




#%%#############################################################################
"""
Batch reactor models for different reaction types,
in terms of conversion of the limiting reactant.
"""

def batchX(t, X, nA0, k, V, order=1):
    """
    Differential equation for rate of conversion, X
    in a batch reactor
    for a general reaction: aA + bB -> cC + dD

    Parameters
    ----------
    t : float
        reaction time, [s]

    X : float
        conversion, [-]
    
    n0 : array
        initial amount of substance [nA, nC, nD], [mol]

    s : array
        stoichiometric coefficients [a, c, d], [-]

    k : float
        reaction rate constant, with appropiate units...
    
    V : float
        volume of reactor, [m3]

    order : int
        reaction order: 0, 1, 2
        
    Returns
    -------
        array
        rate of change of conversion [s-1]
    """

    # estimation of rates (power series rate law)
    cA = nA0*(1-X)/V
    rA = -1.0 * rate(k, cA, order)

    # construction of differential equation for conversion
    dX_dt = -rA * V / nA0
    return np.array([dX_dt])



def batchXEnd(t, X, nA0, k, V, order=1):
    """
    End-point condition for differential equation.
        Takes same input as batch().
    End-point is when one of the reactants is fully consumed.
    """
    return 1 - X[0]
batchXEnd.terminal = True
