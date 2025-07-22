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

def rateT(k, cA, order=1):
    """
    Power series reaction rate law (up to order 2),
    for a transformation reaction: aA -> cC

    Parameters
    ----------
    k : float
        reaction rate constant, with appropiate units...
    
    cA : float
        concentration of limiting reactant A, with appropiate units...

    order : int
        reaction order: 0, 1, 2
        
    Returns
    -------
        float
        reaction rate, r, in [(mol.m-3).s-1]
    """
    if order==0: return k           # k, [(mol.m-3).s-1]
    elif order==1: return k*cA      # k, [s-1]
    elif order==2: return k*(cA)**2 # k, [(m3.mol-1).s-1]
    else: raise NotImplementedError("only implemented orders are: 0, 1, 2")


def rateS(k, cA, cB, order=1):
    """
    Power series reaction rate law (up to order 2),
    for a synthesis reaction: aA + bB -> cC

    Parameters
    ----------
    k : float
        reaction rate constant, with appropiate units...
    
    cA : float
        concentration of limiting reactant A, with appropiate units...

    cB : float
        concentration of another reactant B, with appropiate units...

    order : int
        reaction order: 0, 1, 2 (-2 for case: -r=k*cA*cB)
        
    Returns
    -------
        float
        reaction rate, r, in [(mol.m-3).s-1]
    """
    if order==0: return k           # k, [(mol.m-3).s-1]
    elif order==1: return k*cA      # k, [s-1]
    elif order==2: return k*(cA)**2 # k, [(m3.mol-1).s-1]
    elif order==-2: return k*cA*cB  # k, [(m3.mol-1).s-1]
    else: raise NotImplementedError("only implemented orders are: 0, 1, 2")


def rateD(k, cA, order=1):
    """
    Power series reaction rate law (up to order 2),
    for a decomposition reaction: aA -> cC + dD

    Parameters
    ----------
    k : float
        reaction rate constant, with appropiate units...
    
    cA : float
        concentration of limiting reactant A, with appropiate units...

    order : int
        reaction order: 0, 1, 2
        
    Returns
    -------
        float
        reaction rate, r, in [(mol.m-3).s-1]
    """
    if order==0: return k           # k, [(mol.m-3).s-1]
    elif order==1: return k*cA      # k, [s-1]
    elif order==2: return k*(cA)**2 # k, [(m3.mol-1).s-1]
    else: raise NotImplementedError("only implemented orders are: 0, 1, 2")


def rateG(k, cA, cB, order=1):
    """
    Power series reaction rate law (up to order 2),
    for a general reaction: aA + bB -> cC + dD

    Parameters
    ----------
    k : float
        reaction rate constant, with appropiate units...
    
    cA : float
        concentration of limiting reactant A, with appropiate units...

    cB : float
        concentration of another reactant B, with appropiate units...

    order : int
        reaction order: 0, 1, 2 (-2 for case: -r=k*cA*cB)
        
    Returns
    -------
        float
        reaction rate, r, in [(mol.m-3).s-1]
    """
    if order==0: return k
    elif order==1: return k*cA
    elif order==2: return k*(cA)**2
    elif order ==-2: return k*cA*cB
    else: raise NotImplementedError("only implemented orders are: 0, 1, 2")




#%%#############################################################################
"""
Batch reactor models for different reaction types,
in terms of rate of change of amount of all species.
"""

def batchT(t, n, s, k, V, order=1):
    """
    Differential equation system (batch reactor)
    for a transformation reaction: aA -> cC

    Parameters
    ----------
    t : float
        reaction time, [s]
    
    n : array
        amount of substance [nA, nC], [mol]

    s : array
        stoichiometric coefficients [a, c], [-]

    k : float
        reaction rate constant, with appropiate units...
    
    V : float
        volume of reactor, [m3]

    order : int
        reaction order: 0, 1, 2
        
    Returns
    -------
        array
        rate of change of amount of substance [mol.s-1]
    """
    nA, nC = n
    a, c = s
    cA = nA/V
    rA = -1*rateT(k, cA, order)
    dNA_dt, dNC_dt = rA*V, -(c/a)*rA*V
    return np.array([dNA_dt, dNC_dt])


def batchS(t, n, s, k, V, order=1):
    """
    Differential equation system (batch reactor)
    for a synthesis reaction: aA + bB -> cC

    Parameters
    ----------
    t : float
        reaction time, [s]
    
    n : array
        amount of substance [nA, nB, nC], [mol]

    s : array
        stoichiometric coefficients [a, b, c], [-]

    k : float
        reaction rate constant, with appropiate units...
    
    V : float
        volume of reactor, [m3]

    order : int
        reaction order: 0, 1, 2
        
    Returns
    -------
        array
        rate of change of amount of substance [mol.s-1]
    """
    nA, nB, nC = n
    a, b, c = s
    cA, cB = nA/V, nB/V
    rA = -1*rateS(k, cA, cB, order)
    dNA_dt, dNB_dt, dNC_dt = rA*V, (b/a)*rA*V, -(c/a)*rA*V
    return np.array([dNA_dt, dNB_dt, dNC_dt])


def batchD(t, n, s, k, V, order=1):
    """
    Differential equation system (batch reactor)
    for a decomposition reaction: aA -> cC + dD

    Parameters
    ----------
    t : float
        reaction time, [s]
    
    n : array
        amount of substance [nA, nC, nD], [mol]

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
        rate of change of amount of substance [mol.s-1]
    """
    nA, nC, nD = n
    a, c, d = s
    cA = nA/V
    rA = -1*rateD(k, cA, order)
    dNA_dt, dNC_dt, dND_dt = rA*V, -(c/a)*rA*V, -(d/a)*rA*V
    return np.array([dNA_dt, dNC_dt, dND_dt])


def batchG(t, n, s, k, V, order=1):
    """
    Differential equation system (batch reactor)
    for a general reaction: aA + bB -> cC + dD

    Parameters
    ----------
    t : float
        reaction time, [s]
    
    n : array
        amount of substance [nA, nB, nC, nD], [mol]

    s : array
        stoichiometric coefficients [a, b, c, d], [-]

    k : float
        reaction rate constant, with appropiate units...
    
    V : float
        volume of reactor, [m3]

    order : int
        reaction order: 0, 1, 2
        
    Returns
    -------
        array
        rate of change of amount of substance [mol.s-1]
    """
    nA, nB, nC, nD = n
    a, b, c, d = s
    cA, cB = nA/V, nB/V
    rA = -1*rateG(k, cA, cB, order)
    dNA_dt, dNB_dt, dNC_dt, dND_dt = rA*V, (b/a)*rA*V, -(c/a)*rA*V, -(d/a)*rA*V
    return np.array([dNA_dt, dNB_dt, dNC_dt, dND_dt])


# The function below is for stopping calculation if
# amount of limitining reactant equals to zero
def endBatch(t, n, s, k, V, order=1): return n[0] 
endBatch.terminal = True




#%%#############################################################################
"""
Batch reactor models for different reaction types,
in terms of conversion of the limiting reactant.
"""

def batchXT(t, X, n, s, k, V, order=1):
    """
    Differential equation for rate of conversion, X
    for a transformation reaction: aA -> cC

    Parameters
    ----------
    t : float
        reaction time, [s]

    X : float
        conversion, [-]
    
    n : array
        initial amount of substance [nA, nC], [mol]

    s : array
        stoichiometric coefficients [a, c], [-]

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
    nA, nC = n
    a, c = s # not used, but left for completeness...
    tA = nA/nA
    cA = nA*(tA-X)/V
    rA = -1*rateT(k, cA, order)
    dX_dt = -rA * V / nA
    return np.array([dX_dt])


def batchXS(t, X, n, s, k, V, order=1):
    """
    Differential equation for rate of conversion, X
    for a synthesis reaction: aA + bB -> cC

    Parameters
    ----------
    t : float
        reaction time, [s]

    X : float
        conversion, [-]
    
    n : array
        initial amount of substance [nA, nC], [mol]

    s : array
        stoichiometric coefficients [a, c], [-]

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
    nA, nB, nC = n
    a, b, c = s
    tA, tB = nA/nA, nB/nB
    cA, cB = nA*(tA-X)/V, nA*(tB -(b/a)*X)
    rA = -1*rateS(k, cA, cB, order)
    dX_dt = -rA * V / nA
    return np.array([dX_dt])



def batchXD(t, X, n, s, k, V, order=1):
    """
    Differential equation for rate of conversion, X
    for a decomposition reaction: aA -> cC + dD

    Parameters
    ----------
    t : float
        reaction time, [s]

    X : float
        conversion, [-]
    
    n : array
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
    nA, nC, nD = n
    a, c, d = s
    tA = nA/nA
    cA = nA*(tA-X)/V
    rA = -1*rateD(k, cA, order)
    dX_dt = -rA * V / nA
    return np.array([dX_dt])


def batchXG(t, X, n, s, k, V, order=1):
    """
    Differential equation for rate of conversion, X
    for a general reaction: aA + bB -> cC + dD

    Parameters
    ----------
    t : float
        reaction time, [s]

    X : float
        conversion, [-]
    
    n : array
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
    nA, nB, nC, nD = n
    a, b, c, d = s
    tA, tB = nA/nA, nB/nB
    cA, cB = nA*(tA-X)/V, nA*(tB -(b/a)*X)
    rA = -1*rateG(k, cA, cB, order)
    dX_dt = -rA * V / nA
    return np.array([dX_dt])


# The function below is for stopping calculation if
# amount of limitining reactant equals to zero
def endBatchX(t, X, n, s, k, V, order=1): return n[0] 
endBatch.terminal = True