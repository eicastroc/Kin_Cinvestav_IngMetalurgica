# -*- coding: utf-8 -*-
"""
Edgar Ivan CASTRO CEDENO
2025/07/xx
E-mail: edgar.castro@cinvestav.mx
"""

#%% Imports
import numpy as np
from scipy.integrate import solve_ivp




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
PFR models for different reaction types,
in terms of conversion of the limiting reactant.
"""

def pfrX(V, X, Q, cA0, k, order=1):
    """
    Differential equation for rate of conversion, X
    in a piston flow reactor (PFR)
    for a general reaction: aA + bB -> cC + dD

    Parameters
    ----------
    V : float
        pfr volume, [m3]

    X : float
        conversion, [-]
    
    Q : float
        steady-state regime volumetric flow rate, [m3.s-1]
    
    cA0 : float
        initial concentration of limiting reactant A, [mol.m-3]

    k : float
        reaction rate constant, with appropiate units...

    order : int
        reaction order: 0, 1, 2
        
    Returns
    -------
        array
        rate of change of conversion [s-1]
    """

    # estimation of rates (power series rate law)
    cA = cA0*(1-X)
    rA = -1.0 * rate(k, cA, order)

    # construction of differential equation for conversion
    FA0 = Q*cA0
    dX_dV = -rA / FA0
    return np.array([dX_dV])





def pfrXEnd(V, X, Q, cA0, k, order=1):
    """
    End-point condition for differential equation.
        Takes same input as pfrX().
    End-point is when one of the reactants is fully consumed.
    """
    return 1 - X[0]
pfrXEnd.terminal = True




def pfrXFun(Xo, k, cA0, Q, Vn, order=1):
    sol = solve_ivp(
        pfrX, 
        (0, Vn),
        y0=[Xo], 
        args=(Q, cA0, k, order)
    )
    Xn=sol.y[0][-1]
    return Xn


def pfrInSeriesX(num_pfrs, k, cA0, Q, V_total, order=1, Xo=0):
    eps=1e-10
    V = V_total / num_pfrs

    # PFR Calculations for Cumulative Conversion
    pfr_indices = np.arange(0, num_pfrs + 1)
    X_pfr_cumulative = np.zeros(num_pfrs + 1)  # Cumulative conversion after each CSTR
    X_pfr_cumulative[0] = Xo  # Initial cumulative conversion (no reaction

    for i in range(1, num_pfrs + 1):
        Xo = X_pfr_cumulative[i-1]
        # Conversion in the current PFR step
        Xn = pfrXFun(Xo, k, cA0, Q, V, order)
        # Update cumulative conversion
        X_pfr_cumulative[i] = min(1.0*Xn, 1.0)

    return pfr_indices, X_pfr_cumulative


def pfrInParallelX(num_pfrs, k, cA0, Q, V_total, order=1, Xo=0):
    eps = 1e-10
    V = V_total / num_pfrs
    Q = Q / num_pfrs

    # PFR Calculations for parallel conversion
    pfr_indices = np.arange(1, num_pfrs+1)
    X_pfr_paral = np.zeros(num_pfrs)
    
    for i in range(0, num_pfrs):
        # Conversion in the current PFR step
        Xn = pfrXFun(Xo, k, cA0, Q, V, order)
        # Update conversion
        X_pfr_paral[i] = min(1.0*Xn, 1.0)
   

    return pfr_indices, X_pfr_paral