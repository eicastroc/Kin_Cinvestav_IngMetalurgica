#%% Imports
import numpy as np

#%% Event where the limiting reactant is completely consummed

# Terminar el cálculo si el reactivo A es consumido completamente
def Aconsumed(t, n, s, k, V, order=1): return n[0] 
Aconsumed.terminal = True


#%% Batch reactor equations transformation reactions, aA --> cC


def rateT(k, cA, order=1):
    """
    Ley de velocidad de reacción, aA -> cC (orden 0, 1 y 2)

    Parametros:
        k: constante de reacción, [(mol.m-3).s-1], [s-1], [(m3.mol-1).s-1]
        cA: concentración del reactivo limitante, [mol.m-3]
        order: orden de la reacción: 0, 1, 2
    """
    if order==0: return k           # k, [(mol.m-3).s-1]
    elif order==1: return k*cA      # k, [s-1]
    elif order==2: return k*(cA)**2 # k, [(m3.mol-1).s-1]
    else: raise NotImplementedError("only implemented orders are: 0, 1, 2")


def batchT(t, n, s, k, V, order=1):
    """
    Sistema de ecuaciones reactor batch, aA -> cC
    
    Parametros:
        t: tiempo, [s]
        n: cantidad de especie [nA, nC], [mol]
        s: coefs. estequiométricos [a, c], [-]
        k: constante de reacción, [(mol.m-3).s-1], [s-1], [(m3.mol-1).s-1]
        V: volumen del reactor, [m3]
        order: orden de la reacción: 0, 1, 2
    """
    nA, nC = n
    a, c = s
    cA, cC = nA/V, nC/V
    rA = -1*rateT(k, cA, order)
    dNA_dt, dNC_dt = rA*V, -(c/a)*rA*V
    return np.array([dNA_dt, dNC_dt])


#%% Batch reactor equations for synthesis reactions, aA + bB --> cC 

def rateS(k, cA, cB, order=1):
    """
    Ley de velocidad de reacción, aA + bB -> cC (orden 0, 1 y 2)

    Parametros:
        k: constante de reacción, [(mol.m-3).s-1], [s-1], [(m3.mol-1).s-1]
        cA: concentración del reactivo limitante, [mol.m-3]
        cB: concentración del segundo reactivo, [mol.m-3]
        order: orden de la reacción: 0, 1, 2
    """
    if order==0: return k           # k, [(mol.m-3).s-1]
    elif order==1: return k*cA      # k, [s-1]
    elif order==2: return k*(cA)**2 # k, [(m3.mol-1).s-1]
    elif order==-2: return k*cA*cB  # k, [(m3.mol-1).s-1]
    else: raise NotImplementedError("only implemented orders are: 0, 1, 2")


def batchS(t, n, s, k, V, order=1):
    """
    Sistema de ecuaciones reactor batch, aA + bB -> cC
    
    Parametros:
        t: tiempo, [s]
        n: cantidad de especie [nA, nC], [mol]
        s: coefs. estequiométricos [a, c], [-]
        k: constante de reacción, [(mol.m-3).s-1], [s-1], [(m3.mol-1).s-1]
        V: volumen del reactor, [m3]
        order: orden de la reacción: 0, 1, 2
    """
    nA, nB, nC = n
    a, b, c = s
    cA, cB, cC = nA/V, nB/V, nC/V
    rA = -1*rateS(k, cA, cB, order)
    dNA_dt, dNB_dt, dNC_dt = rA*V, (b/a)*rA*V, -(c/a)*rA*V
    return np.array([dNA_dt, dNB_dt, dNC_dt])



#%% Batch reactor equation for decomposition reactions, aA --> cC + dD

def rateD(k, cA, order=1):
    """
    Ley de velocidad de reacción, aA -> cC + dD (orden 0, 1 y 2)

    Parametros:
        k: constante de reacción, [(mol.m-3).s-1], [s-1], [(m3.mol-1).s-1]
        cA: concentración del reactivo limitante, [mol.m-3]
        order: orden de la reacción: 0, 1, 2
    """
    if order==0: return k           # k, [(mol.m-3).s-1]
    elif order==1: return k*cA      # k, [s-1]
    elif order==2: return k*(cA)**2 # k, [(m3.mol-1).s-1]
    else: raise NotImplementedError("only implemented orders are: 0, 1, 2")


def batchD(t, n, s, k, V, order=1):
    """
    Sistema de ecuaciones reactor batch, aA -> cC + dD
    
    Parametros:
        t: tiempo, [s]
        n: cantidad de especie [nA, nC], [mol]
        s: coefs. estequiométricos [a, c], [-]
        k: constante de reacción, [(mol.m-3).s-1], [s-1], [(m3.mol-1).s-1]
        V: volumen del reactor, [m3]
        order: orden de la reacción: 0, 1, 2
    """
    nA, nC, nD = n
    a, c, d = s
    cA, cC, cD = nA/V, nC/V, nD/V
    rA = -1*rateD(k, cA, order)
    dNA_dt, dNC_dt, dND_dt = rA*V, -(c/a)*rA*V, -(d/a)*rA*V
    return np.array([dNA_dt, dNC_dt, dND_dt])


#%% Batch reactor equation for general reaction, aA + bB --> cC + dD

def rateG(k, cA, cB, order=1):
    """
    Ley de velocidad de reacción, aA + bB -> cC + dD (orden 0, 1 y 2)

    Parametros:
        k: constante de reacción, [(mol.m-3).s-1], [s-1], [(m3.mol-1).s-1]
        cA: concentración del reactivo limitante, [mol.m-3]
        cB: concentración del segundo reactivo, [mol.m-3]
        order: orden de la reacción: 0, 1, 2
    """
    if order==0: return k
    elif order==1: return k*cA
    elif order==2: return k*(cA)**2
    elif order ==-2: return k*cA*cB
    else: raise NotImplementedError("only implemented orders are: 0, 1, 2")


def batchG(t, n, s, k, V, order=1):
    """
    Sistema de ecuaciones reactor batch, A -> B
    """
    nA, nB, nC, nD = n
    a, b, c, d = s
    cA, cB, cC, cD = nA/V, nB/V, nC/V, nD/V
    rA = -1*rateG(k, cA, cB, order)
    dNA_dt, dNB_dt, dNC_dt, dND_dt = rA*V, (b/a)*rA*V, -(c/a)*rA*V, -(d/a)*rA*V
    return np.array([dNA_dt, dNB_dt, dNC_dt, dND_dt])