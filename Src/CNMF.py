# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 16:49:28 2015

@author: aepaezt
"""

import numpy as np
from numpy import dot

def CNMF(KX,epochs,W,G,calculateErrors = False):
    KXP,KXN = PostiveNegative(KX)
    errors = list()
    if calculateErrors:
        errors.append(Error(KX,W,G))
    for i in xrange(epochs):
        n = dot(KXP,W) + dot(G,dot(W.T,dot(KXN,W)))
        d = dot(KXN,W) + dot(G,dot(W.T,dot(KXP,W)))
        f = np.sqrt(n/d)
        G = G * f
        n = dot(KXP,G) + dot(KXN,dot(W,dot(G.T,G)))
        d = dot(KXN,G) + dot(KXP,dot(W,dot(G.T,G)))
        f = np.sqrt(n / d)
        W = W * f
        if calculateErrors:
            errors.append(Error(KX,W,G))
    return W,G,errors

def PostiveNegative(X):
    XP = (abs(X)+ X)/2
    XN = (abs(X)- X)/2
    return XP,XN

def Error(KX,W,G):
    E1 = np.trace(KX)
    E2 = np.trace(np.dot(np.dot(KX,W),G.T))
    E3 = np.trace(np.dot(G,np.dot(W.T,np.dot(np.dot(KX,W),G.T))))
    return E1 + E3 - (2.0 * E2)