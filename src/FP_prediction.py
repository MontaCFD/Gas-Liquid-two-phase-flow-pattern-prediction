#! /usr/bin/env python3
'''===================================================================================================================
 - Author:  Guesmi
 - Adress: Chair of process engineering TU Dresden, Helmholtzstraße 10, 01069 Dresden, Germany
 - Created/updated on Mon Sept 18 2023
 - Summary: Flow pattern prediction for:
            > horizontal and slightly inclined pipes between (-10° to +10°)
            > Liquid-Gas two phase flow
======================================================================================================================'''
import numpy
import numpy as np
import scipy.optimize
from numpy import sin, cos, exp, log, log10, sqrt, arccos, arcsin, arctan, radians
from numpy import pi, e
from scipy.optimize import fsolve
from scipy.optimize import newton
import matplotlib as mpl
import matplotlib.pyplot as plt
import CoolProp.CoolProp as cp
import self as self
from thermodynamic_properties import  *
from utility import  *
from constants import *

# > mass quality
def f_slip_ratio(y, epsilon, rho_l, rho_g):
    '''
            :param vars:
                    y: flow quality
            :param data:
                    parameters are in the list data stored
            :return:
                    ratio s
            '''



    s = 1.0 / epsilon - (1.0 + sqrt(1.0 - y * (1.0 - rho_l / rho_g)) * ((1.0 - y) / y) * (rho_g / rho_l))
    return s


# assume liquid flow is turbulent (which is btw the case) & laminar gas phase flow (see page 67-68)
def smooth_pipe_exponents(Re):
    if (Re <= 2300):
        C = 16.
        n = 1.
    else:
        C = 0.064
        n = 0.2
    return C, n


# friction coefficient assuming pipe is smooth
def smooth_friction_factor(Re):
    '''

    :param Re: Reynolds number
    :return: friction number
    '''
    C, n = smooth_pipe_exponents(Re)
    f = C * Re ** (-n)
    return f


#---------------------------------------------------------------------------------------------------------------------

# > Equilibrium liquid level equation derivied from equation 3.28
def equlibrium_liquid_level(x, *data):
    '''
    :param vars:
            delta_l_tild
    :param data:
            parameters are in the list data stored
    :return:
            delta_l_tild
    '''
    g = 9.81
    U_sg, U_sl, rho_g, mu_g, rho_l, mu_l, epsilon, d, theta = data
    # equations are defined on page 67
    A_L = 0.25 * (pi - arccos(2.0 * x - 1.0) + (2.0 * x - 1.0) * sqrt(1.0 - (2.0 * x - 1.0) ** 2.0)) * (d ** 2.)
    A_G = 0.25 * (arccos(2.0 * x - 1.0) - (2.0 * x - 1.0) * sqrt(1.0 - (2.0 * x - 1.0) ** 2.0)) * (d ** 2.)
    S_L = (pi - arccos(2.0 * x - 1.0)) * d
    S_G = arccos(2.0 * x - 1.0) * d
    S_I = sqrt(1 - (2.0 * x - 1.0) ** 2.0) * d
    alpha = A_G / (A_G + A_L)  # numerically epsilon should be actullay a function alpha =  f(x) of equilibrium liquid level
    U_l = U_sl / (1. - alpha)
    U_g = U_sg / alpha
    #print(U_g, U_l)
    d_l = 4. * A_L / S_L
    d_g = 4. * A_G / (S_G + S_I)
    Re_l = U_l * d_l / (mu_l / rho_l)
    Re_g = U_g * d_g / (mu_g / rho_g)
    # print(Re_g, Re_l)
    f_l = smooth_friction_factor(Re_l)
    # print("fl", f_l)
    f_g = smooth_friction_factor(Re_g)
    # print("fg", f_g)
    f_i = f_g
    # print(f_l / f_g)

    # > combined momentum equation for the two phases 3.28
    # > assuming smooth interface exists (f_I = f_G) & interface velocity is neglected (V_g >> V_I)
    F = f_l * rho_l * (U_l ** 2.) * S_L * A_G - f_g * rho_g * (U_g ** 2.) * (S_G * A_L + S_I * (A_G + A_L)) + 2. * (rho_l - rho_g) * g * sin(radians(theta)) * A_G * A_L
    F = f_l * rho_l * (U_l ** 2.) * S_L * A_G - f_g * rho_g * (U_g ** 2.) * S_G * A_L - f_i * rho_g * (U_g - U_l) * abs(U_l - U_g) * S_I * (A_G + A_L) + 2. * (rho_l - rho_g) * g * sin(radians(theta)) * A_G * A_L

    return F


def FPT_Horizontal_Pipe_test(U_sg, U_sl, rho_g, mu_g, rho_l, mu_l, epsilon, d, theta, lamda, gamma, beta):
    # > flow pattern in horizontal pipe flow
    FP = {
        -1: "Single-phase",
         0: "SS",
         1: "SW",
         2: "A",
         3: "I",
         4: "DB",
    }
    # constants
    U_G = U_sg / epsilon
    U_L = U_sl / (1. - epsilon)
    # solving momentum equation
    data = (U_sg, U_sl, rho_g, mu_g, rho_l, mu_l, epsilon, d, theta)
    h_l_tild = scipy.optimize.bisect(equlibrium_liquid_level, a=0.000001, b=.99999, args=data, xtol=tol, maxiter=1000)
    # geometrical variables
    A_L = 0.25 * (pi - arccos(2.0 * h_l_tild - 1.0) + (2.0 * h_l_tild - 1.0) * sqrt(1.0 - (2.0 * h_l_tild - 1.0) ** 2.0)) * (d ** 2.)
    A_G = 0.25 * (arccos(2.0 * h_l_tild - 1.0) - (2.0 * h_l_tild - 1.0) * sqrt(1.0 - (2.0 * h_l_tild - 1.0) ** 2.0)) * (d ** 2.)
    S_L = (pi - arccos(2.0 * h_l_tild - 1.0)) * d
    S_G = arccos(2.0 * h_l_tild - 1.0) * d
    S_I = sqrt(1 - (2.0 * h_l_tild - 1.0) ** 2.0) * d
    # Superficial Froude numbers
    Fr_sg = sqrt(rho_g / ((rho_l - rho_g) * g * d)) * U_sg
    Fr_sl = sqrt(rho_l / ((rho_l - rho_g) * g * d)) * U_sl

    # coefficients and factors 
    # source doi:10.1016/j.cherd.2011.08.009
    # S-NS: lamda = 0.25  S is predicted to 100 % and lamda = 7.0   NS is predicted to 100 %
    #lamda = 1.0
    # B-NB: alpha factor 
    #gamma = 1.0
    # Annular- Non annular hL/d between 0.35 and 0.5 
    #beta = 0.35 
    
    if (U_sg == 0):
        ptt = -1
        print("single phase flow")
    else:
        # check stratified to non stratified transition (eq. 3.49)
        
        S_to_NS = (U_G >= sqrt(lamda)  * (1.0 - h_l_tild) * ((rho_l - rho_g) / rho_g * g * (A_G / S_I) * cos(radians(theta))) ** .5)

        if (not S_to_NS):
            s = 0.01
            Smooth_Wavy = (U_G >= (4. * mu_l * (rho_l - rho_g) * g * cos(radians(theta)) / (s * rho_l * rho_g * U_L)) ** 0.5)  # eq. 3.53
            if (Smooth_Wavy):
                #print("wavy-stratified")
                ptt = 1
            else:
                #print("smooth-stratifed")
                ptt = 0
        else:
            # check the transition from Bubbly or Intermittent to annular flow
            IB_to_A = (h_l_tild <= beta)
            if (IB_to_A):
                #print("Annular Flow")
                ptt = 2

            # if flow is not annular
            else:
                # check Intermittent to dispersed-bubble transition
                Re_l = A_L / S_L * U_L / (mu_l / rho_l)
                Fl   = smooth_friction_factor(Re_l)
                # alternative
                #S_I  = 2 * d * sin(0.5 * radians(angle(epsilon)))
                #I_to_B = (U_L ** 2 >= (4. * epsilon * (pi / 4 * d ** 2.) / S_I * g * cos(radians(theta)) / Fl * (1. - rho_g / rho_l)))
                I_to_B = (U_L >= gamma * (4. * A_G / S_I * g * cos(radians(theta))/ Fl * (1. - rho_g / rho_l)) ** 0.5)
                if (I_to_B):
                    #print("Bubbly flow")
                    ptt = 4
                else:
                    #print("Slug")
                    ptt = 3

    return ptt, FP[ptt], Fr_sl, Fr_sg



def FPT_Horizontal_Pipe(U_sg, U_sl, rho_g, mu_g, rho_l, mu_l, epsilon, d, theta):
    # > flow pattern in horizontal pipe flow
    FP = {
        -1: "Single-phase",
         0: "SS",
         1: "SW",
         2: "A",
         3: "I",
         4: "DB",
    }
    # constants
    U_G = U_sg / epsilon
    U_L = U_sl / (1. - epsilon)
    # solving momentum equation
    data = (U_sg, U_sl, rho_g, mu_g, rho_l, mu_l, epsilon, d, theta)
    h_l_tild = scipy.optimize.bisect(equlibrium_liquid_level, a=0.000001, b=.99999, args=data, xtol=tol, maxiter=1000)
    # geometrical variables
    A_L = 0.25 * (pi - arccos(2.0 * h_l_tild - 1.0) + (2.0 * h_l_tild - 1.0) * sqrt(1.0 - (2.0 * h_l_tild - 1.0) ** 2.0)) * (d ** 2.)
    A_G = 0.25 * (arccos(2.0 * h_l_tild - 1.0) - (2.0 * h_l_tild - 1.0) * sqrt(1.0 - (2.0 * h_l_tild - 1.0) ** 2.0)) * (d ** 2.)
    S_L = (pi - arccos(2.0 * h_l_tild - 1.0)) * d
    S_G = arccos(2.0 * h_l_tild - 1.0) * d
    S_I = sqrt(1 - (2.0 * h_l_tild - 1.0) ** 2.0) * d
    # Superficial Froude numbers
    Fr_sg = sqrt(rho_g / ((rho_l - rho_g) * g * d)) * U_sg
    Fr_sl = sqrt(rho_l / ((rho_l - rho_g) * g * d)) * U_sl

    
    # coefficients and factors 
    # source doi:10.1016/j.cherd.2011.08.009
    # S-NS: lamda = 0.25  S is predicted to 100 % and lamda = 7.0   NS is predicted to 100 %
    lamda = 1.0
    # B-NB: alpha factor 
    gamma = 1.0
    # Annular- Non annular hL/d between 0.35 and 0.5 
    beta = 0.35 
    
    if (U_sg == 0):
        ptt = -1
        print("single phase flow")
    else:
        # check stratified to non stratified transition (eq. 3.49)
        
        S_to_NS = (U_G >= sqrt(lamda)  * (1.0 - h_l_tild) * ((rho_l - rho_g) / rho_g * g * (A_G / S_I) * cos(radians(theta))) ** .5)

        if (not S_to_NS):
            s = 0.01
            Smooth_Wavy = (U_G >= (4. * mu_l * (rho_l - rho_g) * g * cos(radians(theta)) / (s * rho_l * rho_g * U_L)) ** 0.5)  # eq. 3.53
            if (Smooth_Wavy):
                #print("wavy-stratified")
                ptt = 1
            else:
                #print("smooth-stratifed")
                ptt = 0
        else:
            # check the transition from Bubbly or Intermittent to annular flow
            IB_to_A = (h_l_tild <= beta)
            if (IB_to_A):
                #print("Annular Flow")
                ptt = 2

            # if flow is not annular
            else:
                # check Intermittent to dispersed-bubble transition
                Re_l = A_L / S_L * U_L / (mu_l / rho_l)
                Fl   = smooth_friction_factor(Re_l)
                # alternative
                #S_I  = 2 * d * sin(0.5 * radians(angle(epsilon)))
                #I_to_B = (U_L ** 2 >= (4. * epsilon * (pi / 4 * d ** 2.) / S_I * g * cos(radians(theta)) / Fl * (1. - rho_g / rho_l)))
                I_to_B = (U_L >= gamma * (4. * A_G / S_I * g * cos(radians(theta))/ Fl * (1. - rho_g / rho_l)) ** 0.5)
                if (I_to_B):
                    #print("Bubbly flow")
                    ptt = 4
                else:
                    #print("Slug")
                    ptt = 3

    return ptt, FP[ptt], Fr_sl, Fr_sg


def Transition_line(U_sg, U_sl, rho_g, mu_g, rho_l, mu_l,epsilon, d, theta):
    # > check if the (Fr_sg, Fr_sl) is on one of the transition-lines. 
    # > if it is the case, return boolean true and store it in the corresponding T-line 
    FP = {
        -1: "Single-phase",
         0: "SS",
         1: "SW",
         2: "A",
         3: "I",
         4: "DB",
    }
    
    # constants
    U_G = U_sg / epsilon
    U_L = U_sl / (1. - epsilon)
    # solving momentum equation
    data = (U_sg, U_sl, rho_g, mu_g, rho_l, mu_l, epsilon, d, theta)
    h_l_tild = scipy.optimize.bisect(equlibrium_liquid_level, a=0.00001, b=.9999, args=data, xtol=tol, maxiter=1000)
    #print(tol)
    # geometrical variables
    A_L = 0.25 * (pi - arccos(2.0 * h_l_tild - 1.0) + (2.0 * h_l_tild - 1.0) * sqrt(1.0 - (2.0 * h_l_tild - 1.0) ** 2.0)) * (d ** 2.)
    A_G = 0.25 * (arccos(2.0 * h_l_tild - 1.0) - (2.0 * h_l_tild - 1.0) * sqrt(1.0 - (2.0 * h_l_tild - 1.0) ** 2.0)) * (d ** 2.)
    S_L = (pi - arccos(2.0 * h_l_tild - 1.0)) * d
    S_G = arccos(2.0 * h_l_tild - 1.0) * d
    S_I = sqrt(1 - (2.0 * h_l_tild - 1.0) ** 2.0) * d
    # Superficial Froude numbers
    Fr_sg = sqrt(rho_g / ((rho_l - rho_g) * g * d)) * U_sg
    Fr_sl = sqrt(rho_l / ((rho_l - rho_g) * g * d)) * U_sl
    # check wether the point is on the transition line or not 
    transition = False 
    tr_tol = 0.01
    L = ''
    if (U_sg == 0):
        ptt = -1
        print("single phase flow")
    else:
        # check stratified to non stratified transition (eq. 3.49)
        lamda      = 1.0
        lamda      = 0.25   # S is predicted to 100 %
        #lamda      = 7.0    #NS is predicted to 100 %
        S_to_NS    = ( U_G >= sqrt(lamda)  * (1.0 - h_l_tild) * ((rho_l - rho_g) / rho_g * g * (A_G / S_I) * cos(radians(theta))) ** .5
                      )
        #print(abs(1.0 - (1.0 / U_G) * sqrt(lamda)  * (1.0 - h_l_tild) * ((rho_l - rho_g) / rho_g * g * (A_G / S_I) * cos(radians(theta))) ** .5))
        transition = (abs(1.0 - (1.0 / U_G) * sqrt(lamda)  * (1.0 - h_l_tild) * ((rho_l - rho_g) / rho_g * g * (A_G / S_I) * cos(radians(theta))) ** .5) <= tr_tol)
        if (transition):
            L = 'TR1'
        if (not S_to_NS):
            s = 0.01
            Smooth_Wavy = (1.0  >= (1.0 / U_G) * (4. * mu_l * (rho_l - rho_g) * g * cos(radians(theta)) / (s * rho_l * rho_g * U_L)) ** 0.5)  # eq. 3.53
            #print(abs(1.0 - (1.0 / U_G) * (4. * mu_l * (rho_l - rho_g) * g * cos(radians(theta)) / (s * rho_l * rho_g * U_L)) ** 0.5))
            tr_tol = 0.005
            transition  = (abs(1.0 - (1.0 / U_G) * (4. * mu_l * (rho_l - rho_g) * g * cos(radians(theta)) / (s * rho_l * rho_g * U_L)) ** 0.5) <= tr_tol)
            if (transition):
                L = 'TR2'
            if (Smooth_Wavy):
                #print("wavy-stratified")
                ptt = 1
            else:
                #print("smooth-stratifed")
                ptt = 0
        else:
            # check the transition from Bubbly or Intermittent to annular flow
            IB_to_A    = (h_l_tild <= 0.35)
            tr_tol = 0.001
            transition = (abs(h_l_tild - 0.35) <= tr_tol)
            if (transition):
                L = 'TR3'
            if (IB_to_A):
                #print("Annular Flow")
                ptt = 2

            # if flow is not annular
            else:
                # check Intermittent to dispersed-bubble transition
                Re_l = A_L / S_L * U_L / (mu_l / rho_l)
                Fl   = smooth_friction_factor(Re_l)
                # I_to_B = (U_L ** 2 >= (4. * epsilon * A / S_I * g * cos(radians(theta)) / Fl * (1. - rho_g / rho_l)))
                I_to_B     = ( 1.0  >= (1.0 / U_L) * (4. * A_G / S_I * g * cos(radians(theta))/ Fl * (1. - rho_g / rho_l)) ** 0.5)
                tr_tol     = 0.001
                #print(abs(1.0 - (1.0 / U_L) * (4. * A_G / S_I * g * cos(radians(theta))/ Fl * (1. - rho_g / rho_l)) ** 0.5))
                transition = (abs(1.0 - (1.0 / U_L) * (4. * A_G / S_I * g * cos(radians(theta))/ Fl * (1. - rho_g / rho_l)) ** 0.5) <= tr_tol)
                if (transition):
                    L = 'TR4'
                    
                if (I_to_B):
                    #print("Bubbly flow")
                    ptt = 4
                else:
                    #print("Slug")
                    ptt = 3

    return ptt, L, transition


