#! /usr/bin/env python3
'''===================================================================================================================
 - Author:  Guesmi
 - Adress: Chair of process engineering TU Dresden, Helmholtzstraße 10, 01069 Dresden, Germany
 - Summary: Define heat transfer functions
            > convective heat transfer
            > Two phase models
            > pressure drop approximation
            > Nusselt functions
            > physics & Geometry class definition
======================================================================================================================'''
import numpy as np
from numpy import sin, cos, exp, log, log10, sqrt, arccos, arcsin, arctan
from numpy import pi, e
from scipy.optimize import fsolve
from scipy.optimize import newton
import matplotlib as mpl
import matplotlib.pyplot as plt
import CoolProp.CoolProp as cp

import self as self
from thermodynamic_properties import  *
from utility import *
from FP_prediction import *
from constants import *

# > class geometry
class Geometry(object):
    def __init__(self, length: float, inner_diameter: float, outer_diameter: float, roughness:float) -> None:
        self.length = length
        self.inner_diameter = inner_diameter
        self.outer_diameter = outer_diameter
        self.roughness      = roughness
    def cross_section(self) -> float :
        cross_section = pi * (self.inner_diameter/2)**2
        return cross_section
    def heat_area(self)-> float :
        heat_area = pi * self.inner_diameter * self.length
        return  heat_area

    def conductivity(self):
        """
            This function calculates the thermal conductivity of steel
            Args:
                    T (in °C) Temperature
                    p (in bar) pressure
            Returns:
                    lambda (in W/mK) steel thermal conductivity
                    kupfer 400

        """
        Lambda = 50.
        return Lambda

# > class physics
class physics(object):
    def __init__(self, T_EL: float, T_EG: float, pressure: float, cooling_temperature: float, vof: float, mass_flow: float,  flow_form) -> None:
        self.T_EL = T_EL
        self.T_EG = T_EG
        self.pressure = pressure
        self.cooling_temperature = cooling_temperature
        self.vof = vof
        self.mass_flow = mass_flow
        self.flow_form = flow_form

# > class to return flow quality x and mass flow rate of a chosen phase
# > TODO: consider the case of constant volume flow rate besides constant flow rate --> Done :)
# > TODO: consider more than one slip ratio model --> Done :)
class two_phase_model(object):
    """
             This class contains the thermodynamics properties of a mixture and returns the flow quality, velocity of the phases
             Args:
                     phases
                     T (in °C) Temperature 0 bis 99
                     p (in bar) pressure
                     volume of fraction
                     mass flow rate
                     cross section of the pipe
             Returns:
                     flow quality x
                     mass flow rate of each phase
                     thermodynamic properties of the mixture

    """
    def __init__(self, phase_1, phase_2, T_FL: float, T_FG : float, pressure: float, vof: float, mass_flow: float, const_flow_rate: str, cross_section :float, assumption : bool, slip_ratio_model : str) -> None:

        self.phase_1          = phase_1
        self.phase_2          = phase_2
        self.bulkTemp         = T_FL
        self.gasTemp          = T_FG
        self.pressure         = pressure
        self.vof              = vof
        self.mass_flow        = mass_flow
        self.cross_section    = cross_section
        self.const_flow_rate  = const_flow_rate
        epsilon               = self.vof
        self.slip_ratio_model = slip_ratio_model

        # fluid properties
        phase_1 = "water"
        phase_2 = "oxygen"
        # phase 1: water
        Liquid = Fluid(self.bulkTemp, self.pressure, phase_1)
        # phase 2: oxygen
        Gas = Fluid(self.gasTemp, self.pressure, phase_2)

        # flow quality x = m_G/m_total
        @staticmethod
        def flow_quality(self, Liquid, Gas, slip_ratio_model):
            global x
            epsilon  = self.vof
            rho_G    = Gas.rho
            rho_L    = Liquid.rho
            mu_G     = Gas.muu
            mu_L     = Liquid.muu
            para     = (epsilon, rho_G, rho_L, slip_ratio_model)

            def f_slip_ratio(y):
                '''
                        :param vars:
                                x: flow quality
                        :param data:
                                parameters are in the list data stored
                        :return:
                                flow quality & slip ratio s
                        '''
                if (slip_ratio_model == "homogeneous"):
                    f = 1.0 / epsilon - (1.0 + ((1.0 - y) / y) * (rho_G / rho_L))

                elif (slip_ratio_model == "chisholm"):
                    f = 1.0 / epsilon - (
                            1.0 + sqrt(1.0 - y * (1.0 - rho_L / rho_G)) * ((1.0 - y) / y) * (rho_G / rho_L))

                elif (slip_ratio_model == "smith"):
                    K = 0.4
                    f = 1.0 / epsilon - (1 + (K + (1.0 - K) * sqrt(
                        ((rho_L / rho_G) + K * (1.0 - y) / y) / (1.0 + K * (1.0 - y) / y))) * ((1.0 - y) / y) * (
                                                  rho_G / rho_L))

                else:
                    print("please enter one of the three models: homogeneous, chisholm or smith!")
                    f = 1.0 / epsilon - (1.0 + (1.0 - y) / y * (rho_G / rho_L))
                return f

            rho_m = epsilon * Gas.rho + (1 - epsilon) * Liquid.rho
            if (epsilon == 0.):
                x = 0.0
                s = 0.0
            elif (epsilon == 1.0):
                x = 1.0
                s = NaN

            else:
                # > Chisholm model (1983)
                #'''Chisholm, D., 1983. Two-phase flow in pipelines and heat exchangers.'''
                x0 = epsilon * rho_G / rho_m
                if (slip_ratio_model == "homogeneous"):
                    x  = newton(f_slip_ratio, x0)
                    s  = 1.0
                elif (slip_ratio_model == "chisholm"):
                    x  = newton(f_slip_ratio, x0)
                    s  = sqrt(1.0 - x * (1.0 - rho_L / rho_G))
                elif (slip_ratio_model == "smith"):
                    K  = 0.4
                    x  = newton(f_slip_ratio, x0)
                    s  = (K + (1.0 - K) * sqrt(((rho_L / rho_G) + K * (1.0 -x) / x) / (1.0 + K * (1.0 - x) / x)))

                else:
                    print(slip_ratio_model)
                    print("please enter one of the three models: homogeneous, chisholm or smith!")
                    eq = 1.0 / epsilon - ((1.0 - x) / x) * (rho_G / rho_L) * (mu_L / mu_G)
                    s = 1.

            return x , s


        flow_quality_x , slip_ratio  = flow_quality(self, Liquid, Gas, slip_ratio_model)
        self.flow_quality            = flow_quality_x
        self.slip_ratio              = slip_ratio

        # > thermodynamic properties of the mixture
        self.rho = epsilon * Gas.rho + (1 - epsilon) * Liquid.rho
        self.k   = epsilon * Gas.k   + (1 - epsilon) * Liquid.k # volumetric averaging (not realisitc)
        self.k   = flow_quality_x * Gas.k + (1.0 - flow_quality_x) * Liquid.k  # masse averaging
        #print(self.k)
        #self.k   = Liquid.k
        #print(self.k)

        ''' >> different models to estimate the thermal conductivity of the mixture'''
        '''  > Model 1: [1] W.H. McAdams, W.K. Woods, L.C. Heroman, Vaporization inside horizontal tubes II-benzene-oil mixtures, Trans. ASME 64 (3) (1942) 193–200.'''
        #self.muu = ( flow_quality_x / Gas.muu + (1.0 - flow_quality_x) / Liquid.muu) ** (-1)
        '''  > Model 2: S. Lin, C.C.K. Kwok, R.Y. Li, Z.H. Chen, Z.Y. Chen, Local frictional pressure drop during vaporization for R-12 through capillary tubes, Int. J. Multiphase Flow 17 (1) (1991) 95–102'''
        self.muu = epsilon * Gas.muu + (1 - epsilon) * Liquid.muu
        self.Cp  = flow_quality_x * Gas.Cp + (1.0 - flow_quality_x) * Liquid.Cp
        ''' two model assumption: 
            > 1st Assumption: if the boolean variable assumption is false, use the thermodynamic properties of the mixture as defined in the literature
            > 2nd Assumption: if the boolean variable assumption is true,use the thermodynamics properties (rho and lambda) of water '''
        # assumption = False  # Mixture thermodynamic properties (False) otherwise warter thermodynamic properties (True)
        self.nu  = flow_quality_x * Gas.nuu + (1.0 - flow_quality_x) * Liquid.nuu
        #print(self.nu)
        self.nu  = self.muu / self.rho
        #print(self.nu)
        if (assumption):
            self.rho  = Liquid.rho
            self.k    = Liquid.k
            self.muu  = Liquid.muu
            self.Cp   = Liquid.Cp
            self.nu   = Liquid.nuu


        #print(self.nu)
        self.Pr = self.muu * self.Cp / self.k
        #print(self.muu, self.Cp, self.k)
        #print("Pr=", self.Pr)

        @staticmethod
        def mass_flow_phase(m_tp, x):
            m_g = x * m_tp
            m_l = (1-x) * m_tp
            return m_g, m_l

        # > mass flow rate depending on the assumption of
        # > - constant mass flow rate of two phase system
        # > - constant volume flow rate of the two phase system
        # > - constant mass flow rate of the liquid phase
        m_tp = mass_flow
        if(const_flow_rate == "const_volume_flow_rate"):
            m_tp = (epsilon * Gas.rho + (1 - epsilon) * Liquid.rho) / Liquid.rho * mass_flow
        if(const_flow_rate == "const_mass_flow_rate"):
            m_tp = mass_flow
        if (const_flow_rate == "const_liquid_flow_rate"):
            m_tp = mass_flow / (1.0 - flow_quality_x)

        self.m_G, self.m_L = mass_flow_phase(m_tp, flow_quality_x)
        self.mass_flow = m_tp
        # flow velocity
        self.velocity = self.mass_flow / ((epsilon * Gas.rho + (1 - epsilon) * Liquid.rho) * cross_section) # > self.mass_flow / (self.rho * A)


# > estimation of friction factor
def friction_factor(Re, e, D, f_g):
    """
            this function determines the friction factor
            Args:
                     Re: Reynolds number
                     e: wall roughness
                     D: pipe Diameter
                     f_g: initial guess of the friction factor

             Returns:
                     friction factor f_n
        """
    # > laminar flow
    if(Re < 2300.):
        f_n = 64.0 / Re
    # > turbulent flow 
    else:
        err   = 1.0
        i     = 0
        e_rel = e / D
        tol   = 1.e-5
        while (err > tol):
            i = i + 1
            # > evaluate the new value of f
            a   = e_rel / 3.7 + 2.51 / (Re * sqrt(f_g))
            f_n = 1.0 / (- 2.0 * log10(a)) ** 2.
            # > calculate error
            err = abs(f_n - f_g)
            # > update guess
            f_g = f_n
        
    return  f_n

#TODO
# > function to estimate the pressure drop of two phase system --> Done :)
def pressure_drop(T_FL, T_FG, Geometry, physics, phase_1, phase_2, assumption, const_flow_rate, slip_ratio_model):

    """
             this function uses the Lockhart-Martinelli-Method to estimate the pressure drop
             Args:
                     T_FL: Liquid phase temperature
                     T_FG: Gas phase temperature
                     P: pressure
                     vof: volume of fraction
                     phase_1, phase2: the Liquid and the Gas phases
                     D: Diameter
                     e: wall roughness
             Returns:
                     dP: Pressure drop

     """
    # > pipe wall roughness
    e = Geometry.roughness
    # > system pressure
    P = physics.pressure
    # > mass flow
    mf = physics.mass_flow
    # > Pipe Diameter
    D = Geometry.inner_diameter
    # > Pipe length
    L = Geometry.length
    # > cross section
    A = Geometry.cross_section()
    # > Gas fraction
    epsilon = physics.vof

    # thermodynamic properties of the mixture
    tp_model = two_phase_model(phase_1=phase_1, phase_2=phase_2, T_FL=T_FL, T_FG=T_FG, pressure=P, vof=epsilon, mass_flow=mf, const_flow_rate=const_flow_rate, cross_section=A, assumption=assumption, slip_ratio_model=slip_ratio_model)
    # > Gas phase fluid properties
    fluid_G  = Fluid(T_FG, P, phase_2)
    # > Liquid phase fluid properties
    fluid_L = Fluid(T_FL, P, phase_1)
    # > Superficial velocity U_SG & superficial Reynolds number of Gas phase
    U_SG     = tp_model.m_G / (fluid_G.rho * A)
    Re_SG    = U_SG * D / fluid_G.nuu

    # > Superficial velocity U_SL & Superficial Reynolds number of Liquid phase
    U_SL     = tp_model.m_L / (fluid_L.rho * A)
    Re_SL    = U_SL * D / fluid_L.nuu

    # > Pressure drop dP_G
    if(Re_SG > 0.):
        f_g  = friction_factor(Re_SG, e, D, f_g=0.01)
        dP_G = f_g * L / D * fluid_G.rho / 2.0 * (U_SG) ** 2
    else:
        dP_G = 0.
    #print(dP_G)
    # > Pressure drop dP_L
    if (Re_SL > 0.):
        f_l  = friction_factor(Re_SL, e, D, f_g= 0.01)
        dP_L = f_l * L / D * fluid_L.rho / 2.0 * U_SL ** 2
    else:
        dP_L = 0.0
    #print(dP_L)
    # > constant C
    '''---------------------------------------
       Liquid         Gas                C
       turbulent      turbulent (TT)     20
       laminar        turbulent (LT)     12
       turbulent      laminar   (TL)     10
       laminar        laminar   (LL)     5.
       ---------------------------------------
    '''
    Re_critical = 2300.
    if ((Re_SG < Re_critical) and (Re_SL < Re_critical)):
        C = 5.0
    elif ((Re_SG < Re_critical) and not (Re_SL < Re_critical)):
        C = 10.
    elif (not (Re_SG < Re_critical) and (Re_SL < Re_critical)):
        C = 12.
    else:
        C = 20.
    # > Lockhart-Martinelli-Parameter X
    if ((dP_L > 0.) and (dP_G > 0.)):
        X = ((dP_L/L)/(dP_G/L)) ** 0.5
        if (X < 1.0):
            phi = 1.0 + C * X + X ** 2
            dP_r = phi * dP_G
        else:
            phi = 1.0 + C / X + 1 / X ** 2
            dP_r = phi * dP_L
    else:
        if(dP_L == 0.):
            phi  = 1.
            dP_r = dP_G
        if (dP_G == 0.):
            phi = 1.
            dP_r = dP_L

    dP = dP_r
    return  dP

# > energy equations to solve without LMTD
def initial_guess(vars, *data):
    '''
    :param vars:
            T_AG  and T_AL
    :param data:
            parameters are in the list data stored
    :return:
            outlet temperature of the the Gas & Liquid phases in °C
    '''
    m_L, Cp_L, T_EL, K_L, alpha_L, S_L, T_L, \
    m_G, Cp_G, T_EG, K_G, alpha_G, S_G, T_G, \
    S_I, L, T_C = data
    # > variables
    T_AG, T_AL = vars
    # > balance equation WITHOUT USING LMTD
    eq3 = m_G * Cp_G * (T_AG - T_EG) + K_G * (0.5 * (T_AG + T_EG) - T_C) * S_G * L
    eq4 = m_L * Cp_L * (T_AL - T_EL) + K_L * (0.5 * (T_AL + T_EL) - T_C) * S_L * L

# > energy equations to solve without LMTD
def energy_equations(vars, *data):
    '''
    :param vars:
            T_AG  and T_AL
    :param data:
            parameters are in the list data stored
    :return:
            outlet temperature of the the Gas & Liquid phases in °C
    '''
    m_L, Cp_L, T_EL, K_L, alpha_L, S_L, T_L, \
    m_G, Cp_G, T_EG, K_G, alpha_G, S_G, T_G, \
    S_I, L, T_C = data
    # > variables
    T_AG, T_AL = vars
    # > balance equation WITHOUT USING LMTD
    eq3 = m_G * Cp_G * (T_AG - T_EG) + K_G * (0.5 * (T_AG + T_EG) - T_C) * S_G * L - alpha_G * (0.5 * (T_AL + T_EL) - 0.5 * (T_AG + T_EG)) * S_I * L  # was +
    eq4 = m_L * Cp_L * (T_AL - T_EL) + K_L * (0.5 * (T_AL + T_EL) - T_C) * S_L * L + alpha_G * (0.5 * (T_AL + T_EL) - 0.5 * (T_AG + T_EG)) * S_I * L


    return [eq3, eq4]

# > energy equations to solve with LMTD
def equations(vars, *data):
    '''
        :param vars:
                T_AG  and T_AL
        :param data:
                parameters are in the list data stored
        :return:
                outlet temperature of the the Gas & Liquid phases in °C
        '''
    m_L, Cp_L, T_EL, K_L, alpha_L, S_L, T_L, \
    m_G, Cp_G, T_EG, K_G, alpha_G, S_G, T_G, \
    S_I, L, T_C = data
    # > variables
    T_AG, T_AL = vars
    # > balance equation using the LMTD
    eq1 = m_G * Cp_G * (T_AG - T_EG) + K_G * (T_AG - T_EG) / log((T_AG - T_C) / (T_EG - T_C)) * S_G * L - alpha_G * (0.5 * (T_AL + T_EL) - 0.5 * (T_AG + T_EG)) * S_I * L
    eq2 = m_L * Cp_L * (T_AL - T_EL) + K_L * (T_AL - T_EL) / log((T_AL - T_C) / (T_EL - T_C)) * S_L * L + alpha_G * (0.5 * (T_AL + T_EL) - 0.5 * (T_AG - T_EG)) * S_I * L
    # > balance equation WITHOUT USING LMTD
    #eq3 = m_G * Cp_G * (T_AG - T_EG) + K_G * (0.5 * (T_AG + T_EG) - T_C) * S_G * L + alpha_G * (0.5 * (T_AL + T_EL) - 0.5 * (T_AG + T_EG)) * S_I * L
    #eq4 = m_L * Cp_L * (T_AL - T_EL) + K_L * (0.5 * (T_AL + T_EL) - T_C) * S_L * L - alpha_G * (0.5 * (T_AL + T_EL) - 0.5 * (T_AG - T_EG)) * S_I * L


    return [eq1, eq2]

# > Nusselt function
def Nusselt(Pr, Re, flow, Geometry):
    """
             this function returns the Nusselt number
             Args:
                     Pr: Prandtl-Number
                     Re: Reynolds-Number
                     Geometry: Length, inner diameter, outer diameter
             Returns:
                     Nu: Nusselt-Number

         """
    di = Geometry.inner_diameter
    da = Geometry.outer_diameter
    L  = Geometry.length
    Re_critical = 2300.  #2300.
    if (Re < Re_critical):    # laminar flow
        Nu = (3.66 ** 3 + 1.61 ** 3 * Re * Pr * di / L) ** (1 / 3)
        return Nu

    else:           # turbulent flow

        if(flow == "thermischer Anlauf"):
            Nusselt = 0.0235 * Re ** 0.8 * Pr ** 0.48
        if(flow == "hydraulisch ausgebildet"):
            f     = (1.8 * log10(Re) - 1.5) ** (-2)
            K     = 1.0
            #K     = ( 1 + (di / L) ** (2 / 3))

            # Gnielinski correlation (1976)
            N       = 1.0 + 12.7 * (f / 8.0) ** 0.5 * (Pr ** (2 / 3) - 1.0)
            Nusselt = (f / 8.0) * (Re) * Pr / N * K  # Nusselt function f(Re, Pr), d/L <=1
            # this formula was used by Ghajar & al. (2006)
            #Nusselt = 0.027 * Re ** (4. / 5.) * Pr ** (1. / 3.)  

        return Nusselt

# > TODO: implement Nusselt function for bubbly flow
# > Nusselt for bubbly turbulent flow
'''Zhang, H.-Q.; Wang, Q.; Sarica, C.; Brill, J. P.: Unified Model of Heat Transfer in Gas-Liquid Pipe Flow. DOI: 10.2523/90459-MS.'''

# > this function solves the convective heat transfer problem with Nusselt-functions
def convection(Geometry, physics, fluid_name):
    # Geometry
    di = Geometry.inner_diameter
    da = Geometry.outer_diameter
    cross_section = Geometry.cross_section()
    heat_area = Geometry.heat_area()
    lambda_R  = Geometry.conductivity()
    # Physics
    T_E = physics.T_EL
    P   = physics.pressure
    T_C = physics.cooling_temperature
    T_A = T_E
    mass_flow_rate = physics.mass_flow
    flow = physics.flow_form
    while True:
        T_A_old = T_A
        # update fluid temperature
        T_F = 0.5 * (T_E + T_A)
        # fluid properties
        fluid         = Fluid(T_F, P, fluid_name)
        density       = fluid.rho # > update density of fluid
        nu            = fluid.nuu # > update kinematic viscosity
        Pr            = fluid.Pr  # > update prandtl-number
        cp            = fluid.Cp  # > update heat capacity
        lambda_fluid  = fluid.k   # > update thermal conductivity
        U = mass_flow_rate / (density * cross_section)
        Re = U * di / nu          # > reynolds-number
        # Nusselt function
        Nu = Nusselt(Pr, Re, flow, Geometry) # > convective heat transfer coefficient
        alpha_i = Nu * lambda_fluid / di     # > heat transfer coefficient HTC
        HTC = 1 / (1 / alpha_i + di / (2 * lambda_R) * log(da / di))

        T_A = T_C + (T_E - T_C) * exp(- HTC * heat_area / (mass_flow_rate * cp)) # > calculate outlet temperature
        Q = HTC * heat_area * (T_A - T_E) / log((T_A - T_C) / (T_E - T_C)) # > calculate heat flux
        dH = mass_flow_rate * cp * (T_E - T_A) # > calculate enthalpy stream change
        # Residual
        error = abs(T_A - T_A_old)
        #print("error:", error)
        if (error < 10 ** -3):
            heat_flux = Q
            break

    return Nu, Re, alpha_i, HTC, T_A, heat_flux

# > extended function to solve the convective heat transfer problem with Nusselt-functions for a mixture
def convection_mixture(Geometry, physics, phase_1, phase_2, assumption, const_flow_rate, slip_ratio_model):

    # Geometry
    di = Geometry.inner_diameter
    da = Geometry.outer_diameter

    cross_section = Geometry.cross_section()
    heat_area     = Geometry.heat_area()
    lambda_R      = Geometry.conductivity()

    # Physics
    T_E = physics.T_EL
    P   = physics.pressure
    T_C = physics.cooling_temperature
    mass_flow_rate = physics.mass_flow
    T_A = T_E
    flow = physics.flow_form

    while True:
        T_A_old = T_A
        # update fluid temperature
        T_F = 0.5 * (T_E + T_A)
        # volume of fraction
        vof = physics.vof
        # thermodynamic properties of the mixture
        tp_model = two_phase_model(phase_1=phase_1, phase_2= phase_2, T_FL=T_F, T_FG=T_F, pressure=P,
                                            vof=vof, mass_flow=mass_flow_rate, const_flow_rate= const_flow_rate,  cross_section=cross_section, assumption=assumption, slip_ratio_model= slip_ratio_model)

        density_m = tp_model.rho   # update density of the mixture in [kg/m^3]
        mu_m      = tp_model.muu   # update dynamic viscosity of the mixture
        nu_m      = tp_model.nu    # update kinematic viscosity
        cp_m      = tp_model.Cp    # update heat capacity
        lambda_m  = tp_model.k     # update thermal conductivity
        Pr_m      = tp_model.Pr    # update Prandtl-number

        # > two phase mass flow rate
        m_tp = tp_model.mass_flow
        # velocity U in [m/s]
        U         = tp_model.velocity
        #print(U)

        #print(tp_model.m_L/(cross_section*(1.-vof)*tp_model.rho))
        #print("U=", U)
        # reynolds-number
        Re = U * di / nu_m
        # Nusselt function
        Nu = Nusselt(Pr_m, Re, flow, Geometry)
        # convective heat transfer coefficient
        alpha = Nu * lambda_m / di
        # heat transfer coefficient HTC
        HTC = 1 / (1 / alpha + di / (2 * lambda_R) * log(da / di))
        #print(alpha)
        #print(HTC)
        # calculate outlet temperature
        T_A = T_C + (T_E - T_C) * exp(- HTC * heat_area / (mass_flow_rate * cp_m))
        # LMTD
        LMTD = (T_A - T_E) / log((T_A - T_C) / (T_E - T_C))
        # calculate heat flux
        Q = HTC * heat_area * LMTD
        # calculate enthalpy stream change
        dH = mass_flow_rate * cp_m * (T_E - T_A)
        # residual
        error = abs(T_A - T_A_old)
        #print("error:", error)
        if (error < 10 ** -4):
            heat_flux = Q
            break

    return Nu, Re, alpha, HTC, T_A, heat_flux

# > extended function to solve the convective heat transfer problem with Nusselt-functions for the case as if only one phase alone were flowing
def convection_stratified(Geometry, physics, phase_1, phase_2, assumption, const_flow_rate, slip_ratio_model):

    # > Geometry
    di = Geometry.inner_diameter
    da = Geometry.outer_diameter
    L  = Geometry.length
    cross_section = Geometry.cross_section()
    heat_area     = Geometry.heat_area()
    lambda_R      = Geometry.conductivity()

    # > Physics
    T_EL = physics.T_EL
    T_EG = physics.T_EG
    P    = physics.pressure
    T_C  = physics.cooling_temperature
    flow = physics.flow_form
    mass_flow_rate  = physics.mass_flow
    # > volume of fraction
    vof = physics.vof
    # > initialize
    T_AL = physics.T_EL
    T_AG = physics.T_EL
    K_L = 3000.
    K_G = 300.
    if (vof == 0.0):
        Nu_G    = 0.0
        alpha_G = 0.0
        Re_G    = 0.0
        Q_G     = 0.0
        Nu_L, Re_L, alpha_L, K_L, T_AL, Q_L = convection(Geometry, physics, phase_1)
    elif (vof == 1.0):
        Nu_L    = 0.0
        alpha_L = 0.0
        Re_L    = 0.0
        Q_L     = 0.0
        Nu_G, Re_G, alpha_G, K_G, T_AG, Q_G = convection(Geometry, physics, phase_2)

    else:
        # > Angle Gamma
        gamma = angle(vof)

        # > hydraulic diameter
        A_L = (1 - vof) * cross_section
        S_I = di * sqrt((1 - cos(gamma)) / 2.0)
        S_L = (pi - gamma / 2) * di

        # > Hydraulic diameter
        '''Flow and heat transfer for a two-phase slug flow in horizontal pipes: A mechanistic model'''
        D_L = 4.0 * A_L / (S_L + S_I)
        '''Mechanistic modeling of flow and heat transfer in turbulent–laminar/turbulent Gas–Liquid stratified flow'''
        D_L = 4.0 * A_L / S_L  # Taitel and Dukler(1976).
        heat_area_L = (pi - gamma / 2.0) * heat_area
        A_G = vof * cross_section
        S_G = 0.5 * gamma * di
        D_G = 4.0 * A_G / (S_G + S_I)
        # > first guess of outlet temperatures of the separate phases
        T_AL = T_EL
        T_AG = T_EG
        while True and (vof > 0.0) and (vof < 1.0):
            T_AG_old = T_AG
            T_AL_old = T_AL
            # update fluid temperatures of the phases
            T_L = 0.5 * (T_EL + T_AL)
            T_G = 0.5 * (T_EG + T_AG)
            # thermodynamic properties of the mixture
            tp_model = two_phase_model(phase_1=phase_1, phase_2= phase_2, T_FL=T_L, T_FG=T_G, pressure=P,
                                            vof=vof, mass_flow=mass_flow_rate, const_flow_rate=const_flow_rate, cross_section=cross_section, assumption=assumption, slip_ratio_model=slip_ratio_model)



            #print("U=", U)
            # mass flow rates of two phases
            x   = tp_model.flow_quality
            s   = tp_model.slip_ratio
            m_L = tp_model.m_L
            m_G = tp_model.m_G
            # Liquid phase
            Liquid   = Fluid(T_L, P, phase_1)
            lambda_L = Liquid.k
            rho_L    = Liquid.rho
            nu_L     = Liquid.nuu
            Pr_L     = Liquid.Pr
            Cp_L     = Liquid.Cp
            # velocity U in [m/s]
            U_L      = m_L / ((1.0 -vof) * cross_section * rho_L )
            # > reynolds-number
            Re_L     = U_L * D_L / nu_L
            # > Nusselt function
            Nu_L     = Nusselt(Pr_L, Re_L, flow, Geometry)
            # > convective heat transfer coefficient
            alpha_L   = Nu_L * lambda_L / D_L
            # > Gas phase
            Gas       = Fluid(T_G, P, phase_2)
            rho_G     = Gas.rho
            nu_G      = Gas.nuu
            lambda_G  = Gas.k
            Pr_G      = Gas.Pr
            Cp_G      = Gas.Cp

            # velocity U in [m/s]
            U_G = m_G / (vof * cross_section * rho_G)
            # > reynolds-number
            # > Reynolds-number
            Re_G      = U_G * D_G / nu_G
            Nu_G      = Nusselt(Pr_G, Re_G, flow, Geometry)
            #print(Nu_G)
            # > convective heat transfer alpha_i_G
            lambda_G  = Gas.k
            alpha_G   = Nu_G * lambda_G / D_G
            # > alpha
            # > heat transfer coefficient HTC of the Liquid and the Gas phases
            K_L       = 1 / (1 / alpha_L + di / (2 * lambda_R) * log(da / di))
            K_G       = 1 / (1 / alpha_G + di / (2 * lambda_R) * log(da / di))
            # > calculate outlet temperatures

            data = (m_L, Cp_L, T_EL, K_L, alpha_L, S_L, T_L, \
                    m_G, Cp_G, T_EG, K_G, alpha_G, S_G, T_G, \
                    S_I, L, T_C)

            #if( vof <= 0.2):
                #T_AG, T_AL = fsolve(initial_guess, (T_EG * 0.9, T_EL * 0.8), args=data)
            #else:
               #T_AG_guess, T_AL_guess = fsolve(initial_guess, (T_EG * 0.9, T_EL * 0.8), args=data, xtol=1e-03, maxfev=500)
               #T_AG, T_AL             = fsolve(initial_guess, (T_AG_guess, T_AL_guess), args=data, xtol=1e-03, maxfev=500)
            Y_G       = m_G * Cp_G / (K_G * S_G * L)
            Y_L       = m_L * Cp_L / (K_L * S_L * L)

            T_AL_init = (Y_L - 0.5) / (Y_L + 0.5) * T_EL + T_C / (Y_L + 0.5)
            T_AG_init = (Y_G - 0.5) / (Y_G + 0.5) * T_EG + T_C / (Y_G + 0.5)

            T_AG, T_AL           = fsolve(energy_equations, (T_AG_init, T_AL_init),    args=data, xtol=1e-04, maxfev=500)
            #T_AG, T_AL = T_AG_init, T_AL_init

            # > residual
            error1 = abs(T_AL - T_AL_old)
            error2 = abs(T_AG - T_AG_old)
            #print(T_AG-T_AG_init, T_AL - T_AL_init)
            #print(error1)
            #print(error2)
            if ((error1 < 10 ** -4) and (error2 < 10 ** -4)) :
                #print(error1, error2) #debug
                break
        # > calculate heat flux
        Q_G = K_G * S_G * L * abs(0.5 * (T_EG + T_AG) - T_C)
        #Q_I = alpha_G * S_I * L * 0.5 * abs((T_EG + T_AG) - (T_EL + T_AL))
        #print("Q_G+Q_I", Q_G + Q_I)
        #print(Q_G)
        #Q_L = m_L * Cp_L * (T_EL - T_AL)
        Q_L = K_L * S_L * L * (0.5 * (T_EL + T_AL) - T_C)
    # > total heat flux
    Q = Q_L + Q_G

    return  Nu_L, Re_L, alpha_L, K_L, T_AL, Q_L, Nu_G, Re_G, alpha_G, K_G, T_AG, Q_G

#TODO  > extended function to solve the convective heat transfer problem with
# Nusselt-functions for a Two Phase Flow based on correlation for two phase heat transfer coefficient
# Correlation of Kim & Ghajar doi:10.1016/j.ijmultiphaseflow.2006.01.002
def TP_Correlation_Ghajar(Geometry, physics, phase_1, phase_2, assumption, const_flow_rate, slip_ratio_model):

    # Geometry
    di = Geometry.inner_diameter
    da = Geometry.outer_diameter

    cross_section = Geometry.cross_section()
    heat_area     = Geometry.heat_area()
    lambda_R      = Geometry.conductivity()

    # Physics
    T_E = physics.T_EL
    P   = physics.pressure
    T_C = physics.cooling_temperature
    mass_flow_rate = physics.mass_flow
    T_A = T_E
    flow = physics.flow_form
    T_W  = T_E
    while True:
        T_A_old = T_A
        # update fluid temperature
        T_F = 0.5 * (T_E + T_A)
        # volume of fraction
        vof = physics.vof
        # thermodynamic properties of the mixture
        tp_model = two_phase_model(phase_1=phase_1, phase_2= phase_2, T_FL=T_F, T_FG=T_F, pressure=P,
                                            vof=vof, mass_flow=mass_flow_rate, const_flow_rate=const_flow_rate, cross_section=cross_section, assumption=assumption, slip_ratio_model=slip_ratio_model)

        density_m = tp_model.rho   # update density of the mixture in [kg/m^3]
        mu_m      = tp_model.muu   # update dynamic viscosity of the mixture
        nu_m      = tp_model.nu    # update kinematic viscosity
        cp_m      = tp_model.Cp    # update heat capacity
        lambda_m  = tp_model.k     # update conductivity
        Pr_m      = tp_model.Pr    # update Prandtl-number
        # Liquid velocity U in [m/s]
        Liquid    = Fluid(T_F, P, phase_1)
        rho_L     = Liquid.rho
        mu_L      = Liquid.muu
        mu_W      = Fluid(T_W, P, phase_1).muu
        Pr_L      = Liquid.Pr
        lambda_L  = Liquid.k
        m_L       = tp_model.m_L
        U_L       = m_L / (rho_L * cross_section * (1.0 - vof))

        # Gas velocity U in [m/s]
        Gas      = Fluid(T_F, P, phase_2)
        rho_G    = Gas.rho
        mu_G     = Gas.muu
        Pr_G     = Gas.Pr
        lambda_G = Gas.k
        m_G      = tp_model.m_G
        if (vof == 0):
            U_G  = 0.
        else:
            U_G      = m_G / (rho_G * cross_section * vof)
        #print(nu_m)
        # reynolds-number
        Re_L      = m_L / (pi / 4.0 * sqrt(1 - vof) * mu_L * di)
        # Nusselt function
        Nu        = Nusselt(Pr_L, Re_L, flow, Geometry)
        # convective heat transfer coefficient
        alpha_L   = Nu * lambda_L / di
        alpha_L   = 0.027 * Re_L ** (4./5.) * Pr_L ** (1. / 3.) * lambda_L / di * (mu_L / mu_W) ** 0.14 # Ghajar & al. 2006
        # heat transfer coefficient HTC
        h_L       = 1.0 / (1 / alpha_L + di / (2 * lambda_R) * log(da / di))
        # two phase HTC Correlation I (Ghajar 2006)
        C         =  0.7
        r         =  0.08
        n         =  0.06
        p         =  0.03
        q         = -0.14
        # two phase HTC Correlation I (stratified flow) https://www.journal4research.org/articles/J4RV3I11006.pdf
        #C = 9.786
        #r = -2.18
        #n = 5.288
        #p = -12.9
        #q = 11.24
        #h_TP = (1.0 - vof) * h_L * (1.0 + C * ((x / (1.0 - x)) ** r * (vof / (1.0 - vof)) ** n * (Pr_G / Pr_L) ** p * (mu_G / mu_L) ** q))

        g         =  9.81     # gravity in m/s^2
        F_s       = 2. / pi  * arctan(sqrt(rho_G * (U_G - U_L) ** 2.0 / (g * di * (rho_L - rho_G)))) # shape factor
        #print(F_s)
        F_p       = (1.0 - vof)  + vof * F_s ** 2.0  # flow pattern factor
        x         = tp_model.flow_quality
        #h_TP      = F_p * h_L * (1.0 + C * ((x / (1.0 - x)) ** r * ((1.0 - F_p) / F_p ) ** n * (Pr_G / Pr_L) ** p * (mu_G / mu_L) ** q))
        # two phase HTC Correlation II (Ghajar 2002)
        #alpha_TP  = F_p * h_L * (1.0 + C * ((x / (1.0 - x)) ** r * ((1.0 - F_p) / F_p ) ** n * (Pr_G / Pr_L) ** p * (mu_G / mu_L) ** q))
        # two phase HTC Correlation II
        C         =  2.86
        r         =  0.42
        n         =  0.35
        p         =  0.66
        q         = -0.72
        
        # TP-HTC Corrleation   # Ghajar 2002
        h_TP  = (1.0 - vof) * h_L * (1.0 + C * ((x / (1.0 - x)) ** r * (vof / (1.0 - vof)) ** n * (Pr_G / Pr_L) ** p * (mu_G / mu_L) ** q))
        ### test
        # TP-HTC Corrleation   # Ghajar 2002
        #alpha_TP  = (1.0 - vof) * alpha_L * (1.0 + C * ((x / (1.0 - x)) ** r * (vof / (1.0 - vof)) ** n * (Pr_G / Pr_L) ** p * (mu_G / mu_L) ** q))
        #h_TP      = 1.0 / (1 / alpha_TP + di / (2 * lambda_R) * log(da / di))
        ### test 
        # calculate outlet temperature
        m_tp = tp_model.mass_flow  # Two Phase mass flow
        T_A  = T_C + (T_E - T_C) * exp(- h_TP * heat_area / (m_tp * cp_m))
        # LMTD
        LMTD = (T_A - T_E) / log((T_A - T_C) / (T_E - T_C))
        # calculate heat flux
        Q    = h_TP * heat_area * LMTD
        # calculate enthalpy stream change
        dH   = m_tp * cp_m * (T_E - T_A)
        # update Wall temperature T_W
        T_W  = Q / (alpha_L * heat_area) + T_F
        # residual
        error = abs(T_A - T_A_old)
        #print("error:", error)
        if (error < 10 ** -3):
            heat_flux = Q
            break
        Re = tp_model.velocity * di / tp_model.nu
    return Nu, Re, alpha_L, h_TP, T_A, heat_flux


# Correlation of Sims et al.
def TP_Correlation_Sims(Geometry, physics, phase_1, phase_2, assumption, const_flow_rate, slip_ratio_model):

    # Geometry
    di = Geometry.inner_diameter
    da = Geometry.outer_diameter

    cross_section = Geometry.cross_section()
    heat_area     = Geometry.heat_area()
    lambda_R      = Geometry.conductivity()

    # Physics
    T_E = physics.T_EL
    P   = physics.pressure
    T_C = physics.cooling_temperature
    mass_flow_rate = physics.mass_flow
    T_A = T_E
    flow = physics.flow_form
    T_W  = T_E
    while True:
        T_A_old = T_A
        # update fluid temperature
        T_F = 0.5 * (T_E + T_A)
        # volume of fraction
        vof = physics.vof
        # thermodynamic properties of the mixture
        tp_model = two_phase_model(phase_1=phase_1, phase_2= phase_2, T_FL=T_F, T_FG=T_F, pressure=P,
                                            vof=vof, mass_flow=mass_flow_rate, const_flow_rate=const_flow_rate, cross_section=cross_section, assumption=assumption, slip_ratio_model=slip_ratio_model)

        density_m = tp_model.rho   # update density of the mixture in [kg/m^3]
        mu_m      = tp_model.muu   # update dynamic viscosity of the mixture
        nu_m      = tp_model.nu    # update kinematic viscosity
        cp_m      = tp_model.Cp    # update heat capacity
        lambda_m  = tp_model.k     # update conductivity
        Pr_m      = tp_model.Pr    # update Prandtl-number
        # Liquid velocity U in [m/s]
        Liquid    = Fluid(T_F, P, phase_1)
        rho_L     = Liquid.rho
        mu_L      = Liquid.muu
        mu_W      = Fluid(T_W, P, phase_1).muu
        Pr_L      = Liquid.Pr
        lambda_L  = Liquid.k
        m_L       = tp_model.m_L
        U_L       = m_L / (rho_L * cross_section * (1.0 - vof))

        # Gas velocity U in [m/s]
        Gas      = Fluid(T_F, P, phase_2)
        rho_G    = Gas.rho
        mu_G     = Gas.muu
        Pr_G     = Gas.Pr
        lambda_G = Gas.k
        m_G      = tp_model.m_G
        if (vof == 0):
            U_G  = 0.
        else:
            U_G   = m_G / (rho_G * cross_section * vof)
        # reynolds-number
        Re_L      = m_L / (pi / 4.0 * sqrt(1 - vof) * mu_L * di)
        Re_L      = m_L / (pi / 4.0 * sqrt(1 -   0) * mu_L * di)  #TODO Martin and Sims which Re
        # Nusselt function
        Nu        = Nusselt(Pr_L, Re_L, flow, Geometry)
        # convective heat transfer coefficient
        alpha_L   = Nu * lambda_L / di
        alpha_L   = 0.027 * Re_L ** (4./5.) * Pr_L ** (1. / 3.) * lambda_L / di * (mu_L / mu_W) ** 0.14 # Ghajar & al. 2006
        # heat transfer coefficient HTC
        h_L       = 1 / (1 / alpha_L + di / (2 * lambda_R) * log(da / di))
        # TP-HTC Correlation Martin & Sims
        U_SG = m_G / (rho_G * cross_section)
        U_SL = m_L / (rho_L * cross_section)
        h_TP  = h_L * (1.0 + 0.64 * sqrt(U_SG / U_SL)) 

        # calculate outlet temperature
        m_tp = tp_model.mass_flow  # Two Phase mass flow
        T_A  = T_C + (T_E - T_C) * exp(- h_TP * heat_area / (m_tp * cp_m))
        # LMTD
        LMTD = (T_A - T_E) / log((T_A - T_C) / (T_E - T_C))
        # calculate heat flux
        Q    = h_TP * heat_area * LMTD
        # calculate enthalpy stream change
        dH   = m_tp * cp_m * (T_E - T_A)
        # update Wall temperature T_W
        T_W  = Q / (alpha_L * heat_area) + T_F
        # residual
        error = abs(T_A - T_A_old)
        #print("error:", error)
        if (error < 10 ** -3):
            heat_flux = Q
            break
        Re = tp_model.velocity * di / tp_model.nu
    return Nu, Re, alpha_L, h_TP, T_A, heat_flux


# slug characteristics  (#monta) todo --> define slug characteristics as function (done)
def slug_characteristics(d, e, Liquid, Gas, U_sl, U_sg):
    '''
    An equilibrium and constant film thickness is assumed along the entire film zone (Eq. 3.141). For this case hF = const. = hE but −dp/dz ≠ 0.
     > d: diameter of the pipe
     > e: roughness height of the pipe
     > Liquid: Liquid phase
     > Gas: Gas phase
     > U_sl: supericial velocity of liquid phase
     > U_sg: superficial velocity of gas phase
     > Lf: liquid film length
    '''
    # Geometry parameters
    A = pi / 4. * d ** 2.0
    # thermodynamic properties
    rho_l = Liquid.rho
    mu_l = Liquid.muu
    rho_g = Gas.rho
    mu_g = Gas.muu
    # gravity g
    g = 9.81  # [m/s2]
    # step 00: mixture velocity v_m
    v_m = U_sl + U_sg
    # step 01: vTB, vGLS, HLLS and Ls from closure relationships
    C0 = 1.2
    Re_LS = rho_l * v_m * d / mu_l  # liquid slug reynolds number
    # closure relationships
    LS = 32. * d
    HLLS = 1.0 * exp(- 2.48 * 10 ** (-6) * Re_LS)  # slug liquid holdup (inclination = 0°) Gomez & al. 2000
    # print("HLLS=",HLLS)
    # HLLS  = 1.0 / (1.0 + (v_m / 8.66) ** 1.39)                        # //    //     //          //           Gregory & al. 1975
    # print("HLLS=",HLLS)
    vTB = C0 * v_m + 0.54 * sqrt(g * d)  # translational velocity
    vGLS = C0 * v_m  # Gas-Pocket velocity

    # print(vTB)
    # step 03: vLLS
    vLLS = 1.0 / HLLS * (v_m - vGLS * (1.0 - HLLS))

    # Initialize HLTB, SI, SL and SG
    hF = 0.1 * d
    HLTB = 1.0 / pi * (pi - arccos(2.0 * hF / d - 1.0) + (2.0 * hF / d - 1.0) * sqrt(
        1.0 - (2.0 * hF / d - 1.0) ** 2.0))  # eq. 3.38
    # print(HLTB)
    S_I = d * sqrt(1.0 - (2.0 * hF / d - 1.0) ** 2.0)  #
    S_L = d * (pi - arccos(2.0 * hF / d - 1.0))
    S_G = pi * d - S_L
    dF = 4.0 * HLTB * A / S_L
    dG = 4.0 * (1.0 - HLTB) * A / (S_G + S_I)

    # initialize velocities vF and vG
    vLTB = vTB - HLLS / HLTB * (vTB - vLLS)
    vGTB = vTB - (1.0 - HLLS) / (1.0 - HLTB) * (vTB - vGLS)
    vF = vTB - vLTB
    vG = vTB - vGTB
    Re_F = rho_l * abs(vF) * dF / mu_l
    Re_G = rho_g * abs(vG) * dG / mu_g
    # initialize friction coefficients
    f_L = friction_factor(Re_F, e, d, 0.001)
    f_G = friction_factor(Re_G, e, d, 0.001)
    # Determination of the interfacial friction factor, fI, is more complex.
    # For the case of low liquid and gas velocities, the smooth-interface friction factor can be used, namely, fI = fG.
    # For wavy interface, a constant value of fI = 0.014, which was suggested for stratified-wavy flow, can be used.
    f_I = 0.0142

    while True:
        vF_old = vF
        vG_old = vG
        # print(vG, vF)
        # step 02: determine hF numerically
        x0 = hF
        data = (f_L, f_G, f_I, rho_l, rho_g, vLLS, vGLS, HLLS, vTB, d)

        # always use *data
        def F(x, *data):
            '''
                :param vars:
                        x = hF
                :param data:
                        parameters are in the list data stored
                :return:
                        eq. 5.40 force balance
                '''
            f_L, f_G, f_I, rho_l, rho_g, vLLS, vGLS, HLLS, vTB, d = data
            # > HLTB, S_L & S_I
            Y = 2.0 * x / d - 1.0
            # print("Y ",Y)
            HLTB = 1.0 / pi * (pi - arccos(Y) + Y * sqrt(1 - Y ** 2.0))  # eq. 3.38
            vLTB = vTB - HLLS / HLTB * (vTB - vLLS)
            vGTB = vTB - (1.0 - HLLS) / (1.0 - HLTB) * (vTB - vGLS)
            # print("HLTB",HLTB)
            S_I = d * sqrt(1.0 - Y ** 2.0)  #
            S_L = d * (pi - arccos(Y))
            vGTB
            res = f_L * rho_l * S_L / HLTB * abs(vLTB) * vLTB \
                  - f_G * rho_g * S_G / (1.0 - HLTB) * abs(vGTB) * vGTB \
                  - f_I * rho_g * S_I / (HLTB * (1.0 - HLTB)) * abs(vGTB - vLTB) * (vGTB - vLTB)

            return res

        result = my_bisection(F, data, 0.0001 * d, 0.99 * d, 1.e-5)
        hF = result
        # print("result",result)
        # print("hF",hF)
        # determine HLTB and vLTB
        HLTB = 1.0 / pi * (pi - arccos(2.0 * hF / d - 1.0) + (2.0 * hF / d - 1.0) * sqrt(
            1 - (2.0 * hF / d - 1.0) ** 2.0))  # eq. 3.38
        # print(HLTB)
        S_I = d * sqrt(1.0 - (2.0 * hF / d - 1.0) ** 2.0)  #
        S_L = d * (pi - arccos(2.0 * hF / d - 1.0))
        S_G = pi * d - S_L

        # step 04: vLTB and vGTB from Eqs. 5.47 and 5.48, respectively
        vLTB = vTB - HLLS / HLTB * (vTB - vLLS)
        vGTB = vTB - (1.0 - HLLS) / (1.0 - HLTB) * (vTB - vGLS)
        # vF and vG
        vF = vTB - vLTB
        vG = vTB - vGTB
        # characteristic length
        dF = 4.0 * HLTB * A / S_L
        dG = 4.0 * (1.0 - HLTB) * A / (S_G + S_I)
        Re_F = rho_l * abs(vF) * dF / mu_l
        Re_G = rho_g * abs(vG) * dG / mu_g
        # rough pipe
        f_L = friction_factor(Re_F, e, d, 0.001)
        f_G = friction_factor(Re_G, e, d, 0.001)
        # Determination of the interfacial friction factor, fI, is more complex.
        # For the case of low liquid and gas velocities, the smooth-interface friction factor can be used, namely, fI = fG.
        # For wavy interface, a constant value of fI = 0.014, which was suggested for stratified-wavy flow, can be used.
        f_I = 0.0142

        # calculate tau_F, tau_G and tau_I from eq. 3.131 through 3.133
        tau_F = f_L * rho_l * abs(vLTB) * vLTB / 2. * S_L / HLTB
        tau_G = f_G * rho_g * abs(vGTB) * vGTB / 2. * S_G / (1.0 - HLTB)
        tau_I = f_I * rho_g * abs(vGTB - vLTB) * (vGTB - vLTB) / 2.0 * S_I / (HLTB * (1.0 - HLTB))
        converg = tau_F - tau_G - tau_I
        # error
        error1 = abs(vG - vG_old)
        error2 = abs(vF - vF_old)

        if (error1 < tol and error2 < tol):
            # print("hF=", hF)
            # print("errors",error1, error2)
            # print("convergence", converg)
            # print(F(hF, *data))
            break

    # step
    # liquid film length LF
    LU = LS * (vLLS * HLLS - vLTB * HLTB) / (U_sl - vLTB * HLTB)
    print((vLLS * HLLS - vLTB * HLTB) / (U_sl - vLTB * HLTB))
    LF = LS * (vLLS * HLLS - U_sl) / (U_sl - vLLS * HLLS + vTB * (HLLS - HLTB))
    # LF = LU - LS
    return LF

