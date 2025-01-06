#! /usr/bin/env python3
'''===================================================================================================================
 - Author: M. Guesmi
 - Adress: Chair of process engineering TU Dresden, Helmholtzstraße 10, 01069 Dresden, Germany
 - Summary: Define thermodynamic properties class
======================================================================================================================'''
import CoolProp.CoolProp as cp

class Fluid(object):
    """
          This class contains the thermodynamics properties of a fluid
          Args:
                  T (in °C) Temperature 0 bis 99
                  p (in bar) pressure
          Returns:
                  roh (in Kg/m3) water density
                  c_p (in J/KGK) Specific heat capacity
                  lambda (in W/mK) water thermal conductivity
                  nu (in m2/s) water kinematic viscosity
                  Pr (-) water prandtl number

      """
    def __init__(self, T, P, media):
        self.T = T
        self.P = P
        self.media = media
        self.k   = cp.PropsSI('L', 'P', P*10**5, 'T', T + 273.15, media)
        self.Cp  = cp.PropsSI('C', 'P', P*10**5, 'T', T + 273.15, media)
        self.rho = cp.PropsSI('D', 'P', P*10**5, 'T', T + 273.15, media)
        self.muu = cp.PropsSI('V', 'P', P*10**5, 'T', T + 273.15, media)
        self.nuu = self.muu/self.rho
        self.Pr = cp.PropsSI('Prandtl', 'P', P*10**5, 'T', T + 273.15, media)


