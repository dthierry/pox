#!/usr/bin/env python
# coding: utf-8

import pyomo.environ as pe
import pyomo.dae as dae
import numpy as np
import sys

import pandas as pd
import matplotlib.pyplot as plt
from itertools import product


# Optimization solver
# ipexe = "/Users/dthierry/Apps/ipopt_102623/ip_dir/bin/ipopt"
ipexe = "/Users/dthierry/Apps/ipopt_dir/bin/ipopt"
solver = pe.SolverFactory(ipexe)

# # Parameters and data:

# constraints for reactions:
A_rxn = {1: 4.225e15,  # kmol bar^0.5 kgcat^-1 h^-1
         2: 1.955e6,  # kmol bar^-1 kgcat^-1 h^-1
         3: 1.020e15  # kmol bar^0.5 kgcat^-1 h^-1
         }
Ak1 = 4.255e15   # pre-exponential factor of rate constant of reaction 1
Ak2 = 1.955e6    # pre-exponential factor of rate constant of reaction 2
Ak3 = 1.02e15    # pre-exponential factor of rate constant of reaction 3

E1 = 240.1       # kJ/mol activation energy of reaction 1
E2 = 67.13       # kJ/mol activation energy of reaction 2
E3 = 243.9       # kJ/mol activation energy of reaction 3


R_g = 0.0083144621  # kJ/(K*mol) gas constant

# A_eq ("R_1"): 4.707e12;//bar^2
# A_eq ("R_2"): 1.142e-2;//[dimensionless]
# A_eq ("R_3"): 5.375e10;//bar^2

A_eq = {1: 4.707e12,  # bar^2
        2: 1.142e-2,  # []
        3: 5.375e10}  # bar^2, [], bar^2

# dhr in kJ/kmol
dHr = {1: 206.1,
       2: -41.15,
       3: 164.9}
# pre-exponential factor of adsorption constant
AKa = {'CH4': 6.65e-4,  # bar^-1
       'H2O': 1.77e5,  #
       'H2': 6.121e-9,  # bar^-1
       'CO': 8.23e-5,  # bar^-1
       "N2": 0e0,
       "CO2": 0e0
       }

dHa = {'CH4': -38.28,
       'H2O': 88.68,
       'H2': -82.90,
       'CO': -70.65,
       'N2': 0e0,
       'CO2': 0e0} # adsorption kJ/mol # enthalpy

d_p = 0.0173  # m
d_t_in = 0.1016  # m
d_t_out = 0.1322  # m

rho_p = 2355.2  # kg m^-3
eta_1 = 0.001
eta_2 = 0.001
eta_3 = 0.001
eta = {1: eta_1, 2: eta_2, 3: eta_3}
CP_p = 950 # J kg^-1 K^-1
lambda_p = 0.3489  # W m^-1 K^-1
Tw = 1000  # K

eb = 0.38 + 0.073 * (1-(d_t_in/d_p-2)**2/(d_t_in/d_p)**2);


# @dav: radius = 0.03
# radius = 0.03/2                                   # m  radius
radius = 0.1016/2

Ac = np.pi * radius**2                          # m^2 cross sectional area
rho_cata = 1820                                 # kg/m3 catalyst density
d_cata = 5e-3                                   # m  catalyst diameter
mu_ave = 2.97e-5                                # Pa*s  gas average viscosity

# @dav: Cp_ave should have J/kg-K units
Cp_ave = 41.06 # J/(mol*K) gas mixture specific heat

rho_g = 1.6  # kg m^-3, gas density

# # Define variables and parameters

model = m = pe.ConcreteModel()

# model reactor parameters
model.L = pe.Param(initialize=12)  # reactor total length
model.R = pe.Param(initialize=2)  # reactor radius
model.z = dae.ContinuousSet(bounds=(0,model.L))   # axial direction
model.r = dae.ContinuousSet(bounds=(0,model.R))   # radial direction

model.SPECIES = pe.Set(initialize=['CH4', 'H2O', 'H2', 'CO', 'CO2', 'N2'])
model.REACTIONS = pe.RangeSet(3)

model.P0 = pe.Param(initialize = 10., mutable=True) # inlet pressure
model.T0 = pe.Param(initialize = 793.15) # inlet temperature
model.C_in = pe.Param(model.SPECIES, mutable=True, initialize=1.0)

# define reactor inlet variables

model.FCH4_in = pe.Param(initialize = 30)    # fix inlet CH4

model.F_in = pe.Var(model.SPECIES, bounds=(0,None))   # inlet flow rate kmol/h

model.Ft_in = pe.Var(bounds = (0,None))   # total inlet flow rate kmol/h
model.X_in = pe.Var(model.SPECIES)   # inlet mole fraction
# shouldn't this be below 1

# in m/s
model.u = pe.Var(bounds = (0,None), initialize = 4.5) # superfacial velocity
model.Rep = pe.Var(bounds = (0,None))   # practical Raynolds number
model.f = pe.Var(bounds = (0,None)) # friction factor

# define variables that vary along reactor length

model.F = pe.Var(model.z, model.r, model.SPECIES, bounds = (0,None))
model.C = pe.Var(model.z, model.r, model.SPECIES, bounds = (0,None)) # kmol/m^3

model.X = pe.Var(model.z, model.r, model.SPECIES)

model.y = pe.Var(model.z, model.r, bounds = (0, None))
model.Pt = pe.Var(model.z, model.r, bounds = (0, None))
model.T = pe.Var(model.z, model.r, bounds = (1e-06, None), initialize=m.T0)
model.Ft = pe.Var(model.z, model.r, bounds = (0, None))
model.P = pe.Var(model.z, model.r, model.SPECIES, bounds = (0,None), initialize=1e0)
model.Rate = pe.Var(model.z, model.r, model.REACTIONS)

model.r_comp = pe.Var(model.z, model.r, model.SPECIES)

model.DEN = pe.Var(model.z, model.r, initialize=1e0)

# define coeffs
model.k1 = pe.Var(model.z, model.r, bounds=(0, None))
model.k2 = pe.Var(model.z, model.r)
model.k3 = pe.Var(model.z, model.r, bounds=(0, None))


model.Ke = pe.Var(model.z, model.r, model.REACTIONS,
                  bounds=(1e-08, None), initialize=1e3)

model.Ka = pe.Var(model.z, model.r, model.SPECIES, bounds=(0e-00, None), initialize=1e0)

# define derivative variables
model.dCz = dae.DerivativeVar(model.C, wrt=model.z) # 1
model.dCr = dae.DerivativeVar(model.C, wrt=model.r) # 2

model.dy = dae.DerivativeVar(model.y, wrt=model.z) # 3

model.dTz = dae.DerivativeVar(model.T, wrt=model.z) # 4
model.dTr = dae.DerivativeVar(model.T, wrt=model.r) # 5


# # Constraints
# ## Reaction coefficients equations

# kmol*bar^(1/2)/(kgcat*hr) rate constant of reaction 1
def defk1_rule(m, z, r):
    return m.k1[z, r] == A_rxn[1]*pe.exp(-E1/(R_g*m.T[z, r]))
model.defk1 = pe.Constraint(model.z, model.r, rule = defk1_rule)

# kmol/(bar* kgcat*hr) rate constant of reaction 2
def defk2_rule(m, z, r):
    return m.k2[z, r] == A_rxn[2]*pe.exp(-E2/(R_g*m.T[z, r]))
model.defk2 = pe.Constraint(model.z, model.r, rule = defk2_rule)

# kmol*bar^(1/2)/(kgcat*hr) rate constant of reaction 3
def defk3_rule(m, z, r):
    return m.k3[z, r] == A_rxn[3]*pe.exp(-E3/(R_g*m.T[z, r]))
model.defk3 = pe.Constraint(model.z, model.r, rule = defk3_rule)

# rate equilibrium constant
def defKe_rule(m, z, r, k):
    return pe.log(m.Ke[z, r, k] / A_eq[k]) == (-dHr[k]/(R_g* m.T[z, r]))
model.defKe = pe.Constraint(model.z, model.r, model.REACTIONS,
                            rule=defKe_rule)


model.logKa = pe.Var(model.z, model.r, model.SPECIES, initialize=1e-3)

# adsorption constant for each species
def deflogKa_rule(m, z, r, s):
    if s == 'CO2' or s == 'N2':
        return pe.Constraint.Skip
    #return pe.log(m.Ka[z, r, s] / AKa[s])== (-dHa[s]/(R_g*m.T[z, r]))
    return m.logKa[z, r, s] == (-dHa[s]/(R_g*m.T[z, r]))

model.deflogKa = pe.Constraint(model.z, model.r, model.SPECIES, rule=deflogKa_rule)

# adsorption constant for each species
def defKa_rule(m, z, r, s):
    if s == 'CO2' or s == 'N2':
        return pe.Constraint.Skip
    #return pe.log(m.Ka[z, r, s] / AKa[s])== (-dHa[s]/(R_g*m.T[z, r]))
    return m.Ka[z, r, s]/AKa[s] == pe.exp(m.logKa[z, r, s])

model.defKa = pe.Constraint(model.z, model.r, model.SPECIES, rule=defKa_rule)

# ## Inlet Condition Constraints

# define inlet mole fractions:

# define velocity
def Def_Ft_in_rule(m):
    return m.Ft_in == sum(m.F_in[s] for s in model.SPECIES)
model.Def_Ft_in = pe.Constraint(rule = Def_Ft_in_rule)


def Flow_fraction_relation_rule(m,s):
    return m.F_in[s] == m.Ft_in * m.X_in[s]
model.Flow_fraction_relation = pe.Constraint(model.SPECIES,
                                             rule=Flow_fraction_relation_rule)
 # u is in m/s
 # Ft is in kmol/h -> mol/s
 # P is in bar
 # R_g is in kJ/mol-K -> J/mol-K
def IdealGasLaw_in_rule(m):
    return m.Ft_in*(1000/3600)*R_g*1000*m.T0 == \
        m.P0*1e5*m.u*(3.14*d_t_in**2/4)
#model.IdealGasLaw_in = pe.Constraint(rule = IdealGasLaw_in_rule)

def Def_Rep_rule(m):            # partical Raynolds number
    return m.Rep == m.u * rho_g * d_cata / mu_ave
model.Def_Rep= pe.Constraint(rule = Def_Rep_rule)

def Def_friction_rule(m):  # friction factor
    return m.f == 150 + 1.75*m.Rep/(1-eb)
model.Def_friction= pe.Constraint(rule = Def_friction_rule)


# ## ODE Constraints

# ODEs Constraints

# total pressure of pcl = nondementionalized pressure of
#pcl * Initial pressure of pcl, here y = P0/P
def Pressure_nondim_rule(m, z, r):
    return m.Pt[z, r] == m.y[z, r] * m.P0
model.Pressure_nondimensionalize = pe.Constraint(model.z, model.r,
                                                 rule=Pressure_nondim_rule)

def Total_Flow_rule(m,z, r): # total flow rate is sum of single species flow rate
    return m.Ft[z, r] == sum(m.F[z, r, s] for s in model.SPECIES)
model.Total_Flow = pe.Constraint(model.z, model.r, rule = Total_Flow_rule)

def Patial_pressure_rule(m, z, r, s):  # patial pressure
    #return m.P[z, r,s] == m.F[z,r,s] / m.Ft[z, r] * m.Pt[z, r]
    return m.P[z, r, s] * m.Ft[z, r] == m.F[z, r,s] * m.Pt[z, r]
model.Patial_pressure = pe.Constraint(model.z, model.r, model.SPECIES,
                                      rule=Patial_pressure_rule)
# define DEN (for rate functions)
def Def_DEN_rule(m, z, r):
    return m.DEN[z, r] * m.P[z, r,'H2'] == m.P[z, r,'H2'] \
    + m.Ka[z, r,'CO']*m.P[z, r,'CO'] * m.P[z, r,'H2'] \
    + m.Ka[z, r,'CH4']*m.P[z, r, 'CH4'] * m.P[z, r, 'H2'] \
    + m.Ka[z, r, 'H2']*m.P[z, r, 'H2'] * m.P[z, r, 'H2'] \
    + m.Ka[z, r, 'H2O']*m.P[z, r, 'H2O']
model.Def_DEN = pe.Constraint(model.z, model.r, rule=Def_DEN_rule)

# kmol/(kgcat*s) rate law for reaction 1
def Def_Rate1_rule(m,z, r):
    return m.Rate[z, r, 1] == \
        (m.k1[z, r] / (pow(m.P[z, r,'H2'],2.5)*m.DEN[z, r]**2)) \
        *(m.P[z, r,'CH4']*m.P[z, r,'H2O'] \
           - pow(m.P[z, r,'H2'], 3) * m.P[z, r,'CO']/m.Ke[z, r, 1])*(1/3600)

model.Def_Rate1 = pe.Constraint(model.z, model.r, rule=Def_Rate1_rule)

# kmol/(kgcat*s) rate law for reaction 2
def Def_Rate2_rule(m, z, r):
    return m.Rate[z, r, 2] == \
        (m.k2[z, r] / (m.P[z, r, 'H2'] * m.DEN[z, r]**2)) \
        *(m.P[z, r,'CO']*m.P[z, r,'H2O'] \
          - m.P[z, r,'H2'] * m.P[z, r,'CO2']/m.Ke[z, r, 2])*(1/3600)

model.Def_Rate2 = pe.Constraint(model.z, model.r, rule=Def_Rate2_rule)

# kmol/(kgcat*s) rate law for reaction 1
def Def_Rate3_rule(m, z, r):
    return m.Rate[z, r,3] == \
        (m.k3[z, r] / (pow(m.P[z, r,'H2'],3.5) * m.DEN[z, r]**2)) \
        *(m.P[z, r,'CH4']*m.P[z, r,'H2O']**2 \
          - pow(m.P[z, r,'H2'],4) * m.P[z, r,'CO2']/m.Ke[z, r, 3])*(1/3600)

model.Def_Rate3 = pe.Constraint(model.z, model.r, rule=Def_Rate3_rule)

###
def d_r_comp_CH4(m, z, r):
    return m.r_comp[z, r,"CH4"] == \
        -eta_1*m.Rate[z, r, 1] - eta_3*m.Rate[z, r, 3]
model.Def_r_comp_ch4 = pe.Constraint(model.z, model.r, rule=d_r_comp_CH4)

def d_r_comp_CO(m, z, r):
    return m.r_comp[z, r, "CO"] == \
        eta_1*m.Rate[z, r,1] -eta_2*m.Rate[z, r, 2]
model.Def_r_comp_co = pe.Constraint(model.z, model.r, rule=d_r_comp_CO)

def d_r_comp_CO2(m, z, r):
    return m.r_comp[z, r, "CO2"] == \
        eta_2*m.Rate[z, r, 2] + eta_3*m.Rate[z, r, 3]
model.Def_r_comp_co2 = pe.Constraint(model.z, model.r, rule=d_r_comp_CO2)

def d_r_comp_H2(m, z, r):
    return m.r_comp[z, r, "H2"] == \
        3*eta_1*m.Rate[z, r,1] + eta_2*m.Rate[z, r,2] + 4*eta_3*m.Rate[z, r,3]
model.Def_r_comp_h2 = pe.Constraint(model.z, model.r, rule=d_r_comp_H2)

def d_r_comp_H2O(m, z, r):
    return m.r_comp[z, r, "H2O"] == \
        -eta_1*m.Rate[z, r, 1] - eta_2*m.Rate[z, r,2] - 2*eta_3*m.Rate[z, r, 3]
model.Def_r_comp_h2o = pe.Constraint(model.z, model.r, rule=d_r_comp_H2O)


def d_r_comp_N2(m, z, r):
    return m.r_comp[z, r, "N2"] == 0.0
model.Def_r_comp_n2 = pe.Constraint(model.z, model.r, rule=d_r_comp_N2)


def Def_dy_rule(m,z, r):
    if z == 0:
        return pe.Constraint.Skip
    else:
        return -m.P0 * 1e5*m.dy[z, r] == \
            (m.u/d_p)*((1-eb)/eb**3)*(150*(1-eb)*mu_ave*1e-7/d_p + 1.75*rho_g*m.u)
model.Def_dy = pe.Constraint(model.z, model.r, rule=Def_dy_rule)

# u is in m / s

def def_fi(m, z, r, c):  # kmol h^-1
    return m.F[z, r, c] == m.C[z, r, c] * m.u * (3.14*d_t_in**2/4)*3600
model.Def_fi = pe.Constraint(model.z, model.r, model.SPECIES, rule=def_fi)


def def_fi_in(m, c):  # kmol h^-1
    return m.F_in[c] == m.C_in[c] * m.u * (3.14*d_t_in**2/4)*3600
model.Def_fi_in = pe.Constraint(model.SPECIES, rule=def_fi_in)


# dC_Zz and d2Tdz variables
model.C_Z = pe.Var(model.z, model.r, model.SPECIES, initialize=1e-4)
# d2Cdz2
model.dC_Zz = dae.DerivativeVar(model.C_Z, wrt=model.z) # 6

model.C_R = pe.Var(model.z, model.r, model.SPECIES, initialize=1e-4)
# d2Cdr2
model.dC_Rr = dae.DerivativeVar(model.C_R, wrt=model.r) # 7

model.T_Z = pe.Var(model.z, model.r, initialize=1e-4)
# d2Tdz2
model.dT_Zz = dae.DerivativeVar(model.T_Z, wrt=model.z) # 8

model.T_R = pe.Var(model.z, model.r, initialize=1e-4)
# d2Tdr2
model.dT_Rr = dae.DerivativeVar(model.T_R, wrt=model.r) # 9

discretizer = pe.TransformationFactory('dae.collocation')
# discretization for axial
discretizer.apply_to(m, wrt=model.z, nfe=6, ncp=3, scheme='LAGRANGE-RADAU')
# discretization for radial
discretizer.apply_to(m, wrt=model.r, nfe=2, ncp=3, scheme='LAGRANGE-RADAU')



def x_rule(m, z, r, s):
    return m.X[z, r, s] * m.Ft[z, r] == m.F[z, r, s]
model.x_eq = pe.Constraint(model.z, model.r, model.SPECIES, rule=x_rule)

A_lambda = {}
A_lambda["CH4"] = -0.0093
A_lambda["CO"] = 0.00158
A_lambda["CO2"] = -0.0120
A_lambda["H2"] = 0.03951
A_lambda["H2O"] = 0.00053
A_lambda["N2"] = 0.00309

B_lambda = {}
B_lambda["CH4"] = 1.4028e-4
B_lambda["CO"] = 8.2511e-5
B_lambda["CO2"] = 1.0208e-4
B_lambda["H2"] = 4.5918e-4
B_lambda["H2O"] = 4.7093e-5
B_lambda["N2"] = 7.5930e-5

C_lambda = {}
C_lambda["CH4"] = 3.3180e-8
C_lambda["CO"] = -1.9081e-8
C_lambda["CO2"] = -2.2403e-8
C_lambda["H2"] = -6.4933e-8
C_lambda["H2O"] = 4.9551e-8
C_lambda["N2"] = -1.1014e-8

model.lambda_i = pe.Var(model.z, model.r, model.SPECIES,
                        initialize=lambda m, z, r, s: (A_lambda[s] +
                                                       B_lambda[s]*m.T0 +
                                                       C_lambda[s]*m.T0**2))

def lambda_i_rule(m, z, r, s):
    return m.lambda_i[z, r, s] == A_lambda[s] + B_lambda[s] * m.T[z, r] \
        + C_lambda[s] * m.T[z, r]**2
model.lamda_i_eq = pe.Constraint(model.z, model.r, model.SPECIES, rule=lambda_i_rule)


# Some more parameters
Tc = {}
Tc["CH4"] = 190.58
Tc["CO"] = 132.92
Tc["CO2"] = 304.19
Tc["H2"] = 33.18
Tc["H2O"] = 647.13
Tc["N2"] = 126.10

Pc = {}
Pc["CH4"] = 46.04
Pc["CO"] = 34.99
Pc["CO2"] = 73.82
Pc["H2"] = 13.13
Pc["H2O"] = 220.55
Pc["N2"] = 33.94

MV = {}
MV["CH4"]= 16.04276
MV["CO"]= 28.0104
MV["CO2"]= 44.0098
MV["H2"]= 2.01588
MV["H2O"]= 18.01528
MV["N2"]= 28.01348

phi = {}
phi["CH4","CH4"] = (MV["CH4"]/MV["CH4"])**0.5
phi["CH4","CO"] = (MV["CO"]/MV["CH4"])**0.5
phi["CH4","CO2"] = (MV["CO2"]/MV["CH4"])**0.5
phi["CH4","H2"] = (MV["H2"]/MV["CH4"])**0.5
phi["CH4","H2O"] = (MV["H2O"]/MV["CH4"])**0.5
phi["CH4","N2"] = (MV["N2"]/MV["CH4"])**0.5


phi["CO","CH4"] = (MV["CH4"]/MV["CO"])**0.5
phi["CO","CO"] = (MV["CO"]/MV["CO"])**0.5
phi["CO","CO2"] = (MV["CO2"]/MV["CO"])**0.5
phi["CO","H2"] = (MV["H2"]/MV["CO"])**0.5
phi["CO","H2O"] = (MV["H2O"]/MV["CO"])**0.5
phi["CO","N2"] = (MV["N2"]/MV["CO"])**0.5


phi["CO2","CH4"] = (MV["CH4"]/MV["CO2"])**0.5
phi["CO2","CO"] = (MV["CO"]/MV["CO2"])**0.5
phi["CO2","CO2"] = (MV["CO2"]/MV["CO2"])**0.5
phi["CO2","H2"] = (MV["H2"]/MV["CO2"])**0.5
phi["CO2","H2O"] = (MV["H2O"]/MV["CO2"])**0.5
phi["CO2","N2"] = (MV["N2"]/MV["CO2"])**0.5


phi["H2","CH4"] = (MV["CH4"]/MV["H2"])**0.5
phi["H2","CO"] = (MV["CO"]/MV["H2"])**0.5
phi["H2","CO2"] = (MV["CO2"]/MV["H2"])**0.5
phi["H2","H2"] = (MV["H2"]/MV["H2"])**0.5
phi["H2","H2O"] = (MV["H2O"]/MV["H2"])**0.5
phi["H2","N2"] = (MV["N2"]/MV["H2"])**0.5


phi["H2O","CH4"] = (MV["CH4"]/MV["H2O"])**0.5
phi["H2O","CO"] = (MV["CO"]/MV["H2O"])**0.5
phi["H2O","CO2"] = (MV["CO2"]/MV["H2O"])**0.5
phi["H2O","H2"] = (MV["H2"]/MV["H2O"])**0.5
phi["H2O","H2O"] = (MV["H2O"]/MV["H2O"])**0.5
phi["H2O","N2"] = (MV["N2"]/MV["H2O"])**0.5


phi["N2","CH4"] = (MV["CH4"]/MV["N2"])**0.5
phi["N2","CO"] = (MV["CO"]/MV["N2"])**0.5
phi["N2","CO2"] = (MV["CO2"]/MV["N2"])**0.5
phi["N2","H2"] = (MV["H2"]/MV["N2"])**0.5
phi["N2","H2O"] = (MV["H2O"]/MV["N2"])**0.5
phi["N2","N2"] = (MV["N2"]/MV["N2"])**0.5

Gamma = {}
for i in ["CH4", "CO", "CO2", "H2", "H2O", "N2"]:
   # Gamma = 210 * (Tc * MV^3 / Pc^4)^(1/6);
   Gamma[i] = 210 * (Tc[i] * MV[i]**3 / Pc[i]**4)**(1/6);

model.lambda_tr = pe.Var(model.z, model.r, model.SPECIES, model.SPECIES,
                         initialize=1.1)

# C6
def lambda_tr_rule(m, z, r, i, j):
    return m.lambda_tr[z, r, i, j] == \
        Gamma[j]*(pe.exp(0.0464*m.T[z, r]/Tc[i]) -
                  pe.exp(-0.2412*m.T[z, r]/Tc[i]))/(Gamma[i]*(pe.exp(0.0464*m.T[z, r]/Tc[j]) - pe.exp(-0.2412*m.T[z, r]/Tc[j])))

model.lambda_tr_eq = pe.Constraint(model.z, model.r,
                                   model.SPECIES, model.SPECIES,
                                   rule=lambda_tr_rule)


model.A_lambda_ij = pe.Var(model.z, model.r, model.SPECIES, model.SPECIES,
                           initialize=1.)

# C5
def A_lambda_ij_rule(m, z, r, s1, s2):
    return m.A_lambda_ij[z, r, s1, s2] == \
        ((1 + pe.sqrt(m.lambda_tr[z, r, s1, s2])*phi[s1, s2]**(-0.5))**2)/pe.sqrt(8 * (1 + phi[s1,s2]**-2))

model.A_lambda_ij_e = pe.Constraint(model.z, model.r,
                                    model.SPECIES, model.SPECIES,
                                    rule=A_lambda_ij_rule)
#

CP_g = 41.06 # J kg^-1 K^-1
model.lg_den = pe.Var(model.z, model.r, model.SPECIES, initialize=1.1)


# (for C4)
def lg_den_rule(m, z, r, s):
    return m.lg_den[z, r, s] == sum(m.X[z, r, s1]*m.A_lambda_ij[z, r, s, s1]
                                    for s1 in m.SPECIES)

model.lg_den_eq = pe.Constraint(model.z, model.r, model.SPECIES, rule=lg_den_rule)

#
model.lambda_g = pe.Var(model.z, model.r, initialize=1.1)

# C4
def lambda_g_rule(m, z, r):
    return m.lambda_g[z, r] == \
        sum(m.X[z, r, s]*m.lambda_i[z, r, s]/m.lg_den[z, r, s] for s in m.SPECIES)

model.lambda_g_eq = pe.Constraint(model.z, model.r, rule=lambda_g_rule)


# More parameters (viscosity)
A_mu = {}
A_mu["CH4"] = 3.844
A_mu["CO"] = 23.811
A_mu["CO2"] = 11.811
A_mu["H2"] = 27.758
A_mu["H2O"] = -36.826
A_mu["N2"] = 42.606

B_mu = {}

B_mu["CH4"] = 4.0112e-1
B_mu["CO"] = 5.3944e-1
B_mu["CO2"] = 4.9838e-1
B_mu["H2"] = 2.1200e-1
B_mu["H2O"] = 4.2900e-1
B_mu["N2"] = 4.7500e-1

C_mu = {}

C_mu["CH4"] = -1.4303e-4
C_mu["CO"] = -1.5411e-4
C_mu["CO2"] = -1.0851e-4
C_mu["H2"] = -3.2800e-5
C_mu["H2O"] = -1.6200e-5
C_mu["N2"] = -9.8800e-5

A_cp = {}

A_cp["CH4"] = 34.942
A_cp["CO"] = 29.556
A_cp["CO2"] = 27.437
A_cp["H2"] = 25.399
A_cp["H2O"] = 33.933
A_cp["N2"] = 29.342

B_cp = {}

B_cp["CH4"] = -3.9957e-2
B_cp["CO"] = -6.5807e-3
B_cp["CO2"] = 4.2315e-2
B_cp["H2"] = 2.0178e-2
B_cp["H2O"] = -8.4186e-3
B_cp["N2"] = -3.5395e-3

C_cp = {}
C_cp["CH4"] = 1.9184e-4
C_cp["CO"] = 2.0130e-5
C_cp["CO2"] = -1.9555e-5
C_cp["H2"] = -3.8549e-5
C_cp["H2O"] = 2.9906e-5
C_cp["N2"] = 1.0076e-5

D_cp = {}

D_cp["CH4"] = -1.5303e-7
D_cp["CO"] = -1.2227e-8
D_cp["CO2"] = 3.9968e-9
D_cp["H2"] = 3.1880e-8
D_cp["H2O"] = -1.7825e-8
D_cp["N2"] = -4.3116e-9

E_cp = {}

E_cp["CH4"] = 3.9321e-11
E_cp["CO"] = 2.2617e-12
E_cp["CO2"] = -2.9872e-13
E_cp["H2"] = -8.7585e-12
E_cp["H2O"] = 3.6934e-12
E_cp["N2"] = 2.5935e-13

model.mu_i = pe.Var(model.z, model.r, model.SPECIES, bounds=(0.0, None),
                    initialize=lambda m, z, r, s:(A_mu[s] +
                                                  B_mu[s]*m.T0 +
                                                  C_mu[s]*m.T0**2))

def mu_i_rule(m, z, r, s):
    return m.mu_i[z, r, s] == A_mu[s] + B_mu[s] * m.T[z, r] + C_mu[s] * m.T[z, r]**2
model.mu_i_eq = pe.Constraint(model.z, model.r, model.SPECIES, rule=mu_i_rule)

x0f = 1/len(model.SPECIES)
model.mu_den = pe.Var(model.z, model.r, model.SPECIES,
                      initialize=lambda m, z, r, s: sum(x0f*phi[s, s1] for s1 in
                                                        m.SPECIES))


def mu_d_rule(m, z, r, s):
    return m.mu_den[z, r, s] == sum(m.X[z, r, s1]*phi[s,s1] for s1 in m.SPECIES)
model.mu_den_eq = pe.Constraint(model.z, model.r, model.SPECIES, rule=mu_d_rule)


model.mu = pe.Var(model.z, model.r, bounds=(0, None))

def mu_rule(m, z, r):
    return m.mu[z, r] == sum(m.X[z, r, s]*m.mu_i[z, r, s]/m.mu_den[z, r, s]
                             for s in m.SPECIES)

model.mu_eq = pe.Constraint(model.z, model.r, rule=mu_rule)
#

model.CP_i = pe.Var(model.z, model.r, model.SPECIES,
                    initialize=lambda m, z, r, s: (A_cp[s] +
                    B_cp[s]*m.T0 + C_cp[s]*m.T0**2 + D_cp[s]*m.T0**3 +
                    E_cp[s]*m.T0**4)*(1000/MV[s]))

# component heat capacity  [J kg-1 K-1]
def cp_i_rule(m, z, r, s):
    return m.CP_i[z, r, s] == (A_cp[s]
                            + B_cp[s]*m.T[z, r]
                            + C_cp[s]*m.T[z, r]**2
                            + D_cp[s]*m.T[z, r]**3
                            + E_cp[s]*m.T[z, r]**4) * (1000/MV[s])

model.CPi_eq = pe.Constraint(model.z, model.r, model.SPECIES, rule=cp_i_rule)

model.CP_g = pe.Var(model.z, model.r, bounds=(0, None))

# heat capacity of gas mixture
def cp_g_rule(m, z, r):
    return m.CP_g[z, r] == sum(m.X[z, r, s] * m.CP_i[z, r, s] for s in m.SPECIES)
model.CP_g_eq = pe.Constraint(model.z, model.r, rule=cp_g_rule)

#
model.Re = pe.Var(model.z, bounds=(0, None))

# this is only axial
def re_rule(m, z):
    return m.Re[z] == 1e7 * d_p * rho_g * m.u / m.mu[z, 0]
model.Re_eq = pe.Constraint(model.z, rule=re_rule)



model.Pr = pe.Var(model.z, bounds=(0, None))


def pr_rule(m, z):
    return m.Pr[z] == 1e-7 * m.CP_g[z, 0] * m.mu[z, 0] / m.lambda_g[z, 0]
model.Pr_eq = pe.Constraint(model.z, rule=pr_rule)


model.U = pe.Var(model.z, bounds=(0, None))

# only in the z, 0
def u_rule(m, z):
    return m.U[z] == \
        0.4*(m.lambda_g[z, 0]/d_p)*(2.58*(m.Re[z]**0.333)*(m.Pr[z]**0.333)\
                                 +0.094*(m.Re[z]**0.8)*(m.Pr[z]**0.4))

model.U_eq = pe.Constraint(model.z, rule=u_rule)

# C_Z = dCz
model.Def_c_Z = pe.Constraint(model.z, model.r, model.SPECIES,
                              rule=lambda m, z, r, s: m.C_Z[z, r, s] == m.dCz[z, r, s])
# C_R = dCr
model.Def_c_R = pe.Constraint(model.z, model.r, model.SPECIES,
                              rule=lambda m, z, r, s: m.C_R[z, r, s] == m.dCr[z, r, s])


dscale = 1e-05



# T_Z = dTz
model.Def_T_Z = pe.Constraint(model.z, model.r,
                              rule=lambda m, z, r: m.T_Z[z, r] == m.dTz[z, r])
# T_R = dTr
model.Def_T_R = pe.Constraint(model.z, model.r,
                              rule=lambda m, z, r: m.T_R[z, r] == m.dTr[z, r])




model.edC_ZCp = pe.Var(model.z, model.r,
                     initialize=lambda m, z, r:
                     sum(eb*dscale*pe.value(m.C_Z[z, r, s])*pe.value(m.CP_i[z, r, s])
                         for s in m.SPECIES)
                     )

model.edC_RCp = pe.Var(model.z, model.r,
                     initialize=lambda m, z, r:
                     sum(eb*dscale*pe.value(m.C_R[z, r, s])*pe.value(m.CP_i[z, r, s])
                         for s in m.SPECIES)
                     )

# ### PDE T(z, r) #############################################################
def Def_T_PDE_rule(m, z, r):
    if r == 0 or r == m.R or z == 0 or z == m.L:
        return pe.Constraint.Skip
    else:
        return m.u*rho_g*m.CP_g[z, r]* m.T_Z[z, r] == \
            + m.edC_ZCp[z, r] * m.T_Z[z, r] \
            + m.edC_RCp[z, r] * m.T_R[z, r] \
            + m.lambda_g[z, r]*m.dT_Zz[z, r] \
            + m.lambda_g[z, r]*(m.dT_Rr[z, r]+(1/r)*m.T_R[z, r])\
            + (1-eb)*rho_p*1000*1000\
            *sum(eta[k]*m.Rate[z, r, k] * (-dHr[k]) for k in model.REACTIONS)

model.Def_T_PDE = pe.Constraint(model.z, model.r, rule=Def_T_PDE_rule)

# once again we forget the old dtdZ

# diffusivity terms
sigma_ = {}
sigma_["CH4"] = 3.758
sigma_["CO"] = 3.690
sigma_["CO2"] = 3.941
sigma_["H2"] = 2.827
sigma_["H2O"] = 2.641
sigma_["N2"] = 3.798

epsik = {}
epsik["CH4"] = 148.6
epsik["CO"] = 91.7
epsik["CO2"] = 195.2
epsik["H2"] = 59.7
epsik["H2O"] = 809.1
epsik["N2"] = 71.4

sigma_ij = {}
MV_ij = {}
for (s1, s2) in product(model.SPECIES, model.SPECIES):
    sigma_ij[s1, s2] = (sigma_[s1] + sigma_[s2])/2
    MV_ij[s1, s2] = 2./((1/MV[s1]) + (1/MV[s2]))


#-=-##-=-##-
model.omegaD = pe.Var(model.z, model.r, model.SPECIES, model.SPECIES,
                      bounds = (0,None), initialize=1)
#-=-##-=-#


def omega_d_rule(m, z, r, s1, s2):
    return m.omegaD[z, r, s1, s2] == \
    1.06036/(m.T[z, r]*pe.sqrt(epsik[s1]*epsik[s2]))**0.1561 \
    + 0.19300/pe.exp(0.47635*m.T[z, r]/pe.sqrt(epsik[s1]*epsik[s2])) \
    + 1.03587/pe.exp(1.52996*m.T[z, r]/pe.sqrt(epsik[s1]*epsik[s2])) \
    + 1.76474/pe.exp(3.89411*m.T[z, r]/pe.sqrt(epsik[s1]*epsik[s2]))

model.def_omegad = pe.Constraint(model.z, model.r, model.SPECIES, model.SPECIES,
                                 rule=omega_d_rule)


#-=-##-=-#
model.Dij_ = pe.Var(model.z, model.r, model.SPECIES, model.SPECIES, bounds=(0, None),
                    initialize=1)
#-=-##-=-#


def dij_f_rule(m, z, r, s1, s2):
    return m.Dij_[z, r, s1, s2] == \
        (0.00266*m.T[z, r]**(3/2))/(m.Pt[z, r] \
                                 *pe.sqrt(MV_ij[s1, s2]) \
                                 *(sigma_ij[s1, s2]**2)*m.omegaD[z, r, s1, s2])

model.def_dij = pe.Constraint(model.z, model.r, model.SPECIES, model.SPECIES,
                              rule=dij_f_rule)



#-=-##-=-#
model.Dim_ = pe.Var(model.z, model.r, model.SPECIES, bounds = (0, None), initialize=1)
#-=-##-=-#

def dim_rule_(m, z, r, s1):
    return model.Dim_[z, r, s1] == sum(m.X[z, r, s2]/m.Dij_[z, r, s1, s2] for s2 in
                             model.SPECIES if s2 != s1)

model.def_dim_ = pe.Constraint(model.z, model.r, model.SPECIES, rule=dim_rule_)


#-=-##-=-#- Z direction diffusion
model.Diez_ = pe.Var(model.z, model.r, model.SPECIES, bounds=(0, None), initialize=1)
#-=-##-=-#-


def diez_rule(m, z, r, s1):
    return model.Diez_[z, r, s1] == 0.78*model.Dim_[z, r, s1] + \
        (0.54*model.u*d_cata/eb)/(1 + 9.2*model.Dim_[z, r,s1]/(model.u*d_cata/eb))

model.def_diez_ = pe.Constraint(model.z, model.r, model.SPECIES, rule=diez_rule)

# ### PDE C(z, r) #############################################################
def Def_C_PDE_rule(m, z, r, c):
    if r == 0 or r == m.R or z == 0 or z == m.L:
        return pe.Constraint.Skip
    else:
        return m.u*m.C_Z[z, r, c] == \
            eb * m.Diez_[z, r, c] * m.dC_Zz[z, r, c] + \
            eb * m.Diez_[z, r, c] * (m.dC_Rr[z, r, c] + (1/r)*m.C_R[z, r, c])+ \
            +(1-eb)*rho_p*m.r_comp[z, r, c]

model.Def_C_PDE = pe.Constraint(model.z, model.r, model.SPECIES, rule=Def_C_PDE_rule)

#
def Def_edC_ZCp_rule(m, z ,r):
    if r == 0 or r == m.R or z == 0 or z == m.L:
        return pe.Constraint.Skip
    else:
        return m.edC_ZCp[z, r] == \
            sum(eb*m.Diez_[z, r, s]*m.C_Z[z, r, s]*m.CP_i[z, r, s] for s in m.SPECIES)

model.Def_edC_ZCp = pe.Constraint(model.z, model.r, rule=Def_edC_ZCp_rule)

def Def_edC_RCp_rule(m, z ,r):
    if r == 0 or r == m.R or z == 0 or z == m.L:
        return pe.Constraint.Skip
    else:
        return m.edC_RCp[z, r] == \
            sum(eb*m.Diez_[z, r, s]*m.C_R[z, r, s]*m.CP_i[z, r, s] for s in m.SPECIES)

model.Def_edC_RCp = pe.Constraint(model.z, model.r, rule=Def_edC_RCp_rule)


# ### Boundary Conditions #####################################################
def def_bc_C_z0(m, r, s): # 1
    #if r == 0 or r == m.R:
    #    return pe.Constraint.Skip
    #else:
    return m.C[0, r, s] == m.C_in[s]
model.bc_C_z0 = pe.Constraint(model.r, model.SPECIES, rule=def_bc_C_z0)

def def_bc_T_z0(m, r): # 2
    #if r == 0 or r == m.R:
    #    return pe.Constraint.Skip
    #else:
    return m.T[0, r] == m.T0
model.bc_T_z0 = pe.Constraint(model.r, rule=def_bc_T_z0)

def def_bc_y_z0(m, r): # 3
    #if r == 0 or r == m.R:
    #    return pe.Constraint.Skip
    #else:
    return m.y[0, r] == 1.0
model.bc_y_z0 = pe.Constraint(model.r, rule=def_bc_y_z0)

def def_bc_C_ZL(m, r, s): # 4
    #if r == 0 or r == m.R:
    #    return pe.Constraint.Skip
    #else:
    return m.C_Z[m.L, r, s] == 0.0
model.bc_C_Z = pe.Constraint(model.r, model.SPECIES, rule=def_bc_C_ZL)

def def_bc_T_ZL(m, r): # 5
    #if r == 0 or r == m.R:
    #    return pe.Constraint.Skip
    #else:
    return m.T_Z[m.L, r] == 0.0
model.bc_T_Z = pe.Constraint(model.r, rule=def_bc_T_ZL)

def def_bc_C_R0(m, z, s): # 6
    if z == 0 or z == m.L:
        return pe.Constraint.Skip
    else:
        return m.C_R[z, 0, s] == 0.0
model.bc_C_R_0 = pe.Constraint(model.z, model.SPECIES, rule=def_bc_C_R0)


def def_bc_C_RR(m, z, s): # 7
    if z == 0 or z == m.L:
        return pe.Constraint.Skip
    else:
        return m.C_R[z, m.R, s] == 0.0
model.bc_C_R_r = pe.Constraint(model.z, model.SPECIES, rule=def_bc_C_RR)


def def_bc_R_R0(m, z): # 8
    if z == 0 or z == m.L:
        return pe.Constraint.Skip
    else:
        return m.T_R[z, 0] == 0.0
model.bc_T_R_0 = pe.Constraint(model.z, rule=def_bc_R_R0)


def def_bc_R_RR(m, z): # 9
    if z == 0 or z == m.L:
        return pe.Constraint.Skip
    else:
        return (m.lambda_g[z, m.R]*m.T_R[z, m.R] == m.U[z] * (Tw - m.T[z, m.R]))
model.bc_T_R_r = pe.Constraint(model.z, rule=def_bc_R_RR)

m.C_in["CH4"].set_value(5.46/ ( R_g * m.T0 )* (1/10))
m.C_in["CO"].set_value(0/(R_g*m.T0)*(1/10))
m.C_in["CO2"].set_value(0.31/ ( R_g * m.T0 )* (1/10))
m.C_in["H2"].set_value(0.68/ ( R_g * m.T0)* (1/10))
m.C_in["H2O"].set_value(18.34/ ( R_g * m.T0 )* (1/10))
m.C_in["N2"].set_value(0.90/ ( R_g * m.T0 )* (1/10))

m.P0.set_value(sum(pe.value(m.C_in[:]))*10 * R_g * m.T0 )
#m.u.fix(2.14)

# Dummy optimize function
# model.obj = pe.Objective(expr=1) # Dummy Objective
#def Objective_rule(m):
#    return 1.0
#model.obj = pe.Objective(rule = Objective_rule) # Dummy Objective

