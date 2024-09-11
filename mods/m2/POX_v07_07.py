#!/usr/bin/env python
# coding: utf-8

import pyomo.environ as pe
import pyomo.dae as dae
import numpy as np
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product

# # Parameters and data:

# constraints for reactions:
A_rxn = {1: 4.225e15,  #;//kmol bar^0.5 kgcat^-1 h^-1
         2: 1.955e6,  #;//kmol bar^-1 kgcat^-1 h^-1
         3: 1.020e15  #;//kmol bar^0.5 kgcat^-1 h^-1
         }
Ak1 = 4.255e15   # pre-exponential factor of rate constant of reaction 1
Ak2 = 1.955e6    # pre-exponential factor of rate constant of reaction 2
Ak3 = 1.02e15    # pre-exponential factor of rate constant of reaction 3

E1 = 240.1       # kJ/mol activation energy of reaction 1
E2 = 67.13       # kJ/mol activation energy of reaction 2
E3 = 243.9       # kJ/mol activation energy of reaction 3


R_g = 0.0083144621  # kJ/(K*mol) gas constant

#_eq ("R_1"): 4.707e12;//bar^2
#A_eq ("R_2"): 1.142e-2;//[dimensionless]
#A_eq ("R_3"): 5.375e10;//bar^2
A_eq = {1: 4.707e12, # bar^2
        2:1.142e-2, # []
        3:5.375e10} # bar^2, [], bar^2

# dhr in kJ/kmol
dHr = {1: 206.1,
       2: -41.15,
       3:164.9}
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
#radius = 0.03/2                                   # m  radius
radius = 0.1016/2

Ac = np.pi * radius**2                          # m^2 cross sectional area
rho_cata = 1820                                 # kg/m3 catalyst density
d_cata = 5e-3                                   # m  catalyst diameter
mu_ave = 2.97e-5                                # Pa*s  gas average viscosity

# @dav: Cp_ave should have J/kg-K units
Cp_ave = 41.06 # J/(mol*K) gas mixture specific heat


CP_g = 41.06 # J kg^-1 K^-1
rho_g = 1.6  # kg m^-3, gas density

# # Define variables and parameters

model = m = pe.ConcreteModel()

# model reactor parameters
model.L = pe.Param(initialize=12)  # reactor total length
model.z = dae.ContinuousSet(bounds = (0,model.L))   # z direction
model.SPECIES = pe.Set(initialize = ['CH4','H2O','H2','CO','CO2','N2'])
model.REACTIONS = pe.RangeSet(3)
#print(model.REACTIONS.data())
model.P0 = pe.Param(initialize = 10., mutable=True) # inlet pressure
model.T0 = pe.Param(initialize = 793.15) # inlet temperature

# define reactor inlet variables

model.FCH4_in = pe.Param(initialize = 30)    # fix inlet CH4

model.F_in = pe.Var(model.SPECIES, bounds=(0,None))   # inlet flow rate kmol/h
model.C_in = pe.Var(model.SPECIES, bounds=(0,None))
model.Ft_in = pe.Var(bounds = (0,None))   # total inlet flow rate kmol/h
model.X_in = pe.Var(model.SPECIES)   # inlet mole fraction
# shouldn't this be below 1

# in m/s
model.u = pe.Var(bounds = (0,None), initialize = 4.5) # superfacial velocity
model.Rep = pe.Var(bounds = (0,None))   # practical Raynolds number
model.f = pe.Var(bounds = (0,None)) # friction factor

# define variables that vary along reactor length

model.F = pe.Var(model.z, model.SPECIES, bounds = (0,None))
model.C = pe.Var(model.z, model.SPECIES, bounds = (0,None)) # kmol/m^3

model.X = pe.Var(model.z, model.SPECIES)

model.y = pe.Var(model.z, bounds = (0,None))
model.Pt = pe.Var(model.z, bounds = (0,None))
#model.T = pe.Var(model.z, bounds = (1e1,None))
model.T = pe.Var(model.z, bounds = (1e-06,None))
model.Ft = pe.Var(model.z, bounds = (0,None))
model.P = pe.Var(model.z, model.SPECIES, bounds = (0,None), initialize=1e0)
model.Rate = pe.Var(model.z, model.REACTIONS)

model.r_comp = pe.Var(model.z, model.SPECIES)

model.DEN = pe.Var(model.z, initialize=1e0)

    # define coeffs
model.k1 = pe.Var(model.z, bounds=(0, None))
model.k2 = pe.Var(model.z)
model.k3 = pe.Var(model.z, bounds=(0, None))


model.Ke = pe.Var(model.z, model.REACTIONS,
                  bounds=(1e-08, None), initialize=1e3)

model.Ka = pe.Var(model.z, model.SPECIES, bounds=(0e-00, None), initialize=1e0)

# define derivative variables
model.dC = dae.DerivativeVar(model.C, wrt = model.z)
model.dy = dae.DerivativeVar(model.y, wrt = model.z)
model.dT = dae.DerivativeVar(model.T, wrt = model.z)


# # Constraints
# ## Reaction coefficients equations

def defk1_rule(m,z):        # kmol*bar^(1/2)/(kgcat*hr) rate constant of reaction 1
    return m.k1[z] == A_rxn[1]*pe.exp(-E1/(R_g*m.T[z]))
model.defk1 = pe.Constraint(model.z, rule = defk1_rule)

def defk2_rule(m,z):        # kmol/(bar* kgcat*hr) rate constant of reaction 2
    return m.k2[z] == A_rxn[2]*pe.exp(-E2/(R_g*m.T[z]))
model.defk2 = pe.Constraint(model.z, rule = defk2_rule)

def defk3_rule(m,z):        # kmol*bar^(1/2)/(kgcat*hr) rate constant of reaction 3
    return m.k3[z] == A_rxn[3]*pe.exp(-E3/(R_g*m.T[z]))
model.defk3 = pe.Constraint(model.z, rule = defk3_rule)


def defKe_rule(m, z, r):  # rate equilibrium constant
    return pe.log(m.Ke[z, r] / A_eq[r]) == (-dHr[r]/(R_g* m.T[z]))
model.defKe = pe.Constraint(model.z, model.REACTIONS, rule = defKe_rule)


model.logKa = pe.Var(model.z, model.SPECIES, initialize=1e-3)

def deflogKa_rule(m, z, s):   # adsorption constant for each species
    if s == 'CO2' or s == 'N2':
        return pe.Constraint.Skip
    #return pe.log(m.Ka[z,s] / AKa[s])== (-dHa[s]/(R_g*m.T[z]))
    return m.logKa[z,s]== (-dHa[s]/(R_g*m.T[z]))

model.deflogKa = pe.Constraint(model.z, model.SPECIES, rule=deflogKa_rule)

def defKa_rule(m, z, s):   # adsorption constant for each species
    if s == 'CO2' or s == 'N2':
        return pe.Constraint.Skip
    #return pe.log(m.Ka[z,s] / AKa[s])== (-dHa[s]/(R_g*m.T[z]))
    return m.Ka[z, s]/AKa[s] == pe.exp(m.logKa[z,s])

model.defKa = pe.Constraint(model.z, model.SPECIES, rule=defKa_rule)

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
def Pressure_nondim_rule(m,z):
    return m.Pt[z] == m.y[z] * m.P0
model.Pressure_nondimensionalize = pe.Constraint(model.z,
                                                 rule=Pressure_nondim_rule)

def Total_Flow_rule(m,z): # total flow rate is sum of single species flow rate
    return m.Ft[z] == sum(m.F[z,s] for s in model.SPECIES)
model.Total_Flow = pe.Constraint(model.z, rule = Total_Flow_rule)

def Patial_pressure_rule(m,z,s):  # patial pressure
    #return m.P[z,s] == m.F[z,s] / m.Ft[z] * m.Pt[z]
    return m.P[z,s] * m.Ft[z] == m.F[z,s] * m.Pt[z]
model.Patial_pressure = pe.Constraint(model.z, model.SPECIES,
                                      rule =  Patial_pressure_rule)

def Def_DEN_rule(m,z):  # define DEN
    return m.DEN[z] * m.P[z,'H2'] == m.P[z,'H2'] \
    + m.Ka[z,'CO']*m.P[z,'CO'] * m.P[z,'H2'] \
    + m.Ka[z,'CH4']*m.P[z,'CH4'] * m.P[z,'H2'] \
    + m.Ka[z,'H2']*m.P[z,'H2'] * m.P[z,'H2'] \
    + m.Ka[z,'H2O']*m.P[z,'H2O']
model.Def_DEN = pe.Constraint(model.z, rule =  Def_DEN_rule)


def Def_Rate1_rule(m,z):   # kmol/(kgcat*s) rate law for reaction 1
    return m.Rate[z,1] == \
        (m.k1[z] / (pow(m.P[z,'H2'],2.5)*m.DEN[z]**2)) \
        *(m.P[z,'CH4']*m.P[z,'H2O'] \
           - pow(m.P[z,'H2'], 3) * m.P[z,'CO']/m.Ke[z, 1])*(1/3600)

model.Def_Rate1 = pe.Constraint(model.z, rule =  Def_Rate1_rule)

def Def_Rate2_rule(m,z):   # kmol/(kgcat*s) rate law for reaction 2
    return m.Rate[z,2] == \
        (m.k2[z] / (m.P[z,'H2'] * m.DEN[z]**2)) \
        *(m.P[z,'CO']*m.P[z,'H2O'] \
          - m.P[z,'H2'] * m.P[z,'CO2']/m.Ke[z, 2])*(1/3600)

model.Def_Rate2 = pe.Constraint(model.z, rule =  Def_Rate2_rule)

def Def_Rate3_rule(m,z):   # kmol/(kgcat*s) rate law for reaction 1
    return m.Rate[z,3] == \
        (m.k3[z] / (pow(m.P[z,'H2'],3.5) * m.DEN[z]**2)) \
        *(m.P[z,'CH4']*m.P[z,'H2O']**2 \
          - pow(m.P[z,'H2'],4) * m.P[z,'CO2']/m.Ke[z, 3])*(1/3600)

model.Def_Rate3 = pe.Constraint(model.z, rule =  Def_Rate3_rule)

###
def d_r_comp_CH4(m, z):
    return m.r_comp[z,"CH4"] == \
        -eta_1*m.Rate[z,1] - eta_3*m.Rate[z,3]
model.Def_r_comp_ch4 = pe.Constraint(model.z, rule=d_r_comp_CH4)

def d_r_comp_CO(m, z):
    return m.r_comp[z, "CO"] == \
        eta_1*m.Rate[z,1] -eta_2*m.Rate[z,2]
model.Def_r_comp_co = pe.Constraint(model.z, rule=d_r_comp_CO)

def d_r_comp_CO2(m, z):
    return m.r_comp[z, "CO2"] == \
        eta_2*m.Rate[z,2] + eta_3*m.Rate[z,3]
model.Def_r_comp_co2 = pe.Constraint(model.z, rule=d_r_comp_CO2)

def d_r_comp_H2(m, z):
    return m.r_comp[z, "H2"] == \
        3*eta_1*m.Rate[z,1] + eta_2*m.Rate[z,2] + 4*eta_3*m.Rate[z,3]
model.Def_r_comp_h2 = pe.Constraint(model.z, rule=d_r_comp_H2)

def d_r_comp_H2O(m, z):
    return m.r_comp[z, "H2O"] == \
        -eta_1*m.Rate[z,1] - eta_2*m.Rate[z,2] - 2*eta_3*m.Rate[z,3]
model.Def_r_comp_h2o = pe.Constraint(model.z, rule=d_r_comp_H2O)


def d_r_comp_N2(m, z):
    return m.r_comp[z, "N2"] == 0.0
model.Def_r_comp_n2 = pe.Constraint(model.z, rule=d_r_comp_N2)


#
def Def_C_rule(m,z, c):
    return m.u*m.dC[z,c] ==(1-eb)*rho_p*m.r_comp[z, c]

model.Def_C_ = pe.Constraint(model.z, model.SPECIES,
                             rule=Def_C_rule)

#
def Def_dy_rule(m,z):
    return -m.P0*1e5*m.dy[z] == \
        (m.u/d_p)*((1-eb)/eb**3)*(150*(1-eb)*mu_ave*1e-7/d_p + 1.75*rho_g*m.u)
model.Def_dy = pe.Constraint(model.z, rule =  Def_dy_rule)

model.Uf = pe.Param(initialize=1.0, mutable =True)
def Def_dT_rule(m,z):
    return m.u*rho_g*CP_g* m.dT[z] == \
        4* m.Uf * (Tw - m.T[z])/d_t_in \
        + (1-eb)*rho_p*1000*1000\
        *sum(eta[r]*m.Rate[z,r] * (-dHr[r]) for r in model.REACTIONS)

# energy balance
model.Def_dT = pe.Constraint(model.z, rule =  Def_dT_rule)
#print(type(model.Def_dT_index))
# u is in m / s


def def_fi(m, z, c):  # kmol h^-1
    return m.F[z, c] == m.C[z, c] * m.u * (3.14*d_t_in**2/4)*3600
model.Def_fi = pe.Constraint(model.z, model.SPECIES, rule=def_fi)


def def_fi_in(m, c):  # kmol h^-1
    return m.F_in[c] == m.C_in[c] * m.u * (3.14*d_t_in**2/4)*3600
model.Def_fi_in = pe.Constraint(model.SPECIES, rule=def_fi_in)

# ##  Initial Conditions

# Initial conditions

def InitCon_rule(m):
    for s in model.SPECIES:
        yield m.C[0,s] == m.C_in[s]
    yield m.y[0] == 1
    yield m.T[0] == m.T0
model.InitCon = pe.ConstraintList(rule = InitCon_rule)



m.C_in["CH4"].fix(5.46/ ( R_g * m.T0 )* (1/10))
m.C_in["CO"].fix(0/(R_g*m.T0)*(1/10))
m.C_in["CO2"].fix(0.31/ ( R_g * m.T0 )* (1/10))
m.C_in["H2"].fix(0.68/ ( R_g * m.T0)* (1/10))
m.C_in["H2O"].fix(18.34/ ( R_g * m.T0 )* (1/10))
m.C_in["N2"].fix(0.90/ ( R_g * m.T0 )* (1/10))

m.P0.set_value(sum(pe.value(m.C_in[:]))*10 * R_g * m.T0 )
m.u.fix(2.14)


# Dummy optimize function
# model.obj = pe.Objective(expr=1) # Dummy Objective
def Objective_rule(m):
    return 1.0
model.obj = pe.Objective(rule = Objective_rule) # Dummy Objective


# # Solve

model.cdum = pe.Var(model.z, model.SPECIES, initialize=1e-4)
model.d2C = dae.DerivativeVar(model.cdum, wrt=model.z)
model.tdum = pe.Var(model.z, initialize=1e-4)
model.d2T = dae.DerivativeVar(model.tdum, wrt=model.z)

discretizer = pe.TransformationFactory('dae.collocation')
discretizer.apply_to(m,nfe=6,ncp=7,scheme='LAGRANGE-RADAU')

m.d2C_disc_eq.deactivate()
m.d2T_disc_eq.deactivate()


m.P[:, 'H2'].setlb(1e-8)
m.P[:, 'H2O'].setlb(0e-00)

# ipexe = "/Users/dthierry/Apps/ipopt_102623/ip_dir/bin/ipopt"
ipexe = "/Users/dthierry/Apps/ipopt_dir/bin/ipopt"

solver = pe.SolverFactory(ipexe)

solver.options['halt_on_ampl_error'] = 'yes'

ntry = 0

with open("ipopt.opt", "w") as f:
    f.write("start_with_resto\tyes\n")
    f.write("linear_solver\tma57\n")
    f.write("expect_infeasible_problem\tyes\n")
    f.write("mu_strategy\tadaptive\n")
    f.write("output_file\tout.txt\n")
    f.write("print_info_string\tyes\n")
    f.write("max_iter\t2480\n")

ntry = 1 # 1
print(f"TRY {ntry}")
m.write("my_nl.nl")
solver.solve(m,tee=True, symbolic_solver_labels=True)


with open("ipopt.opt", "w") as f:
    f.write("start_with_resto\tyes\n")
    f.write("linear_solver\tma57\n")
    f.write("expect_infeasible_problem\tyes\n")
    f.write("mu_strategy\tadaptive\n")
    f.write("output_file\tout.txt\n")
    f.write("print_info_string\tyes\n")
    f.write("max_iter\t1373\n")
    #f.write("max_iter\t102\n")

ntry = 2 # 2
print(f"TRY {ntry}")
solver.solve(m,tee=True, symbolic_solver_labels=True)

with open("ipopt.opt", "w") as f:
    f.write("start_with_resto\tyes\n")
    f.write("linear_solver\tma57\n")
    f.write("expect_infeasible_problem\tyes\n")
    f.write("mu_strategy\tadaptive\n")
    f.write("output_file\tout.txt\n")
    f.write("print_info_string\tyes\n")
    #f.write("tol\t1e-04\n")
    f.write("constr_viol_tol\t1e-04\n")
    f.write("max_iter\t542\n")

ntry = 3 # 2
print(f"TRY {ntry}")
solver.solve(m,tee=True, symbolic_solver_labels=True)
#

with open("ipopt.opt", "w") as f:
    f.write("start_with_resto\tyes\n")
    f.write("linear_solver\tma57\n")
    f.write("expect_infeasible_problem\tyes\n")
    f.write("mu_strategy\tadaptive\n")
    f.write("output_file\tout.txt\n")
    f.write("print_info_string\tyes\n")
    #f.write("tol\t1e-04\n")
    f.write("constr_viol_tol\t1e-04\n")
    #f.write("max_iter\t440\n")

ntry = 4 # 2
print(f"TRY {ntry}")
solver.solve(m,tee=True, symbolic_solver_labels=True)

m.Uf.set_value(100.0)

solver.solve(m,tee=True, symbolic_solver_labels=True)

# plots
k = 0
cvm = np.zeros(2)
for i in m.SPECIES:
    fv = np.array(pe.value(m.C[:, i]))
    if k == 0:
        cvm = fv
    else:
        cvm = np.vstack([cvm, fv])
    k += 1

tv = np.array(pe.value(m.T[:]))
yv = np.array(pe.value(m.y[:]))
zval = np.array(m.z.data())

f, a = plt.subplots(1, 2, figsize=[4*2, 3])

a[0].plot(zval, tv, "ro-")
a[0].set_title("Temperature")
a[1].plot(zval, yv, "ro-")

a[1].set_title("y (dimensionless pressure)")
a[1].set_ylim(bottom=0)

f.tight_layout()
# f.savefig("restults_v0124_a.png")

nfigs = cvm.shape[0]
f, a = plt.subplots(nfigs, 1, figsize=[4, 3*nfigs])

for i in range(cvm.shape[0]):
    a[i].plot(zval, cvm[i, :], "ro-", label=m.SPECIES.data()[i])
    a[i].set_title(m.SPECIES.data()[i])
    a[i].set_ylabel("Concentration kmol m^-3")
    #a[i].set_ylim(bottom=0)

f.tight_layout()
# f.savefig("restults_v0124_b_2.png")

# flow
k = 0
fvm = np.zeros(2)
for i in m.SPECIES:
    fv = np.array(pe.value(m.F[:, i]))
    if k == 0:
        fvm = fv
    else:
        fvm = np.vstack([fvm, fv])
    k += 1


nfigs = fvm.shape[0]
f, a = plt.subplots(nfigs, 1, figsize=[4, 3*nfigs])

for i in range(fvm.shape[0]):
    a[i].plot(zval, fvm[i, :], "go-", label=m.SPECIES.data()[i])
    a[i].set_title(m.SPECIES.data()[i])
    a[i].set_ylabel("Flow kmol h^-1")
    a[i].set_ylim(bottom=0)

f.tight_layout()
# f.savefig("restults_v0124_c.png")

def x_rule(m, z, s):
    return m.X[z, s] * m.Ft[z] == m.F[z, s]
model.x_eq = pe.Constraint(model.z, model.SPECIES, rule=x_rule)


solver.solve(m,tee=True, symbolic_solver_labels=True)

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

model.lambda_i = pe.Var(model.z, model.SPECIES, initialize=0.1)

def lambda_i_rule(m, z, s):
    return m.lambda_i[z, s] == A_lambda[s] + B_lambda[s] * m.T[z] \
        + C_lambda[s] * m.T[z]**2
model.lamda_i_eq = pe.Constraint(model.z, model.SPECIES, rule=lambda_i_rule)
print("kenny")
solver.solve(m,tee=True, symbolic_solver_labels=True)


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

model.lambda_tr = pe.Var(model.z, model.SPECIES, model.SPECIES, initialize=1.1)

def lambda_tr_rule(m, z, s1, s2):
    return m.lambda_tr[z, s1, s2] == \
        Gamma["CH4"]*(pe.exp(0.0464*m.T[z]/Tc["CH4"])-pe.exp(-0.2412*m.T[z]/Tc["CH4"])) \
        /Gamma["CH4"]*(pe.exp(0.0464*m.T[z]/Tc["CH4"])-pe.exp(-0.2412*m.T[z]/Tc["CH4"]))

model.lambda_tr_eq = pe.Constraint(model.z, model.SPECIES, model.SPECIES,
                                rule=lambda_tr_rule)


print("KUNG")
solver.solve(m,tee=True, symbolic_solver_labels=True)

model.A_lambda_ij = pe.Var(model.z, model.SPECIES, model.SPECIES, initialize=1.)

def A_lambda_ij_rule(m, z, s1, s2):
    return m.A_lambda_ij[z, s1, s2] == \
        ((1 + pe.sqrt(m.lambda_tr[z, s1, s2])*phi[s1, s2]**(-0.5))**2)/pe.sqrt(8 * (1 + phi[s1,s2]**-2))

model.A_lambda_ij_e = pe.Constraint(model.z, model.SPECIES, model.SPECIES,
                                    rule=A_lambda_ij_rule)
#
solver.solve(m,tee=True, symbolic_solver_labels=True)

model.lg_den = pe.Var(model.z, model.SPECIES, initialize=1.1)
# y_i("CH4")(z)*A_lambda_ij("CH4","CH4")(z) +
# y_i("CO")(z) *A_lambda_ij("CH4","CO")(z) +
# y_i("CO2")(z)*A_lambda_ij("CH4","CO2")(z) +
# y_i("H2")(z) *A_lambda_ij("CH4","H2")(z)+
# y_i("H2O")(z)*A_lambda_ij("CH4","H2O")(z)+
# y_i("N2")(z) *A_lambda_ij("CH4","N2")(z))

def lg_den_rule(m, z, s):
    return m.lg_den[z, s] == sum(m.X[z, s1]*m.A_lambda_ij[z, s, s1] for s1 in
                                 m.SPECIES)

model.lg_den_eq = pe.Constraint(model.z, model.SPECIES, rule=lg_den_rule)

solver.solve(m,tee=True, symbolic_solver_labels=True)
#
model.lambda_g = pe.Var(model.z, initialize=1.1)

def lambda_g_rule(m, z):
    return m.lambda_g[z] == sum(m.X[z, s]*m.lambda_i[z, s]/m.lg_den[z, s]
                                for s in m.SPECIES)

model.lambda_g_eq = pe.Constraint(model.z, rule=lambda_g_rule)
print("lambda_g")
solver.solve(m,tee=True, symbolic_solver_labels=True)

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

solver.solve(m,tee=True, symbolic_solver_labels=True)

tavg = np.average(tv)
model.mu_i = pe.Var(model.z, model.SPECIES, bounds=(0.0, None),
                    initialize=lambda m, z, s: A_mu[s] \
                    + B_mu[s]*tavg + C_mu[s]*tavg)
def mu_i_rule(m, z, s):
    return m.mu_i[z, s] == A_mu[s] + B_mu[s] * m.T[z] + C_mu[s] * m.T[z]**2
model.mu_i_eq = pe.Constraint(model.z, model.SPECIES, rule=mu_i_rule)

print("\n\nmu_i\n\n")
solver.solve(m,tee=True, symbolic_solver_labels=True)
#m.mu_i.setlb(0.0)
#solver.solve(m,tee=True, symbolic_solver_labels=True)

model.mu_den = pe.Var(model.z, model.SPECIES,
                      initialize=lambda m, z, s:
                      sum(pe.value(m.X[z,s1])*phi[s, s1] for s1 in m.SPECIES))

# y_i("CH4")(z)*phi("CH4","CH4")+
# y_i("CO")(z)*phi("CH4","CO")+
# y_i("CO2")(z)*phi("CH4","CO2")+
# y_i("H2")(z)*phi("CH4","H2")+
# y_i("H2O")(z)*phi("CH4","H2O")+
# y_i("N2")(z)*phi("CH4","N2"))

def mu_d_rule(m, z, s):
    return m.mu_den[z, s] == sum(m.X[z, s1]*phi[s,s1] for s1 in m.SPECIES)
model.mu_den_eq = pe.Constraint(model.z, model.SPECIES, rule=mu_d_rule)


print("\n\nmu_den\n\n")
solver.solve(m,tee=True, symbolic_solver_labels=True)

model.mu = pe.Var(model.z, bounds=(0, None),
                  initialize=lambda m, z:
                  sum(pe.value(m.X[z,s])*\
                               pe.value(m.mu_i[z,s])/pe.value(m.mu_den[z,s])
                      for s in m.SPECIES)
                  )

def mu_rule(m, z):
    return m.mu[z] == sum(m.X[z, s]*m.mu_i[z, s]/m.mu_den[z, s]
                          for s in m.SPECIES)

model.mu_eq = pe.Constraint(model.z, rule=mu_rule)
#

print("\n\nmu\n\n")
solver.solve(m,tee=True, symbolic_solver_labels=True)

#
#solver.solve(m,tee=True, symbolic_solver_labels=True)
model.CP_i = pe.Var(model.z, model.SPECIES,
                    initialize=lambda m, z, s: (A_cp[s] \
                    +B_cp[s]*pe.value(m.T[z]) \
                    +C_cp[s]*pe.value(m.T[z])**2\
                    +D_cp[s]*pe.value(m.T[z])**3\
                    +E_cp[s]*pe.value(m.T[z])**4)*(1000./MV[s])
                    )

# CP_i (Components)(z) =
# (A_cp + B_cp * T (z) + C_cp * T (z) ^ 2 + D_cp * T (z) ^ 3 + E_cp * T (z) ^4)
# * (1000/MV);//component heat capacity  [J kg-1 K-1]
def cp_i_rule(m, z, s):
    return m.CP_i[z, s] == (A_cp[s]
                            + B_cp[s]*m.T[z]
                            + C_cp[s]*m.T[z]**2
                            + D_cp[s]*m.T[z]**3
                            + E_cp[s]*m.T[z]**4) * (1000/MV[s])

model.CPi_eq = pe.Constraint(model.z, model.SPECIES, rule=cp_i_rule)

#
print("\n\nCP_i\n\n")
solver.solve(m,tee=True, symbolic_solver_labels=True)
#

#solver.solve(m,tee=True, symbolic_solver_labels=True)
model.CP_g = pe.Var(model.z, bounds=(0, None),
                    initialize=lambda m, z: \
                    sum(pe.value(m.X[z, s])*pe.value(m.CP_i[z, s])
                        for s in m.SPECIES)
                    )

# CP_g(z) = CP_i ("CH4")(z) *y_i ("CH4")(z)
# + CP_i ("CO") (z) *y_i ("CO") (z)
# + CP_i ("CO2")(z) *y_i ("CO2")(z)
# + CP_i ("H2") (z) *y_i ("H2") (z)
# + CP_i ("H2O")(z) *y_i ("H2O")(z)
# + CP_i ("N2") (z) *y_i ("N2") (z); // heat capacity of gas mixture

def cp_g_rule(m, z):
    return m.CP_g[z] == sum(m.X[z, s] * m.CP_i[z, s] for s in m.SPECIES)
model.CP_g_eq = pe.Constraint(model.z, rule=cp_g_rule)



print("\n\nCP_g\n\n")
solver.solve(m,tee=True, symbolic_solver_labels=True)

# Re (z) = 1e7 * d_p * rho_g (z) * Uz (z) / mu (z); // Renolds number
# Pr (z) = 1e-7 * Cp_g (z) * mu (z) / lambda_g (z); // Prandtl number

# U (z) = 0.4 * (lambda_g(z)/d_p) * (2.58 * Re(z)^(1/3) *
# Pr(z)^(1/3) + 0.094 * Re(z)^0.8 * Pr(z)^0.4);

model.Re = pe.Var(model.z, bounds=(0, None),
                  initialize=lambda m,z: 1e7*d_p*rho_g
                  *pe.value(m.u)/pe.value(m.mu[z])
)

def re_rule(m, z):
    return m.Re[z] == 1e7 * d_p * rho_g * m.u / m.mu[z]
model.Re_eq = pe.Constraint(model.z, rule=re_rule)

print("\n\nRe\n\n")
solver.solve(m,tee=True, symbolic_solver_labels=True)


model.Pr = pe.Var(model.z, bounds=(0, None),
                  initialize=lambda m, z: 1e-7*pe.value(m.CP_g[z])
                  *pe.value(m.mu[z])/pe.value(m.lambda_g[z])
                  )


def pr_rule(m, z):
    return m.Pr[z] == 1e-7 * m.CP_g[z] * m.mu[z] / m.lambda_g[z]
model.Pr_eq = pe.Constraint(model.z, rule=pr_rule)


print("\n\nPr\n\n")
solver.solve(m,tee=True, symbolic_solver_labels=True)

model.U = pe.Var(model.z, bounds=(0, None),
                 initialize=lambda m,z:0.4*(pe.value(m.lambda_g[z])/d_p)*
                 (2.58*(pe.value(m.Re[z])**0.333)*
                  (pe.value(m.Pr[z])**0.333)+0.094*(pe.value(m.Re[z]**0.8))*
                  (pe.value(m.Pr[z])**0.4))
                 )

#     U (z) = 0.4 * (lambda_g(z)/d_p) * (2.58 * Re(z)^(1/3) * Pr(z)^(1/3)
#+ 0.094 * Re(z)^0.8 * Pr(z)^0.4);

def u_rule(m, z):
    return m.U[z] == \
        0.4*(m.lambda_g[z]/d_p)*(2.58*(m.Re[z]**0.333)*(m.Pr[z]**0.333)\
                                 +0.094*(m.Re[z]**0.8)*(m.Pr[z]**0.4))

model.U_eq = pe.Constraint(model.z, rule=u_rule)

print("\n\nU\n\n")
solver.solve(m,tee=True, symbolic_solver_labels=True)


def Def_newdT_rule(m,z):
    return m.u*rho_g*CP_g* m.dT[z] == \
        4* m.U[z] * (Tw - m.T[z])/d_t_in \
        + (1-eb)*rho_p*1000*1000\
        *sum(eta[r]*m.Rate[z,r] * (-dHr[r]) for r in model.REACTIONS)

model.del_component(model.Def_dT)
#model.del_component(model.Def_dT_index)

# energy balance
model.Def_dT = pe.Constraint(model.z, rule =  Def_newdT_rule)


print("\n\nnew energy balance\n\n")
solver.solve(m,tee=True, symbolic_solver_labels=True)



k = 0
cvm = np.zeros(2)
for i in m.SPECIES:
    fv = np.array(pe.value(m.C[:, i]))
    if k == 0:
        cvm = fv
    else:
        cvm = np.vstack([cvm, fv])
    k += 1



#model.cdum = pe.Var(model.z, model.SPECIES, initialize=1e-4)
#model.d2C = dae.DerivativeVar(model.cdum, wrt=model.z)

m.d2C_disc_eq.activate()
# cdum = dC
model.Def_cdum = pe.Constraint(model.z, model.SPECIES,
                               rule=lambda m, z, s: m.cdum[z, s] == m.dC[z, s])

model.bc_dc = pe.Constraint(model.SPECIES,
                            rule=lambda m, s: m.cdum[m.L, s] == 0.0)

dscale = 1e-05
#
def Def_C_2_rule(m, z, c):
    return m.u*m.cdum[z, c] == eb * dscale * m.d2C[z, c] \
        +(1-eb)*rho_p*m.r_comp[z, c]

model.Def_C_2 = pe.Constraint(model.z, model.SPECIES, rule=Def_C_2_rule)

model.del_component(model.Def_C_)
model.del_component(model.Def_C__index)


print("\n\nnew component balance\n\n")
solver.solve(m,tee=True, symbolic_solver_labels=True)


#model.tdum = pe.Var(model.z, initialize=1e-4)
#model.d2T = dae.DerivativeVar(model.tdum, wrt=model.z)

m.d2T_disc_eq.activate()
# tdum = dT
model.Def_tdum = pe.Constraint(model.z,
                               rule=lambda m, z: m.tdum[z] == m.dT[z])


model.bc_dt = pe.Constraint(rule=lambda m: m.tdum[m.L] == 0.0)


model.tsddc = pe.Var(model.z,
                     rule=lambda m, z:
                     sum(eb*dscale*pe.value(m.cdum[z,s])*pe.value(m.CP_i[z,s])
                         for s in m.SPECIES)
                     )

def Def_tsddc_rule(m, z):
    return m.tsddc[z] == \
        sum(eb*dscale*m.cdum[z,s]*m.CP_i[z, s] for s in m.SPECIES)

model.Def_tsddc = pe.Constraint(model.z, rule=Def_tsddc_rule)


def Def_newdT_2_rule(m,z):
    return m.u*rho_g*m.CP_g[z]* m.tdum[z] == \
        4* m.U[z] * (Tw - m.T[z])/d_t_in \
        + m.tsddc[z] * m.tdum[z] \
        + m.lambda_g[z] *m.d2T[z] \
        + (1-eb)*rho_p*1000*1000\
    *sum(eta[r]*m.Rate[z,r] * (-dHr[r]) for r in model.REACTIONS)

model.Def_dT_2 = pe.Constraint(model.z, rule=Def_newdT_2_rule)


model.del_component(model.Def_dT)
#model.del_component(model.Def_dT_index)

print("\n\nnew temp balance\n\n")
solver.solve(m,tee=True, symbolic_solver_labels=True)





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
model.omegaD = pe.Var(model.z, model.SPECIES, model.SPECIES,
                      bounds = (0,None), initialize=1)
#-=-##-=-#

for (z, s1, s2) in product(model.z, model.SPECIES, model.SPECIES):
    tv = pe.value(m.T[z])
    print(tv, epsik[s1], epsik[s2])
    val = 1.06036/(tv*pe.sqrt(epsik[s1]*epsik[s2]))**0.1561 \
        + 0.19300/pe.exp(0.47635*tv/pe.sqrt(epsik[s1]*epsik[s2])) \
        + 1.03587/pe.exp(1.52996*tv/pe.sqrt(epsik[s1]*epsik[s2])) \
        + 1.76474/pe.exp(3.89411*tv/pe.sqrt(epsik[s1]*epsik[s2]))
    m.omegaD[z, s1, s2].set_value(val)

def omega_d_rule(m, z, s1, s2):
    return m.omegaD[z, s1, s2] == \
    1.06036/(m.T[z]*pe.sqrt(epsik[s1]*epsik[s2]))**0.1561 \
    + 0.19300/pe.exp(0.47635*m.T[z]/pe.sqrt(epsik[s1]*epsik[s2])) \
    + 1.03587/pe.exp(1.52996*m.T[z]/pe.sqrt(epsik[s1]*epsik[s2])) \
    + 1.76474/pe.exp(3.89411*m.T[z]/pe.sqrt(epsik[s1]*epsik[s2]))

model.def_omegad = pe.Constraint(model.z, model.SPECIES, model.SPECIES,
                                 rule=omega_d_rule)

print("\n*=**=**=**=**=**=**=*\n")
solver.solve(m,tee=True, symbolic_solver_labels=True)

#-=-##-=-#
model.Dij_ = pe.Var(model.z, model.SPECIES, model.SPECIES, bounds=(0, None),
                    initialize=1)
#-=-##-=-#
for (z, s1, s2) in product(model.z, model.SPECIES, model.SPECIES):
    tv = pe.value(m.T[z])
    ptv = pe.value(m.Pt[z])
    omegadv = pe.value(m.omegaD[z, s1, s2])
    val = 0.00266*tv**(3/2) \
    /(ptv*(MV_ij[s1, s2]**0.5)*(sigma_ij[s1,s2]**2)*omegadv)
    m.Dij_[z, s1, s2].set_value(val)


def dij_f_rule(m, z, s1, s2):
    return m.Dij_[z, s1, s2] == \
        (0.00266*m.T[z]**(3/2))/(m.Pt[z] \
                                 *pe.sqrt(MV_ij[s1, s2]) \
                                 *(sigma_ij[s1, s2]**2)*m.omegaD[z, s1, s2])

model.def_dij = pe.Constraint(model.z, model.SPECIES, model.SPECIES,
                              rule=dij_f_rule)


print("\n*=**=**=**=**=**=**=*\n")
solver.solve(m,tee=True, symbolic_solver_labels=True)

#-=-##-=-#
model.Dim_ = pe.Var(model.z, model.SPECIES, bounds = (0, None), initialize=1)
#-=-##-=-#
for (z, s1) in product(model.z, model.SPECIES):
    val = sum(pe.value(m.X[z, s2])/pe.value(m.Dij_[z, s1, s2])
              for s2 in model.SPECIES if s2 != s1)
    model.Dim_[z, s1].set_value(1/val)

def dim_rule_(m, z, s1):
    return model.Dim_[z, s1] == sum(m.X[z, s2]/m.Dij_[z, s1, s2] for s2 in
                             model.SPECIES if s2 != s1)

model.def_dim_ = pe.Constraint(model.z, model.SPECIES, rule=dim_rule_)

print("\n*=**=**=**=**=**=**=*\n")
solver.solve(m,tee=True, symbolic_solver_labels=True)

#-=-##-=-#-
model.Diez_ = pe.Var(model.z, model.SPECIES, bounds=(0, None), initialize=1)
#-=-##-=-#-

for (z, s1) in product(model.z, model.SPECIES):
    val = 0.78*pe.value(model.Dim_[z, s1]) \
    + (0.54*pe.value(model.u)*d_cata/eb) \
    /(1 + 9.2*pe.value(model.Dim_[z,s1])/(pe.value(model.u)*d_cata/eb))
    model.Diez_[z, s1].set_value(val)

def diez_rule(m, z, s1):
    return model.Diez_[z, s1] == 0.78*model.Dim_[z, s1] + \
        (0.54*model.u*d_cata/eb)/(1 + 9.2*model.Dim_[z,s1]/(model.u*d_cata/eb))

model.def_diez_ = pe.Constraint(model.z, model.SPECIES, rule=diez_rule)

print("\n*=**=**=**=**=**=**=*\n")
solver.solve(m,tee=True, symbolic_solver_labels=True)



model.del_component(model.Def_C_2)
model.del_component(model.Def_C_2_index)

def Def_C_2_rule(m, z, c):
    return m.u*m.cdum[z, c] == eb * m.Diez_[z, c] * m.d2C[z, c] \
        +(1-eb)*rho_p*m.r_comp[z, c]

model.Def_C_2 = pe.Constraint(model.z, model.SPECIES, rule=Def_C_2_rule)

#
###########
model.del_component(model.Def_tsddc)
for z in model.z:
    val = sum(eb*pe.value(m.Diez_[z, s])*pe.value(m.cdum[z,s])*pe.value(m.CP_i[z,s])
             for s in m.SPECIES)
    model.tsddc[z].set_value(val)


def Def_tsddc_rule(m, z):
    return m.tsddc[z] == \
        sum(eb*m.Diez_[z, s]*m.cdum[z,s]*m.CP_i[z, s] for s in m.SPECIES)

model.Def_tsddc = pe.Constraint(model.z, rule=Def_tsddc_rule)


print("\n*=**=**=**=**=**=**=*\n")
solver.solve(m,tee=True, symbolic_solver_labels=True)


k = 0
cvm = np.zeros(2)
for i in m.SPECIES:
    fv = np.array(pe.value(m.C[:, i]))
    if k == 0:
        cvm = fv
    else:
        cvm = np.vstack([cvm, fv])
    k += 1

tv = np.array(pe.value(m.T[:]))
yv = np.array(pe.value(m.y[:]))
zval = np.array(m.z.data())

f, a = plt.subplots(1, 2, figsize=[4*2, 3])

a[0].plot(zval, tv, "mo-")
a[0].set_title("Temperature")
a[1].plot(zval, yv, "mo-")

a[1].set_title("y (dimensionless pressure)")
a[1].set_ylim(bottom=0)

f.tight_layout()
f.savefig("results_v0709_a.png")

nfigs = cvm.shape[0]
f, a = plt.subplots(nfigs, 1, figsize=[4, 3*nfigs])

for i in range(cvm.shape[0]):
    a[i].plot(zval, cvm[i, :], "mo-", label=m.SPECIES.data()[i])
    a[i].set_title(m.SPECIES.data()[i])
    a[i].set_ylabel("Concentration kmol m^-3")
    #a[i].set_ylim(bottom=0)

f.tight_layout()
f.savefig("results_v0709b_2.png")

# flow
k = 0
fvm = np.zeros(2)
for i in m.SPECIES:
    fv = np.array(pe.value(m.F[:, i]))
    if k == 0:
        fvm = fv
    else:
        fvm = np.vstack([fvm, fv])
    k += 1


nfigs = fvm.shape[0]
f, a = plt.subplots(nfigs, 1, figsize=[4, 3*nfigs])

for i in range(fvm.shape[0]):
    a[i].plot(zval, fvm[i, :], "mo-", label=m.SPECIES.data()[i])
    a[i].set_title(m.SPECIES.data()[i])
    a[i].set_ylabel("Flow kmol h^-1")
    a[i].set_ylim(bottom=0)

f.tight_layout()
f.savefig("results_v0709_c.png")
