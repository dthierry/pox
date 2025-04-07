#!/usr/bin/env python
# coding: utf-8

import pyomo.environ as pe
import pyomo.dae as dae
import numpy as np
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# # Parameters and data:

# constraints for reactions:
Ak1 = 4.255e15   # pre-exponential factor of rate constant of reaction 1
Ak2 = 1.955e6    # pre-exponential factor of rate constant of reaction 2
Ak3 = 1.02e15    # pre-exponential factor of rate constant of reaction 3
Ak4a = 2.92e6
Ak4b = 2.46e6
E1 = 240.1       # kJ/mol activation energy of reaction 1
E2 = 67.13       # kJ/mol activation energy of reaction 2
E3 = 243.9       # kJ/mol activation energy of reaction 3
E4a = 86         # kJ/mol activation energy of reaction 4
E4b = 86         # kJ/mol activation energy of reaction 4

dHox_CH4 = 27.3   # kJ/mol enthalpy of oxidation of Ch4
dHox_O2 = 92.8    # kJ/mol enthalpy of oxidation of O2

AKox_CH4 = 1.26e-1 # pre-exponential factor of adsorption constant of CH4 oxidation
AKox_O2 = 7.87e-7  # pre-exponential factor of adsorption constant of O2 oxidation


R = 0.0083144621  # kJ/(K*mol) gas constant
R_ = 8.3144621    # J/(K*mol)  gas constant

dHa = {'CH4': -38.28, 'H2O': 88.68, 'H2': -82.9, 'CO': -70.6} # adsorption enthalpy
# pre-exponential factor of adsorption constant
AKa = {'CH4': 6.65e-4 , 'H2O': 1.77e5, 'H2': 6.121e-9, 'CO': 8.23e-5}
dHr = {1:223_078.0, 2:-36_584.0, 3:186_494.0, 4:-802_625.0}  # reaction enthalpy
# dhr in kJ/kmol
eta = {1:0.07, 2:0.07, 3:0.06, 4:0.05}          # eta
epsilon = 0.85                                  # bed void fraction

# @dav: radius = 0.03
radius = 0.03/2                                   # m  radius

Ac = np.pi * radius**2                          # m^2 cross sectional area
rho_cata = 1820                                 # kg/m3 catalyst density
d_cata = 5e-3                                   # m  catalyst diameter
rho_ave = 1.6                                   # kg/m3  gas average density
mu_ave = 2.97e-5                                # Pa*s  gas average viscosity

# @dav: Cp_ave should have J/kg-K units
Cp_ave = 41.06                                  # J/(mol*K) gas mixture specific heat
# shouldn't this be in kg-1?

# # Define variables and parameters

model = m = pe.ConcreteModel()

# model reactor parameters
model.L = pe.Param(initialize = 3)  # reactor total length
model.z = dae.ContinuousSet(bounds = (0,model.L))   # z direction
model.SPECIES = pe.Set(initialize = ['CH4','H2O','H2','CO','CO2','O2'])
model.REACTIONS = pe.RangeSet(4)
print(model.REACTIONS.data())
model.P0 = pe.Param(initialize = 10.) # inlet pressure
model.T0 = pe.Param(initialize = 720.) # inlet temperature

# define reactor inlet variables

model.FCH4_in = pe.Param(initialize = 30)    # fix inlet CH4
model.F_in = pe.Var(model.SPECIES, bounds=(0,None))   # inlet flow rate kmol/h
model.Ft_in = pe.Var(bounds = (0,None))   # total inlet flow rate kmol/h
model.X_in = pe.Var(model.SPECIES)   # inlet mole fraction
model.S_C = pe.Var(bounds = (0,1), initialize = 0.17) # Steam to Carbon ratio
# shouldn't this be below 1
model.C_O = pe.Var(bounds = (0,1), initialize = 0.5)  # Carbon to Oxygen ratio
model.u = pe.Var(bounds = (0,None), initialize = 4.5) # superfacial velocity
model.Rep = pe.Var(bounds = (0,None))   # practical Raynolds number
model.f = pe.Var(bounds = (0,None)) # friction factor

# define variables that vary along reactor length

model.F = pe.Var(model.z, model.SPECIES, bounds = (0,None))
model.y = pe.Var(model.z, bounds = (0,None))
model.Pt = pe.Var(model.z, bounds = (0,None))
model.T = pe.Var(model.z, bounds = (1e1,None))
model.Ft = pe.Var(model.z, bounds = (0,None))
model.P = pe.Var(model.z, model.SPECIES, bounds = (0,None), initialize=1e0)
model.Rate = pe.Var(model.z, model.REACTIONS)
model.DEN = pe.Var(model.z, initialize=1e0)

    # define coeffs
model.k1 = pe.Var(model.z, bounds=(0, None))
model.k2 = pe.Var(model.z)
model.k3 = pe.Var(model.z, bounds=(0, None))
model.k4a = pe.Var(model.z)
model.k4b = pe.Var(model.z)

model.Ke1 = pe.Var(model.z, bounds=(0e-08, None), initialize=1e4)
model.Ke2 = pe.Var(model.z, bounds=(0e-00, None), initialize=1e2)
model.Ke3 = pe.Var(model.z, bounds=(0e-00, None), initialize=1e2)

model.Ka = pe.Var(model.z, model.SPECIES, bounds=(0e-00, None), initialize=1e2)
model.Kox_CH4 = pe.Var(model.z, bounds=(0, None))
model.Kox_O2 = pe.Var(model.z, bounds=(0, None))

# define derivative variables
model.dF = dae.DerivativeVar(model.F, wrt = model.z)
model.dy = dae.DerivativeVar(model.y, wrt = model.z)
model.dT = dae.DerivativeVar(model.T, wrt = model.z)


# # Constraints
# ## Reaction coefficients equations

def defk1_rule(m,z):        # kmol*bar^(1/2)/(kgcat*hr) rate constant of reaction 1
    return m.k1[z] == Ak1*pe.exp(-E1/(R*m.T[z]))
model.defk1 = pe.Constraint(model.z, rule = defk1_rule)

def defk2_rule(m,z):        # kmol/(kgcat*hr)           rate constant of reaction 2
    return m.k2[z] == Ak2*pe.exp(-E2/(R*m.T[z]))
model.defk2 = pe.Constraint(model.z, rule = defk2_rule)

def defk3_rule(m,z):        # kmol*bar^(1/2)/(kgcat*hr) rate constant of reaction 3
    return m.k3[z] == Ak3*pe.exp(-E3/(R*m.T[z]))
model.defk3 = pe.Constraint(model.z, rule = defk3_rule)

def defk4a_rule(m,z):       # kmol*bar^(1/2)/(kgcat*hr) rate constant of reaction 4a
    return m.k4a[z] == Ak4a*pe.exp(-E4a/(R*m.T[z]))
model.defk4a = pe.Constraint(model.z, rule = defk4a_rule)

def defk4b_rule(m,z):       # kmol*bar^(1/2)/(kgcat*hr) rate constant of reaction 4b
    return m.k4b[z] == Ak4b*pe.exp(-E4b/(R*m.T[z]))
model.defk4b = pe.Constraint(model.z, rule = defk4b_rule)


def defKe1_rule(m,z):       # bar^2   rate equilibrium constant of reaction 1
    #return m.Ke1[z] == 1.198e17*pe.exp(-26830/(m.T[z]))
    return pe.log(m.Ke1[z]/1.198e17) == (-26830/(m.T[z]))
model.defKe1 = pe.Constraint(model.z, rule = defKe1_rule)

def defKe2_rule(m,z):       #   rate equilibrium constant of reaction 2
    #return m.Ke2[z] == 1.767e-2*pe.exp(4400/(m.T[z]+1e-5))
    return pe.log(m.Ke2[z] / 1.767e-2) == 4400/(m.T[z])
model.defKe2 = pe.Constraint(model.z, rule = defKe2_rule)

def defKe3_rule(m,z):       #   rate equilibrium constant of reaction 3
    return m.Ke3[z] == 2.117e15*pe.exp(-22430/(m.T[z]))
model.defKe3 = pe.Constraint(model.z, rule = defKe3_rule)

def defKa_rule(m, z, s):   # adsorption constant for each species
    if s == 'CO2' or s == 'O2':
        return pe.Constraint.Skip
    #return m.Ka[z,s] == AKa[s]*pe.exp(-dHa[s]/(R*m.T[z]+1e-5))
    return pe.log(m.Ka[z,s] / AKa[s])== (-dHa[s]/(R*m.T[z]))
model.defKa = pe.Constraint(model.z, model.SPECIES, rule = defKa_rule)

def defKox_CH4_rule(m,z):  # adsorption constant for CH4 oxidation
    return m.Kox_CH4[z] == AKox_CH4*pe.exp(-dHox_CH4/(R*m.T[z]))
model.defKox_CH4 = pe.Constraint(model.z, rule = defKox_CH4_rule)

def defKox_O2_rule(m,z):   # adsorption constant for O2 oxidation
    return m.Kox_O2[z] == AKox_O2*pe.exp(-dHox_O2/(R*m.T[z]))
model.defKox_O2 = pe.Constraint(model.z, rule = defKox_O2_rule)


# ## Inlet Condition Constraints

# define inlet mole fractions:

# define velocity
def Def_Ft_in_rule(m):
    return m.Ft_in == sum(m.F_in[s] for s in model.SPECIES)
model.Def_Ft_in = pe.Constraint(rule = Def_Ft_in_rule)

model.F_in['CH4'].fix(model.FCH4_in) # fix CH4 inlet
model.F_in['H2O'].fix(5) # fix CH4 inlet
model.F_in['O2'].fix(20) # fix CH4 inlet
model.F_in['CO'].fix(0.01) # fix CH4 inlet
model.F_in['CO2'].fix(0.01) # fix CH4 inlet
model.F_in['H2'].fix(0.01) # fix CH4 inlet

def Flow_fraction_relation_rule(m,s):
    return m.F_in[s] == m.Ft_in * m.X_in[s]
model.Flow_fraction_relation = pe.Constraint(model.SPECIES, rule = Flow_fraction_relation_rule)

def IdealGasLaw_in_rule(m):
    return m.Ft_in *1000/3600 * R_ * m.T0 == m.P0*1e5 * m.u * Ac
model.IdealGasLaw_in = pe.Constraint(rule = IdealGasLaw_in_rule)

def Def_Rep_rule(m):            # partical Raynolds number
    return m.Rep == m.u * rho_ave * d_cata / mu_ave
model.Def_Rep= pe.Constraint(rule = Def_Rep_rule)

def Def_friction_rule(m):  # friction factor
    return m.f == 150 + 1.75*m.Rep/(1-epsilon)
model.Def_friction= pe.Constraint(rule = Def_friction_rule)


# ## ODE Constraints

# ODEs Constraints

def Pressure_nondimensionalize_rule(m,z):  # total pressure of pcl = nondementionalized pressure of pcl * Initial pressure of pcl, here y = P0/P
    return m.Pt[z] == m.y[z] * m.P0
model.Pressure_nondimensionalize = pe.Constraint(model.z, rule = Pressure_nondimensionalize_rule)

def Total_Flow_rule(m,z): # total flow rate is sum of single species flow rate
    return m.Ft[z] == sum(m.F[z,s] for s in model.SPECIES)
model.Total_Flow = pe.Constraint(model.z, rule = Total_Flow_rule)

def Patial_pressure_rule(m,z,s):  # patial pressure
    #return m.P[z,s] == m.F[z,s] / m.Ft[z] * m.Pt[z]
    return m.P[z,s] * m.Ft[z] == m.F[z,s] * m.Pt[z]
model.Patial_pressure = pe.Constraint(model.z, model.SPECIES, rule =  Patial_pressure_rule)

def Def_DEN_rule(m,z):  # define DEN
    return m.DEN[z] * m.P[z,'H2'] == m.P[z,'H2'] \
    + m.Ka[z,'CO']*m.P[z,'CO'] * m.P[z,'H2'] \
    + m.Ka[z,'CH4']*m.P[z,'CH4'] * m.P[z,'H2'] \
    + m.Ka[z,'H2']*m.P[z,'H2'] * m.P[z,'H2'] \
    + m.Ka[z,'H2O']*m.P[z,'H2O']
model.Def_DEN = pe.Constraint(model.z, rule =  Def_DEN_rule)


def Def_Rate1_rule(m,z):   # kmol/(kgcat*h) rate law for reaction 1
    return m.Rate[z,1] == eta[1] * m.k1[z] \
        * (m.P[z,'CH4']*m.P[z,'H2O']/m.P[z,'H2']**2.5 \
        -  m.P[z,'CO']*(m.P[z,'H2'])**0.5/m.Ke1[z])/m.DEN[z]**2
model.Def_Rate1 = pe.Constraint(model.z, rule =  Def_Rate1_rule)

def Def_Rate2_rule(m,z):   # kmol/(kgcat*h) rate law for reaction 2
    return m.Rate[z,2] == \
        eta[2] * m.k2[z]*(m.P[z,'CO']*m.P[z,'H2O']/m.P[z,'H2']-m.P[z,'CO2']/m.Ke2[z])/m.DEN[z]**2
model.Def_Rate2 = pe.Constraint(model.z, rule =  Def_Rate2_rule)

def Def_Rate3_rule(m,z):   # kmol/(kgcat*h) rate law for reaction 1
    return m.Rate[z,3] == \
        eta[3] * m.k3[z]*(m.P[z,'CH4']*m.P[z,'H2O']**2/m.P[z,'H2']**3.5-\
                                            m.P[z,'CO2']*m.P[z,'H2']**0.5/m.Ke3[z])/m.DEN[z]**2
model.Def_Rate3 = pe.Constraint(model.z, rule =  Def_Rate3_rule)

def Def_Rate4_rule(m,z):   # kmol/(kgcat*h) rate law for reaction 1
    return m.Rate[z,4]== eta[4]*(m.k4a[z]*m.P[z,'CH4']*m.P[z,'O2']/(1+m.Kox_CH4[z]*m.P[z,'CH4']+m.Kox_O2[z]*m.P[z,'O2'])**2\
                                + m.k4b[z]*m.P[z,'CH4']*m.P[z,'O2']/(1+m.Kox_CH4[z]*m.P[z,'CH4']+m.Kox_O2[z]*m.P[z,'O2']))
model.Def_Rate4 = pe.Constraint(model.z, rule =  Def_Rate4_rule)

def Def_FCH4_rule(m,z):
    return m.dF[z,'CH4'] ==  (-m.Rate[z,1]-m.Rate[z,3]-m.Rate [z,4])*rho_cata*(1-epsilon)*Ac
model.Def_FCH4 = pe.Constraint(model.z, rule =  Def_FCH4_rule)

def Def_FH2O_rule(m,z):
    return m.dF[z,'H2O'] ==  (-m.Rate[z,1]-m.Rate[z,2]-2*m.Rate [z,3]+2*m.Rate [z,4])*rho_cata*(1-epsilon)*Ac
model.Def_FH2O = pe.Constraint(model.z, rule =  Def_FH2O_rule)

def Def_FH2_rule(m,z):
    #return m.dF[z,'H2'] ==  (3*m.Rate[z,1]+m.Rate[z,2]-4*m.Rate [z,3])*rho_cata*(1-epsilon)*Ac
    return m.dF[z,'H2'] ==  (3*m.Rate[z,1]+m.Rate[z,2]+4*m.Rate [z,3])*rho_cata*(1-epsilon)*Ac
model.Def_FH2 = pe.Constraint(model.z, rule =  Def_FH2_rule)

def Def_FCO_rule(m,z):
    return m.dF[z,'CO'] ==  (m.Rate[z,1]-m.Rate[z,2])*rho_cata*(1-epsilon)*Ac
model.Def_FCO = pe.Constraint(model.z, rule =  Def_FCO_rule)

def Def_FCO2_rule(m,z):
    return m.dF[z,'CO2'] ==  (m.Rate[z,2]+m.Rate[z,3]+m.Rate[z,4])*rho_cata*(1-epsilon)*Ac
model.Def_FCO2 = pe.Constraint(model.z, rule =  Def_FCO2_rule)

def Def_dy_rule(m,z):
    return m.dy[z] == -1/(m.P0*1e5)*m.f*m.u*mu_ave/(d_cata**2)*(1-epsilon)**2/epsilon**3
model.Def_dy = pe.Constraint(model.z, rule =  Def_dy_rule)

def Def_dT_rule(m,z):
    #return m.dT[z] == -(Ac*1000/3600*rho_cata*(1-epsilon)*sum(dHr[r]*m.Rate[z,r] for r in model.REACTIONS))\
    return m.dT[z] == (1000/3600*rho_cata*(1-epsilon)*sum(dHr[r]*m.Rate[z,r] for r in model.REACTIONS))\
                    /(m.u*rho_ave*Cp_ave)
    # I think the Ac should not be here
model.Def_dT = pe.Constraint(model.z, rule =  Def_dT_rule)
# u is in m / s

# Outlet 2:1 Ratio
# def Outlet_ratio_rule(m):
#     return m.F[1,'H2'] == 2*m.F[1,'CO']

# model.Outlet_ratio = pe.Constraint(rule = Outlet_ratio_rule)



def Def_FO2_rule(m,z):
    return m.dF[z,'O2'] ==  -2.*m.Rate[z,4]*rho_cata*(1-epsilon)*Ac
model.Def_FO2 = pe.Constraint(model.z, rule =  Def_FO2_rule)



# ##  Initial Conditions

# Initial conditions

def InitCon_rule(m):
    for s in model.SPECIES:
        yield m.F[0,s] == m.F_in[s]
    yield m.y[0] == 1
    yield m.T[0] == m.T0
model.InitCon = pe.ConstraintList(rule = InitCon_rule)

# Dummy optimize function
# model.obj = pe.Objective(expr=1) # Dummy Objective
def Objective_rule(m):
    return 1.0
model.obj = pe.Objective(rule = Objective_rule) # Dummy Objective


# # Solve

discretizer = pe.TransformationFactory('dae.collocation')
discretizer.apply_to(m,nfe=6,ncp=7,scheme='LAGRANGE-RADAU')

m.P[:, 'H2'].setlb(0e-00)
m.P[:, 'H2O'].setlb(0e-00)


solver = pe.SolverFactory('/Users/dthierry/Apps/ipopt_102623/ip_dir/bin/ipopt')

solver.options['halt_on_ampl_error'] = 'yes'

ntry = 0

with open("ipopt.opt", "w") as f:
    f.write("start_with_resto\tyes\n")
    f.write("linear_solver\tmumps\n")
    f.write("expect_infeasible_problem\tyes\n")
    f.write("mu_strategy\tadaptive\n")
    f.write("output_file\tout.txt\n")
    f.write("print_info_string\tyes\n")
    #f.write("print_level\t1\n")
    #
    f.write("max_iter\t71\n")

ntry = 1 # 1
print(f"TRY {ntry}")
m.write("my_nl.nl")
solver.solve(m,tee=True, symbolic_solver_labels=True)


with open("ipopt.opt", "w") as f:
    f.write("start_with_resto\tyes\n")
    f.write("linear_solver\tmumps\n")
    f.write("expect_infeasible_problem\tyes\n")
    f.write("mu_strategy\tadaptive\n")
    f.write("output_file\tout.txt\n")
    f.write("print_info_string\tyes\n")


m.Ke1.setlb(1e-8)
m.Ka.setlb(1e-8)
ntry = 2 # 2
print(f"TRY {ntry}")
solver.solve(m,tee=True, symbolic_solver_labels=True)

with open("ipopt.opt", "w") as f:
    #f.write("linear_solver\tma57\n")
    f.write("linear_solver\tmumps\n")
    #f.write("expect_infeasible_problem\tyes\n")
    f.write("mu_strategy\tadaptive\n")
    f.write("output_file\tout.txt\n")
    f.write("print_info_string\tyes\n")
    #
    #f.write("constr_viol_tol\t1e-02\n")
    #f.write("dual_inf_tol\t1e+03\n")
    #f.write("compl_inf_tol\t1e-02\n")
    f.write("max_iter\t158\n")


m.P[:, 'H2'].setlb(1e-08)
ntry = 3 # 3
print(f"TRY {ntry}")
solver.solve(m,tee=True, symbolic_solver_labels=True)

with open("ipopt.opt", "w") as f:
    #f.write("linear_solver\tma57\n")
    f.write("linear_solver\tmumps\n")
    f.write("expect_infeasible_problem\tyes\n")
    f.write("mu_strategy\tadaptive\n")
    f.write("output_file\tout.txt\n")
    f.write("print_info_string\tyes\n")
    f.write("bound_push\t1e-06\n")
    f.write("max_iter\t1020\n")

ntry = 4 # 4
print(f"TRY {ntry}")
solver.solve(m,tee=True, symbolic_solver_labels=True)

with open("ipopt.opt", "w") as f:
    #f.write("linear_solver\tma57\n")
    f.write("linear_solver\tmumps\n")
    #f.write("expect_infeasible_problem\tyes\n")
    #f.write("mu_strategy\tadaptive\n")
    f.write("output_file\tout.txt\n")
    f.write("print_info_string\tyes\n")
    f.write("tol\t1e+03\n")
    f.write("max_iter\t866\n")

ntry = 5 # 4
print(f"TRY {ntry}")
solver.solve(m,tee=True, symbolic_solver_labels=True)


with open("ipopt.opt", "w") as f:
    #f.write("linear_solver\tma57\n")
    f.write("linear_solver\tmumps\n")
    #f.write("expect_infeasible_problem\tyes\n")
    #f.write("mu_strategy\tadaptive\n")
    f.write("output_file\tout.txt\n")
    f.write("print_info_string\tyes\n")
    f.write("constr_viol_tol\t6.0\n")
    f.write("compl_inf_tol\t1e-02\n")
    f.write("dual_inf_tol\t1e+03\n")
    f.write("tol\t1e+03\n")
    f.write("max_iter\t675\n")

ntry = 6 # 4
print(f"TRY {ntry}")
solver.solve(m,tee=True, symbolic_solver_labels=True)

with open("ipopt.opt", "w") as f:
    #f.write("linear_solver\tma57\n")
    f.write("linear_solver\tmumps\n")
    #f.write("expect_infeasible_problem\tyes\n")
    #f.write("mu_strategy\tadaptive\n")
    f.write("output_file\tout.txt\n")
    f.write("print_info_string\tyes\n")
    f.write("constr_viol_tol\t6.0\n")
    f.write("compl_inf_tol\t1e-02\n")
    f.write("dual_inf_tol\t1e+03\n")
    f.write("tol\t1e+03\n")
    f.write("max_iter\t855\n")

ntry = 7 # 4
print(f"try {ntry}")
solver.solve(m,tee=True, symbolic_solver_labels=True)

with open("ipopt.opt", "w") as f:
    f.write("linear_solver\tma57\n")
    #f.write("linear_solver\tmumps\n")
    #f.write("expect_infeasible_problem\tyes\n")
    #f.write("mu_strategy\tadaptive\n")
    f.write("bound_push\t1e-08\t")
    f.write("output_file\tout.txt\n")
    f.write("print_info_string\tyes\n")
    f.write("max_iter\t3515\n")

m.Ke1.setlb(0e-8)
m.Ka.setlb(0e-0)
m.P[:, 'H2'].setlb(1e-08)
ntry = 8 # 4
print(f"try {ntry}")
solver.solve(m,tee=True, symbolic_solver_labels=True)


# 80 ###########################################################################
# plots
k = 0
fvm = np.zeros(2)
for i in m.SPECIES:
    fv = np.array(pe.value(m.F[:, i]))
    if k == 0:
        fvm = fv
    else:
        fvm = np.vstack([fvm, fv])
    k += 1

tv = np.array(pe.value(m.T[:]))
yv = np.array(pe.value(m.y[:]))
zval = np.array(m.z.data())

nfigs = fvm.shape[0]+2
f, a = plt.subplots(nfigs, 1, figsize=[4, 3*nfigs])

a[0].plot(zval, tv)
a[0].set_title("Temperature")
a[1].plot(zval, yv)
a[1].set_title("y")
for i in range(fvm.shape[0]):
    a[i+2].plot(zval, fvm[i, :], label=m.SPECIES.data()[i])
    a[i+2].set_title(m.SPECIES.data()[i])

f.tight_layout()
f.savefig("restults_v11_5.png")


