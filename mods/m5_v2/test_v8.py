from pyomo.environ import *
import matplotlib.pyplot as plt
import numpy as np
import sys

from m5_v2 import m
sys.path.insert(1, "../utils/")
sys.path.insert(1, "../m5/")
from sav_state_v2 import save_m5_state, load_m5_state
from m5_plt import PoxResult

print("load state")
load_m5_state(m, m2=True)
print("state loaded")


ipexe = "/Users/dthierry/Apps/ipopt_dir/bin/ipopt"
solver = SolverFactory(ipexe)

Tw = 1000  # K
m.bc_C_R_0.deactivate()
m.bc_C_R_r.deactivate()
m.bc_T_R_0.deactivate()
m.bc_T_R_r.deactivate()

def of_rule(m):
    return sum((m.C_R[z, 0, s]/m.C_in["CH4"])**2 for z in m.z for s in m.SPECIES) + \
        sum((m.C_R[z, m.R, s]/m.C_in["CH4"])**2 for z in m.z for s in m.SPECIES) + \
        sum((m.T_R[z, 0]/Tw)**2 for z in m.z) + \
        sum((m.lambda_g[z, m.R]*m.T_R[z, m.R] - m.U[z] * (Tw - m.T[z, m.R]))**2
            for z in m.z)* 1e-05

m.obj = Objective(rule=of_rule) # Dummy Objective

with open("ipopt.opt", "w") as f:
    f.write("linear_solver\tma57\n")
    f.write("output_file\tout.txt\n")
    f.write("print_info_string\tyes\n")
    f.write("max_iter\t500\n")

solver.solve(m,tee=True, symbolic_solver_labels=True)

m.del_component(m.obj)
m.obj = Objective(rule=1)

with open("ipopt.opt", "w") as f:
    f.write("linear_solver\tma57\n")
    f.write("output_file\tout.txt\n")
    f.write("print_info_string\tyes\n")
    f.write("max_iter\t2000\n")
    #f.write("tol\t1e-2\n")
    f.write("constr_viol_tol\t1e-2\n")
    f.write("dual_inf_tol\t1e+2\n")

#1970  1.0000000e+00 2.13e-01 6.45e+04  -2.5 2.47e+00  -0.9 7.14e-02 2.12e-02h  5 l

solver.solve(m,tee=True, symbolic_solver_labels=True)

pox_res = PoxResult(m)
pox_res.plot_r_profiles()

save_m5_state(m)
