from pyomo.environ import *
import matplotlib.pyplot as plt
import numpy as np
import sys

from m5_v8 import m

sys.path.insert(1, "../m5_v4/")
from sav_state_v4 import save_m5_state, load_m5_state

sys.path.insert(1, "../utils/")
from m5_plt import PoxResult

print("Make sure IPOPT is in the path.")

print("load state")
# load the profiles from m2
load_m5_state(m, "../m5_v5", m2=True)
print("state loaded")


#ipexe = "/Users/dthierry/Apps/ipopt_dir/bin/ipopt"
# change this for other solver.
solver = SolverFactory("ipopt")

# first set of options
with open("ipopt.opt", "w") as f:
    f.write("linear_solver\tma57\n")
    f.write("output_file\tout.txt\n")
    f.write("print_info_string\tyes\n")
    f.write("max_iter\t20\n")

s = solver.solve(m,tee=True, symbolic_solver_labels=True)

m.lambda_g.setlb(0e0)
#m.T.setlb(274)

# second set of options
with open("ipopt.opt", "w") as f:
    f.write("linear_solver\tma57\n")
    f.write("start_with_resto\tyes\n")
    f.write("output_file\tout.txt\n")
    f.write("print_info_string\tyes\n")
    #f.write("max_iter\t2000\n")
    f.write("max_iter\t3000\n")
    f.write("required_infeasibility_reduction\t1e-3\n")

s = solver.solve(m,tee=True, symbolic_solver_labels=True)


pox_res = PoxResult(m)
pox_res.plot_r_profiles()

