
from pyomo.environ import *
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os

class PoxResult:
    def __init__(self, m):
        """Initialize the calculator with no stored value."""
        ts = datetime.datetime.now().timestamp()
        self.folder = f"./{ts}"
        self.m = m
        os.makedirs(self.folder)
        print("Folder:")
        print(self.folder)

        Zs = m.z
        Rs = m.r
        Ss = m.SPECIES
        Rxs = m.REACTIONS

        # X
        self.X = np.zeros((len(Zs), len(Rs), len(Ss)))
        # T
        self.T = np.zeros((len(Zs), len(Rs)))
        # Rate
        self.Rate = np.zeros((len(Zs), len(Rs), len(Rxs)))
        # y
        self.y = np.zeros((len(Zs), len(Rs)))
        # C
        self.C = np.zeros((len(Zs), len(Rs), len(Ss)))

        # read model
        for z in range(len(Zs)):
            for r in range(len(Rs)):
                for s in range(len(Ss)):
                    three_idx = (Zs.at(z+1), Rs.at(r+1), Ss.at(s+1))
                    self.X[z, r, s] = value(m.X[three_idx])

        for z in range(len(Zs)):
            for r in range(len(Rs)):
                two_idx_0 = (Zs.at(z+1), Rs.at(r+1))
                self.T[z, r] = value(m.T[two_idx_0])

        for z in range(len(Zs)):
            for r in range(len(Rs)):
                for s in range(len(Rxs)):
                    three_idx_r = (Zs.at(z+1), Rs.at(r+1), Rxs.at(s+1))
                    self.Rate[z, r, s] = value(m.Rate[three_idx_r])

        for z in range(len(Zs)):
            for r in range(len(Rs)):
                two_idx_0 = (Zs.at(z+1), Rs.at(r+1))
                self.y[z, r] = value(m.y[two_idx_0])


        for z in range(len(Zs)):
            for r in range(len(Rs)):
                for s in range(len(Ss)):
                    three_idx = (Zs.at(z+1), Rs.at(r+1), Ss.at(s+1))
                    self.C[z, r, s] = value(m.C[three_idx])



    def plot_r_profiles(self):
        """Add a number to the current value."""
        z_coords = np.array(self.m.z.data())
        r_coords = np.array(self.m.r.data())/self.m.R

        m = self.m

        # number of rows
        sp_rows = 2
        #sp_cols = len(m.SPECIES)//sp_rows + 1
        sp_cols = len(m.SPECIES)//sp_rows

        clinsp = np.linspace(0, 1, len(z_coords))
        cm = plt.cm.binary(clinsp)

        f, a = plt.subplots(dpi=200)
        for z in range(len(z_coords)):
            zlab = z_coords[z]/m.L
            a.plot([1], [1], label=f"z/L={zlab}", color=cm[z])
        a.legend()
        f.savefig(self.folder + "/legend.png")


        f, a = plt.subplots(dpi=200)
        for z in range(len(z_coords)):
            zlab = z_coords[z]/m.L
            a.plot(r_coords, self.T[z, :], label=f"z/L={zlab}", color=cm[z])
        a.set_title("Temperature Profile")
        a.set_xlabel("r/R")
        a.set_ylabel("(K)")
        #a.legend()
        f.savefig(self.folder + "/T.png")

        f, a = plt.subplots(nrows=sp_rows, ncols=sp_cols, dpi=200)
        for s in range(len(m.SPECIES)):
            sp_lab = m.SPECIES.at(s+1)
            for z in range(len(z_coords)):
                zlab = z_coords[z]/m.L
                a.flat[s].plot(r_coords, self.X[z, :, s], label=f"z/L={zlab}", color=cm[z])
            a.flat[s].set_title(f"{sp_lab} molf.")
            a.flat[s].set_xlabel("r/R")

        #a.flat[0].legend()
        f.savefig(self.folder + "/X.png")


        f, a = plt.subplots(nrows=sp_rows, ncols=sp_cols, dpi=200)
        for s in range(len(m.SPECIES)):
            sp_lab = m.SPECIES.at(s+1)
            for z in range(len(z_coords)):
                zlab = z_coords[z]/m.L
                a.flat[s].plot(r_coords, self.C[z, :, s], label=f"z/L={zlab}", color=cm[z])
            a.flat[s].set_title(f"{sp_lab} Conc.")
            a.flat[s].set_xlabel("r/R")
        f.savefig(self.folder + "/C.png")

        f, a = plt.subplots(dpi=200)
        for z in range(len(z_coords)):
            zlab = z_coords[z]/m.L
            a.plot(r_coords, self.y[z, :], label=f"z/L={zlab}", color=cm[z])
        a.set_title("Dimensionless Pressure Profile")
        a.set_xlabel("Radial loc")
        #a.legend()
        f.savefig(self.folder + "/y.png")

        ratecol = 1
        raterow = 3
        f, a = plt.subplots(nrows=raterow, ncols=ratecol, dpi=200)
        for rate in range(len(m.REACTIONS)):
            rx = m.REACTIONS.at(rate+1)
            for z in range(len(z_coords)):
                zlab = z_coords[z]/m.L
                a.flat[rate].plot(r_coords, self.Rate[z, :, rate], color=cm[z])
                a.flat[rate].set_title(f"{sp_lab} Conc.")
                a.flat[rate].set_xlabel("r/R")
        f.savefig(self.folder + "/rate.png")



