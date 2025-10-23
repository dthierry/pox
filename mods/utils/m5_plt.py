
from pyomo.environ import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import datetime
import os

def generate_color(value, colormap_name='viridis'):
    """
    Generates an RGB color (in the range 0 to 1) from a scalar value.

    Args:
        value (float): The scalar value to map to a color, must be between 0 and 1.
        colormap_name (str): The name of the Matplotlib colormap to use.

    Returns:
        tuple: An RGBA tuple (Red, Green, Blue, Alpha) where each component
               is in the range [0.0, 1.0].
    """
    if not 0 <= value <= 1:
        raise ValueError("Value must be between 0 and 1.")

    # Get the colormap
    cmap = cm.get_cmap(colormap_name)

    # Apply the colormap to the value
    # The result is an RGBA tuple with values between 0.0 and 1.0
    rgba_color = cmap(value)

    return rgba_color

def plot_colorbar(colormap_name='viridis',
                  title='Color Bar for Value Range [0.0, 1.0]',
                  figure_name='fig_name.png'):
    """
    Generates and displays a Matplotlib color bar for a given colormap.

    Args:
        colormap_name (str): The name of the Matplotlib colormap to visualize.
        title (str): The title for the plot.
    """
    # 1. Get the colormap object
    cmap = cm.get_cmap(colormap_name)

    # 2. Create a 'Normalize' object to map data values (0 to 1) to the color range
    # This is essential for the color bar to display the correct scale.
    norm = plt.Normalize(vmin=0.0, vmax=1.0)

    # 3. Create a figure and an axis
    fig, ax = plt.subplots(figsize=(6, 1))
    ax.set_title(title)

    # 4. Use the color bar function
    # 'mappable' is set to the colormap and normalization
    # 'orientation' is set to horizontal for a compact view
    # 'label' describes the axis of the color bar
    cb = plt.colorbar(
        cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=ax,
        orientation='horizontal',
        label=f'Value (Mapped to Color via "{colormap_name}")'
    )

    fig.tight_layout()
    # 5. Display the plot
    #plt.savefig(self.folder + "/legend_axis.png")
    plt.savefig(figure_name)


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
        # Pt
        self.Pt = np.zeros((len(Zs), len(Rs)))
        # C
        self.C = np.zeros((len(Zs), len(Rs), len(Ss)))

        # read model
        for z in range(len(Zs)):
            for r in range(len(Rs)):
                two_idx_0 = (Zs.at(z+1), Rs.at(r+1))
                self.T[z, r] = value(m.T[two_idx_0])
                self.y[z, r] = value(m.y[two_idx_0])
                self.Pt[z, r] = value(m.Pt[two_idx_0])
                for s in range(len(Ss)):
                    three_idx = (Zs.at(z+1), Rs.at(r+1), Ss.at(s+1))
                    self.X[z, r, s] = value(m.X[three_idx])
                    #three_idx = (Zs.at(z+1), Rs.at(r+1), Ss.at(s+1))
                    self.C[z, r, s] = value(m.C[three_idx])
                for s in range(len(Rxs)):
                    three_idx_r = (Zs.at(z+1), Rs.at(r+1), Rxs.at(s+1))
                    self.Rate[z, r, s] = value(m.Rate[three_idx_r])


        #for z in range(len(Zs)):
        #    for r in range(len(Rs)):
        #        for s in range(len(Rxs)):
        #            three_idx_r = (Zs.at(z+1), Rs.at(r+1), Rxs.at(s+1))
        #            self.Rate[z, r, s] = value(m.Rate[three_idx_r])

        # for z in range(len(Zs)):
        #     for r in range(len(Rs)):
        #         two_idx_0 = (Zs.at(z+1), Rs.at(r+1))


        # for z in range(len(Zs)):
        #     for r in range(len(Rs)):
        #         for s in range(len(Ss)):



    def plot_r_profiles(self):
        """Add a number to the current value."""
        z_coords = np.array(self.m.z.data())
        r_coords = np.array(self.m.r.data())/self.m.R
        # algebraic variable range
        r_coords_alg = r_coords[1:-1]

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
            a.bar(z, 1, #label=f"z/L={zlab}",
                  color=generate_color(zlab, colormap_name="winter"),
                  align="edge")
        #a.legend()
        f.savefig(self.folder + "/legend.png")


        f, a = plt.subplots(dpi=200)
        for z in range(len(z_coords)):
            zlab = z_coords[z]/m.L
            a.plot(r_coords, self.T[z, :], label=f"z/L={zlab}",
                   color=generate_color(zlab, colormap_name="winter")
                   )
        a.set_title("Temperature Profile")
        a.set_xlabel("r/R")
        a.set_ylabel("(K)")
        #a.legend()
        f.savefig(self.folder + "/T.png")


        f, a = plt.subplots(dpi=200)
        for r in range(len(r_coords)):
            rlab = r_coords[r]
            a.plot(z_coords, self.T[:, r], label=f"r/R={rlab}",
                   color=generate_color(rlab)
                   )
        a.set_title("Temperature Profile")
        a.set_xlabel("x")
        a.set_ylabel("(K)")
        #a.legend()
        f.savefig(self.folder + "/T-axial.png")

        f, a = plt.subplots(nrows=sp_rows, ncols=sp_cols, dpi=200)
        for s in range(len(m.SPECIES)):
            sp_lab = m.SPECIES.at(s+1)
            for z in range(len(z_coords)):
                if z == 0 or z == len(z_coords):
                    continue
                zlab = z_coords[z]/m.L
                a.flat[s].plot(r_coords_alg, self.X[z, 1:-1, s],
                               label=f"z/L={zlab}",
                               color=generate_color(zlab, colormap_name="winter")
                               )
            a.flat[s].set_title(f"{sp_lab} molf.")
            a.flat[s].set_xlabel("r/R")

        f.tight_layout()
        #a.flat[0].legend()
        f.savefig(self.folder + "/X.png")


        f, a = plt.subplots(nrows=sp_rows, ncols=sp_cols, dpi=200)
        for s in range(len(m.SPECIES)):
            sp_lab = m.SPECIES.at(s+1) # species
            for z in range(len(z_coords)):
                zlab = z_coords[z]/m.L
                a.flat[s].plot(r_coords, self.C[z, :, s], label=f"z/L={zlab}",
                               color=generate_color(zlab, colormap_name="winter")
                               )
            a.flat[s].set_title(f"{sp_lab} Conc.")
            a.flat[s].set_xlabel("r/R")
        f.tight_layout()
        f.savefig(self.folder + "/C_radial.png")


        f, a = plt.subplots(nrows=sp_rows, ncols=sp_cols, dpi=200)
        for s in range(len(m.SPECIES)):
            sp_lab = m.SPECIES.at(s+1) # species
            for r in range(len(r_coords)):
                rlab = r_coords[r]
                a.flat[s].plot(z_coords, self.C[:, r, s], label=f"r/R={rlab}",
                               color=generate_color(rlab)
                               )
            a.flat[s].set_title(f"{sp_lab} Conc.")
            a.flat[s].set_xlabel("z")

        f.tight_layout()
        f.savefig(self.folder + "/C_axial.png")

        # f, a = plt.subplots(dpi=200)
        # for z in range(len(z_coords)):
        #     zlab = z_coords[z]/m.L
        #     a.plot(r_coords, self.y[z, :], label=f"z/L={zlab}", color=cm[z])
        # a.set_title("Dimensionless Pressure Profile")
        # a.set_xlabel("Radial loc")
        # #a.legend()
        # f.savefig(self.folder + "/y.png")


        # f, a = plt.subplots(dpi=200)
        # for r in range(len(r_coords)):
        #     rlab = r_coords[r]
        #     a.plot(z_coords, self.y[:, r], label=f"r/R={rlab}")
        # a.set_title("Dimensionless Pressure Profile")
        # a.set_xlabel("Axial loc")
        # #a.legend()
        # f.savefig(self.folder + "/y-axial.png")


        f, a = plt.subplots(dpi=200)
        for z in range(len(z_coords)):
            zlab = z_coords[z]/m.L
            a.plot(r_coords, self.Pt[z, :], label=f"z/L={zlab}",
                   color=generate_color(zlab, colormap_name="winter")
                   )
        a.set_title("Dimensionless Pressure Profile")
        a.set_xlabel("r/R")
        #a.legend()
        f.savefig(self.folder + "/Pt.png")


        f, a = plt.subplots(dpi=200)
        for r in range(len(r_coords)):
            rlab = r_coords[r]
            a.plot(z_coords, self.Pt[:, r], label=f"r/R={rlab}",
                   color=generate_color(rlab)
                   )
        a.set_title("Dimensionless Pressure Profile")
        a.set_xlabel("r/R")
        #a.legend()
        f.savefig(self.folder + "/Pt-axial.png")

        ratecol = 1
        raterow = 3
        f, a = plt.subplots(nrows=raterow, ncols=ratecol, dpi=200)
        for rate in range(len(m.REACTIONS)):
            rx = m.REACTIONS.at(rate+1)
            for z in range(len(z_coords)):
                if z == 0 or z == len(z_coords):
                    continue
                zlab = z_coords[z]/m.L
                a.flat[rate].plot(r_coords_alg, self.Rate[z, 1:-1, rate],
                                  color=generate_color(zlab, colormap_name="winter")
                                  )
                a.flat[rate].set_title(f"Rate {rx}")
                a.flat[rate].set_xlabel("r/R")
        f.savefig(self.folder + "/rate.png")

        fname = self.folder + "/r_label.png"
        plot_colorbar(colormap_name="viridis", figure_name=fname)

        fname = self.folder + "/x_label.png"
        plot_colorbar(colormap_name="winter", figure_name=fname)


