from pyomo.environ import *
import numpy as np


def save_m5_state(m):
    Zs = m.z
    Rs = m.r
    Ss = m.SPECIES
    Rxs = m.REACTIONS

    F_in = np.zeros(len(Ss))
    C_in = np.zeros(len(Ss))
    Ft_in = np.zeros(1)
    X_in = np.zeros(len(Ss))
    u = np.zeros(1)
    Rep = np.zeros(1)
    f = np.zeros(1)
    F = np.zeros((len(Zs), len(Rs), len(Ss)))
    C = np.zeros((len(Zs), len(Rs), len(Ss)))
    X = np.zeros((len(Zs), len(Rs), len(Ss)))
    y = np.zeros((len(Zs), len(Rs)))
    Pt = np.zeros((len(Zs), len(Rs)))
    T = np.zeros((len(Zs), len(Rs)))
    Ft = np.zeros((len(Zs), len(Rs)))
    P = np.zeros((len(Zs), len(Rs), len(Ss)))
    Rate = np.zeros((len(Zs), len(Rs), len(Rxs)))
    r_comp = np.zeros((len(Zs), len(Rs), len(Ss)))
    DEN = np.zeros((len(Zs), len(Rs)))
    k1 = np.zeros((len(Zs), len(Rs)))
    k2 = np.zeros((len(Zs), len(Rs)))
    k3 = np.zeros((len(Zs), len(Rs)))
    Ke = np.zeros((len(Zs), len(Rs), len(Rxs)))
    Ka = np.zeros((len(Zs), len(Rs), len(Ss)))
    dCz = np.zeros((len(Zs), len(Rs), len(Ss)))
    dCr = np.zeros((len(Zs), len(Rs), len(Ss)))
    dy = np.zeros((len(Zs), len(Rs)))
    dTz = np.zeros((len(Zs), len(Rs)))
    dTr = np.zeros((len(Zs), len(Rs)))
    logKa = np.zeros((len(Zs), len(Rs), len(Ss)))
    C_Z = np.zeros((len(Zs), len(Rs), len(Ss)))
    dC_Zz = np.zeros((len(Zs), len(Rs), len(Ss)))
    C_R = np.zeros((len(Zs), len(Rs), len(Ss)))
    dC_Rr = np.zeros((len(Zs), len(Rs), len(Ss)))
    T_Z = np.zeros((len(Zs), len(Rs)))
    dT_Zz = np.zeros((len(Zs), len(Rs)))
    T_R = np.zeros((len(Zs), len(Rs)))
    dT_Rr = np.zeros((len(Zs), len(Rs)))
    lambda_i = np.zeros((len(Zs), len(Rs), len(Ss)))
    lambda_tr = np.zeros((len(Zs), len(Rs), len(Ss), len(Ss)))
    A_lambda_ij = np.zeros((len(Zs), len(Rs), len(Ss), len(Ss)))
    lg_den = np.zeros((len(Zs), len(Rs), len(Ss)))
    lambda_g = np.zeros((len(Zs), len(Rs)))
    mu_i = np.zeros((len(Zs), len(Rs), len(Ss)))
    mu_den = np.zeros((len(Zs), len(Rs), len(Ss)))
    mu = np.zeros((len(Zs), len(Rs)))
    CP_i = np.zeros((len(Zs), len(Rs), len(Ss)))
    CP_g = np.zeros((len(Zs), len(Rs)))
    Re = np.zeros(len(Zs))
    Pr = np.zeros(len(Zs))
    U = np.zeros(len(Zs))
    edC_ZCp = np.zeros((len(Zs), len(Rs)))
    edC_RCp = np.zeros((len(Zs), len(Rs)))
    omegaD = np.zeros((len(Zs), len(Rs), len(Ss), len(Ss)))
    Dij_ = np.zeros((len(Zs), len(Rs), len(Ss), len(Ss)))
    Dim_ = np.zeros((len(Zs), len(Rs), len(Ss)))
    Diez_ = np.zeros((len(Zs), len(Rs), len(Ss)))

    Ft_in[0] = value(m.Ft_in)
    u[0] = value(m.u)
    Rep[0] = value(m.Rep)
    f[0] = value(m.f)

    for rx in range(len(Ss)):
        i0 = Ss.at(rx+1)
        F_in[rx] = value(m.F_in[i0])
        C_in[rx] = value(m.C_in[i0])
        X_in[rx] = value(m.X_in[i0])

    for z in range(len(Zs)):
        for r in range(len(Rs)):
            for s in range(len(Ss)):
                (i0, i1, i2) = (Zs.at(z+1), Rs.at(r+1), Ss.at(s+1))
                F[z, r, s] = value(m.F[i0, i1, i2])
                C[z, r, s] = value(m.C[i0, i1, i2])
                X[z, r, s] = value(m.X[i0, i1, i2])
                P[z, r, s] = value(m.P[i0, i1, i2])
                r_comp[z, r, s] = value(m.r_comp[i0, i1, i2])
                Ka[z, r, s] = value(m.Ka[i0, i1, i2])
                dCz[z, r, s] = value(m.dCz[i0, i1, i2])
                dCr[z, r, s] = value(m.dCr[i0, i1, i2])
                logKa[z, r, s] = value(m.logKa[i0, i1, i2])
                C_Z[z, r, s] = value(m.C_Z[i0, i1, i2])
                C_R[z, r, s] = value(m.C_R[i0, i1, i2])
                dC_Rr[z, r, s] = value(m.dC_Rr[i0, i1, i2])
                lambda_i[z, r, s] = value(m.lambda_i[i0, i1, i2])
                lg_den[z, r, s] = value(m.lg_den[i0, i1, i2])
                mu_i[z, r, s] = value(m.mu_i[i0, i1, i2])
                mu_den[z, r, s] = value(m.mu_den[i0, i1, i2])
                CP_i[z, r, s] = value(m.CP_i[i0, i1, i2])
                Dim_[z, r, s] = value(m.Dim_[i0, i1, i2])
                Diez_[z, r, s] = value(m.Diez_[i0, i1, i2])

    for z in range(len(Zs)):
        for r in range(len(Rs)):
            (i0, i1) = (Zs.at(z+1), Rs.at(r+1))
            y[z, r] = value(m.y[i0, i1])
            Pt[z, r] = value(m.Pt[i0, i1])
            T[z, r] = value(m.T[i0, i1])
            Ft[z, r] = value(m.Ft[i0, i1])
            DEN[z, r] = value(m.DEN[i0, i1])
            k1[z, r] = value(m.k1[i0, i1])
            k2[z, r] = value(m.k2[i0, i1])
            k3[z, r] = value(m.k3[i0, i1])
            dy[z, r] = value(m.dy[i0, i1])
            dTz[z, r] = value(m.dTz[i0, i1])
            dTr[z, r] = value(m.dTr[i0, i1])
            T_Z[z, r] = value(m.T_Z[i0, i1])
            dT_Zz[z, r] = value(m.dT_Zz[i0, i1])
            T_R[z, r] = value(m.T_Z[i0, i1])
            dT_Rr[z, r] = value(m.dT_Rr[i0, i1])
            lambda_g[z, r] = value(m.lambda_g[i0, i1])
            mu[z, r] = value(m.mu[i0, i1])
            CP_g[z, r] = value(m.CP_g[i0, i1])
            edC_ZCp[z, r] = value(m.edC_ZCp[i0, i1])
            edC_RCp[z, r] = value(m.edC_RCp[i0, i1])

    for z in range(len(Zs)):
        for r in range(len(Rs)):
            for s in range(len(Rxs)):
                (i0, i1, i2) = (Zs.at(z+1), Rs.at(r+1), Rxs.at(s+1))
                Rate[z, r, s] = value(m.Rate[i0, i1, i2])
                Ke[z, r, s] = value(m.Ke[i0, i1, i2])


    for z in range(len(Zs)):
        for r in range(len(Rs)):
            for s0 in range(len(Ss)):
                for s1 in range(len(Ss)):
                    (i0, i1, i2, i3) = (Zs.at(z+1), Rs.at(r+1), Rxs.at(s0+1), Rxs.at(s1+1))
                    lambda_tr[z, r, s0, s1] = value(m.lambda_tr[i0, i1, i2, i3])
                    A_lambda_ij[z, r, s0, s1] = value(m.A_lambda_ij[i0, i1, i2, i3])
                    omegaD[z, r, s0, s1] = value(m.omegaD[i0, i1, i2, i3])
                    Dij_[z, r, s0, s1] = value(m.Dij_[i0, i1, i2, i3])

