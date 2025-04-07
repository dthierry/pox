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
                if not(m.dC_Rr[i0, i1, i2].stale):
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
            if not(m.dy[i0, i1].stale):
                dy[z, r] = value(m.dy[i0, i1])
            dTz[z, r] = value(m.dTz[i0, i1])
            dTr[z, r] = value(m.dTr[i0, i1])
            T_Z[z, r] = value(m.T_Z[i0, i1])
            if not(m.dT_Zz[i0, i1].stale):
                dT_Zz[z, r] = value(m.dT_Zz[i0, i1])
            T_R[z, r] = value(m.T_Z[i0, i1])
            if not(m.dT_Rr[i0, i1].stale):
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
                    (i0, i1, i2, i3) = (Zs.at(z+1), Rs.at(r+1), Ss.at(s0+1), Ss.at(s1+1))
                    lambda_tr[z, r, s0, s1] = value(m.lambda_tr[i0, i1, i2, i3])
                    A_lambda_ij[z, r, s0, s1] = value(m.A_lambda_ij[i0, i1, i2, i3])
                    omegaD[z, r, s0, s1] = value(m.omegaD[i0, i1, i2, i3])
                    Dij_[z, r, s0, s1] = value(m.Dij_[i0, i1, i2, i3])
    # save arrays
    np.save("F_in.npy", F_in)
    np.save("C_in.npy", C_in)
    np.save("Ft_in.npy", Ft_in)
    np.save("X_in.npy", X_in)
    np.save("u.npy", u)
    np.save("Rep.npy", Rep)
    np.save("f.npy", f)
    np.save("F.npy", F)
    np.save("C.npy", C)
    np.save("X.npy", X)
    np.save("y.npy", y)
    np.save("Pt.npy", Pt)
    np.save("T.npy", T)
    np.save("Ft.npy", Ft)
    np.save("P.npy", P)
    np.save("Rate.npy", Rate)
    np.save("r_comp.npy", r_comp)
    np.save("DEN.npy", DEN)
    np.save("k1.npy", k1)
    np.save("k2.npy", k2)
    np.save("k3.npy", k3)
    np.save("Ke.npy", Ke)
    np.save("Ka.npy", Ka)
    np.save("dCz.npy", dCz)
    np.save("dCr.npy", dCr)
    np.save("dy.npy", dy)
    np.save("dTz.npy", dTz)
    np.save("dTr.npy", dTr)
    np.save("logKa.npy", logKa)
    np.save("C_Z.npy", C_Z)
    np.save("dC_Zz.npy", dC_Zz)
    np.save("C_R.npy", C_R)
    np.save("dC_Rr.npy", dC_Rr)
    np.save("T_Z.npy", T_Z)
    np.save("dT_Zz.npy", dT_Zz)
    np.save("T_R.npy", T_R)
    np.save("dT_Rr.npy", dT_Rr)
    np.save("lambda_i.npy", lambda_i)
    np.save("lambda_tr.npy", lambda_tr)
    np.save("A_lambda_ij.npy", A_lambda_ij)
    np.save("lg_den.npy", lg_den)
    np.save("lambda_g.npy", lambda_g)
    np.save("mu_i.npy", mu_i)
    np.save("mu_den.npy", mu_den)
    np.save("mu.npy", mu)
    np.save("CP_i.npy", CP_i)
    np.save("CP_g.npy", CP_g)
    np.save("Re.npy", Re)
    np.save("Pr.npy", Pr)
    np.save("U.npy", U)
    np.save("edC_ZCp.npy", edC_ZCp)
    np.save("edC_RCp.npy", edC_RCp)
    np.save("omegaD.npy", omegaD)
    np.save("Dij_.npy", Dij_)
    np.save("Dim_.npy", Dim_)
    np.save("Diez_.npy", Diez_)


def load_m5_state(m):
    Zs = m.z
    Rs = m.r
    Ss = m.SPECIES
    Rxs = m.REACTIONS
    # read the npy files.
    F_in = np.load("F_in.npy")
    C_in = np.load("C_in.npy")
    Ft_in = np.load("Ft_in.npy")
    X_in = np.load("X_in.npy")
    u = np.load("u.npy")
    Rep = np.load("Rep.npy")
    f = np.load("f.npy")
    F = np.load("F.npy")
    C = np.load("C.npy")
    X = np.load("X.npy")
    y = np.load("y.npy")
    Pt = np.load("Pt.npy")
    T = np.load("T.npy")
    Ft = np.load("Ft.npy")
    P = np.load("P.npy")
    Rate = np.load("Rate.npy")
    r_comp = np.load("r_comp.npy")
    DEN = np.load("DEN.npy")
    k1 = np.load("k1.npy")
    k2 = np.load("k2.npy")
    k3 = np.load("k3.npy")
    Ke = np.load("Ke.npy")
    Ka = np.load("Ka.npy")
    dCz = np.load("dCz.npy")
    dCr = np.load("dCr.npy")
    dy = np.load("dy.npy")
    dTz = np.load("dTz.npy")
    dTr = np.load("dTr.npy")
    logKa = np.load("logKa.npy")
    C_Z = np.load("C_Z.npy")
    dC_Zz = np.load("dC_Zz.npy")
    C_R = np.load("C_R.npy")
    dC_Rr = np.load("dC_Rr.npy")
    T_Z = np.load("T_Z.npy")
    dT_Zz = np.load("dT_Zz.npy")
    T_R = np.load("T_R.npy")
    dT_Rr = np.load("dT_Rr.npy")
    lambda_i = np.load("lambda_i.npy")
    lambda_tr = np.load("lambda_tr.npy")
    A_lambda_ij = np.load("A_lambda_ij.npy")
    lg_den = np.load("lg_den.npy")
    lambda_g = np.load("lambda_g.npy")
    mu_i = np.load("mu_i.npy")
    mu_den = np.load("mu_den.npy")
    mu = np.load("mu.npy")
    CP_i = np.load("CP_i.npy")
    CP_g = np.load("CP_g.npy")
    Re = np.load("Re.npy")
    Pr = np.load("Pr.npy")
    U = np.load("U.npy")
    edC_ZCp = np.load("edC_ZCp.npy")
    edC_RCp = np.load("edC_RCp.npy")
    omegaD = np.load("omegaD.npy")
    Dij_ = np.load("Dij_.npy")
    Dim_ = np.load("Dim_.npy")
    Diez_ = np.load("Diez_.npy")

    # set values
    # Ft_in[0] = value(m.Ft_in)
    # u[0] = value(m.u)
    # Rep[0] = value(m.Rep)
    # f[0] = value(m.f)

    for rx in range(len(Ss)):
        i0 = Ss.at(rx+1)
        m.F_in[i0].set_value(F_in[rx])
        m.C_in[i0].set_value(C_in[rx])
        m.X_in[i0].set_value(X_in[rx])

    for z in range(len(Zs)):
        for r in range(len(Rs)):
            for s in range(len(Ss)):
                (i0, i1, i2) = (Zs.at(z+1), Rs.at(r+1), Ss.at(s+1))
                m.F[i0, i1, i2].set_value(F[z, r, s])
                m.C[i0, i1, i2].set_value(C[z, r, s])
                m.X[i0, i1, i2].set_value(X[z, r, s])
                m.P[i0, i1, i2].set_value(P[z, r, s])
                m.r_comp[i0, i1, i2].set_value(r_comp[z, r, s])
                m.Ka[i0, i1, i2].set_value(Ka[z, r, s])
                m.dCz[i0, i1, i2].set_value(dCz[z, r, s])
                m.dCr[i0, i1, i2].set_value(dCr[z, r, s])
                m.logKa[i0, i1, i2].set_value(logKa[z, r, s])
                m.C_Z[i0, i1, i2].set_value(C_Z[z, r, s])
                m.C_R[i0, i1, i2].set_value(C_R[z, r, s])
                if not(m.dC_Rr[i0, i1, i2].stale):
                    m.dC_Rr[i0, i1, i2].set_value(dC_Rr[z, r, s])
                m.lambda_i[i0, i1, i2].set_value(lambda_i[z, r, s])
                m.lg_den[i0, i1, i2].set_value(lg_den[z, r, s])
                m.mu_i[i0, i1, i2].set_value(mu_i[z, r, s])
                m.mu_den[i0, i1, i2].set_value(mu_den[z, r, s])
                m.CP_i[i0, i1, i2].set_value(CP_i[z, r, s])
                m.Dim_[i0, i1, i2].set_value(Dim_[z, r, s])
                m.Diez_[i0, i1, i2].set_value(Diez_[z, r, s])

    for z in range(len(Zs)):
        for r in range(len(Rs)):
            (i0, i1) = (Zs.at(z+1), Rs.at(r+1))
            m.y[i0, i1].set_value(y[z, r])
            m.Pt[i0, i1].set_value(Pt[z, r])
            m.T[i0, i1].set_value(T[z, r])
            m.Ft[i0, i1].set_value(Ft[z, r])
            m.DEN[i0, i1].set_value(DEN[z, r])
            m.k1[i0, i1].set_value(k1[z, r])
            m.k2[i0, i1].set_value(k2[z, r])
            m.k3[i0, i1].set_value(k3[z, r])
            if not(m.dy[i0, i1].stale):
                m.dy[i0, i1].set_value(dy[z, r])
            m.dTz[i0, i1].set_value(dTz[z, r])
            m.dTr[i0, i1].set_value(dTr[z, r])
            m.T_Z[i0, i1].set_value(T_Z[z, r])
            if not(m.dT_Zz[i0, i1].stale):
                m.dT_Zz[i0, i1].set_value(dT_Zz[z, r])
            m.T_Z[i0, i1].set_value(T_R[z, r])
            if not(m.dT_Rr[i0, i1].stale):
                m.dT_Rr[i0, i1].set_value(dT_Rr[z, r])
            m.lambda_g[i0, i1].set_value(lambda_g[z, r])
            m.mu[i0, i1].set_value(mu[z, r])
            m.CP_g[i0, i1].set_value(CP_g[z, r])
            m.edC_ZCp[i0, i1].set_value(edC_ZCp[z, r])
            m.edC_RCp[i0, i1].set_value(edC_RCp[z, r])

    for z in range(len(Zs)):
        for r in range(len(Rs)):
            for s in range(len(Rxs)):
                (i0, i1, i2) = (Zs.at(z+1), Rs.at(r+1), Rxs.at(s+1))
                m.Rate[i0, i1, i2].set_value(Rate[z, r, s])
                m.Ke[i0, i1, i2].set_value(Ke[z, r, s])


    for z in range(len(Zs)):
        for r in range(len(Rs)):
            for s0 in range(len(Ss)):
                for s1 in range(len(Ss)):
                    (i0, i1, i2, i3) = (Zs.at(z+1), Rs.at(r+1), Ss.at(s0+1), Ss.at(s1+1))
                    lambda_tr[z, r, s0, s1] = m.lambda_tr[i0, i1, i2,
                                                          i3].set_value()
                    A_lambda_ij[z, r, s0, s1] = m.A_lambda_ij[i0, i1, i2,
                                                              i3].set_value()
                    m.omegaD[i0, i1, i2, i3].set_value(omegaD[z, r, s0, s1])
                    m.Dij_[i0, i1, i2, i3].set_value(Dij_[z, r, s0, s1])

