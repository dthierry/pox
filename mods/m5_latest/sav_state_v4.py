from pyomo.environ import *
import numpy as np

def adjust_lb(var, val):
    if var.lb != None:
        return var.lb if val < var.lb else val
    else:
        return val

def save_m5_state(m, m2=False):
    Zs = m.z
    Rs = [1] if m2 else m.r
    Ss = m.SPECIES
    Rxs = m.REACTIONS

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

    # for rx in range(len(Ss)):
    #     i0 = Ss.at(rx+1)
    #     C_in[rx] = value(m.C_in[i0])
    #     X_in[rx] = value(m.X_in[i0])

    for z in range(len(Zs)):
        for r in range(len(Rs)):
            for s in range(len(Ss)):
                if m2:
                    three_idx = (Zs.at(z+1), Ss.at(s+1))
                else:
                    three_idx = (Zs.at(z+1), Rs.at(r+1), Ss.at(s+1))
                F[z, r, s] = value(m.F[three_idx])
                C[z, r, s] = value(m.C[three_idx])
                X[z, r, s] = value(m.X[three_idx])
                P[z, r, s] = value(m.P[three_idx])
                r_comp[z, r, s] = value(m.r_comp[three_idx])
                Ka[z, r, s] = value(m.Ka[three_idx])
                dCz[z, r, s] = value(m.dCz[three_idx])
                if not m2:
                    dCr[z, r, s] = value(m.dCr[three_idx])
                logKa[z, r, s] = value(m.logKa[three_idx])
                C_Z[z, r, s] = value(m.C_Z[three_idx])
                if not m2:
                    C_R[z, r, s] = value(m.C_R[three_idx])
                    if not(m.dC_Rr[three_idx].stale):
                        dC_Rr[z, r, s] = value(m.dC_Rr[three_idx])
                lambda_i[z, r, s] = value(m.lambda_i[three_idx])
                lg_den[z, r, s] = value(m.lg_den[three_idx])
                mu_i[z, r, s] = value(m.mu_i[three_idx])
                mu_den[z, r, s] = value(m.mu_den[three_idx])
                CP_i[z, r, s] = value(m.CP_i[three_idx])
                Dim_[z, r, s] = value(m.Dim_[three_idx])
                Diez_[z, r, s] = value(m.Diez_[three_idx])

    for z in range(len(Zs)):
        for r in range(len(Rs)):
            if m2:
                two_idx_0 = (Zs.at(z+1),)
            else:
                two_idx_0 = (Zs.at(z+1), Rs.at(r+1))
            y[z, r] = value(m.y[two_idx_0])
            Pt[z, r] = value(m.Pt[two_idx_0])
            T[z, r] = value(m.T[two_idx_0])
            Ft[z, r] = value(m.Ft[two_idx_0])
            DEN[z, r] = value(m.DEN[two_idx_0])
            k1[z, r] = value(m.k1[two_idx_0])
            k2[z, r] = value(m.k2[two_idx_0])
            k3[z, r] = value(m.k3[two_idx_0])
            if not(m.dy[two_idx_0].stale):
                dy[z, r] = value(m.dy[two_idx_0])
            dTz[z, r] = value(m.dTz[two_idx_0])
            if not m2:
                dTr[z, r] = value(m.dTr[two_idx_0])
            T_Z[z, r] = value(m.T_Z[two_idx_0])
            if not(m.dT_Zz[two_idx_0].stale):
                dT_Zz[z, r] = value(m.dT_Zz[two_idx_0])
            if not m2:
                T_R[z, r] = value(m.T_Z[two_idx_0])
                if not(m.dT_Rr[two_idx_0].stale):
                    dT_Rr[z, r] = value(m.dT_Rr[two_idx_0])
            lambda_g[z, r] = value(m.lambda_g[two_idx_0])
            mu[z, r] = value(m.mu[two_idx_0])
            CP_g[z, r] = value(m.CP_g[two_idx_0])
            edC_ZCp[z, r] = value(m.edC_ZCp[two_idx_0])
            if not m2:
                edC_RCp[z, r] = value(m.edC_RCp[two_idx_0])

    for z in range(len(Zs)):
        for r in range(len(Rs)):
            for s in range(len(Rxs)):
                if m2:
                    three_idx_r = (Zs.at(z+1), Rxs.at(s+1))
                else:
                    three_idx_r = (Zs.at(z+1), Rs.at(r+1), Rxs.at(s+1))
                Rate[z, r, s] = value(m.Rate[three_idx_r])
                Ke[z, r, s] = value(m.Ke[three_idx_r])


    for z in range(len(Zs)):
        for r in range(len(Rs)):
            for s0 in range(len(Ss)):
                for s1 in range(len(Ss)):
                    if m2:
                        four_idx_1 = (Zs.at(z+1), Ss.at(s0+1), Ss.at(s1+1))
                    else:
                        four_idx_1 = (Zs.at(z+1), Rs.at(r+1), Ss.at(s0+1), Ss.at(s1+1))
                    lambda_tr[z, r, s0, s1] = value(m.lambda_tr[four_idx_1])
                    A_lambda_ij[z, r, s0, s1] = value(m.A_lambda_ij[four_idx_1])
                    omegaD[z, r, s0, s1] = value(m.omegaD[four_idx_1])
                    Dij_[z, r, s0, s1] = value(m.Dij_[four_idx_1])

    for z in range(len(Zs)):
        idx = Zs.at(z+1)
        Re[z] = value(m.Re[idx])
        Pr[z] = value(m.Pr[idx])
        U[z] = value(m.U[idx])

    # save arrays
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


def load_m5_state(m, path, m2=False):
    Zs = m.z
    Rs = m.r
    Ss = m.SPECIES
    Rxs = m.REACTIONS
    # read the npy files.
    print(f"Loading path = {path}\n")
    u = np.load(path + "/u.npy")
    Rep = np.load(path + "/Rep.npy")
    f = np.load(path + "/f.npy")
    F = np.load(path + "/F.npy")
    C = np.load(path + "/C.npy")
    X = np.load(path + "/X.npy")
    y = np.load(path + "/y.npy")
    Pt = np.load(path + "/Pt.npy")
    T = np.load(path + "/T.npy")
    Ft = np.load(path + "/Ft.npy")
    P = np.load(path + "/P.npy")
    Rate = np.load(path + "/Rate.npy")
    r_comp = np.load(path + "/r_comp.npy")
    DEN = np.load(path + "/DEN.npy")
    k1 = np.load(path + "/k1.npy")
    k2 = np.load(path + "/k2.npy")
    k3 = np.load(path + "/k3.npy")
    Ke = np.load(path + "/Ke.npy")
    Ka = np.load(path + "/Ka.npy")
    dCz = np.load(path + "/dCz.npy")
    dCr = np.load(path + "/dCr.npy")
    dy = np.load(path + "/dy.npy")
    dTz = np.load(path + "/dTz.npy")
    dTr = np.load(path + "/dTr.npy")
    logKa = np.load(path + "/logKa.npy")
    C_Z = np.load(path + "/C_Z.npy")
    dC_Zz = np.load(path + "/dC_Zz.npy")
    C_R = np.load(path + "/C_R.npy")
    dC_Rr = np.load(path + "/dC_Rr.npy")
    T_Z = np.load(path + "/T_Z.npy")
    dT_Zz = np.load(path + "/dT_Zz.npy")
    T_R = np.load(path + "/T_R.npy")
    dT_Rr = np.load(path + "/dT_Rr.npy")
    lambda_i = np.load(path + "/lambda_i.npy")
    lambda_tr = np.load(path + "/lambda_tr.npy")
    A_lambda_ij = np.load(path + "/A_lambda_ij.npy")
    lg_den = np.load(path + "/lg_den.npy")
    lambda_g = np.load(path + "/lambda_g.npy")
    mu_i = np.load(path + "/mu_i.npy")
    mu_den = np.load(path + "/mu_den.npy")
    mu = np.load(path + "/mu.npy")
    CP_i = np.load(path + "/CP_i.npy")
    CP_g = np.load(path + "/CP_g.npy")
    Re = np.load(path + "/Re.npy")
    Pr = np.load(path + "/Pr.npy")
    U = np.load(path + "/U.npy")
    edC_ZCp = np.load(path + "/edC_ZCp.npy")
    edC_RCp = np.load(path + "/edC_RCp.npy")
    omegaD = np.load(path + "/omegaD.npy")
    Dij_ = np.load(path + "/Dij_.npy")
    Dim_ = np.load(path + "/Dim_.npy")
    Diez_ = np.load(path + "/Diez_.npy")

    # set values
    # Ft_in[0] = value(m.Ft_in)
    # u[0] = value(m.u)
    # Rep[0] = value(m.Rep)
    # f[0] = value(m.f)

    # for rx in range(len(Ss)):
    #     i0 = Ss.at(rx+1)
    #     m.C_in[i0].set_value(adjust_lb(m.C_in[i0], C_in[rx]))
    #     m.X_in[i0].set_value(adjust_lb(m.X_in[i0], X_in[rx]))

    for z in range(len(Zs)):
        for r in range(len(Rs)):
            r_ = 0 if m2 else r
            for s in range(len(Ss)):
                (i0, i1, i2) = (Zs.at(z+1), Rs.at(r+1), Ss.at(s+1))
                m.F[i0, i1, i2].set_value(adjust_lb(m.F[i0, i1, i2], F[z, r_, s]))
                m.C[i0, i1, i2].set_value(adjust_lb(m.C[i0, i1, i2], C[z, r_, s]))
                m.X[i0, i1, i2].set_value(adjust_lb(m.X[i0, i1, i2], X[z, r_, s]))
                m.P[i0, i1, i2].set_value(adjust_lb(m.P[i0, i1, i2], P[z, r_, s]))
                m.r_comp[i0, i1, i2].set_value(adjust_lb(m.r_comp[i0, i1, i2], r_comp[z, r_, s]))
                m.Ka[i0, i1, i2].set_value(adjust_lb(m.Ka[i0, i1, i2], Ka[z, r_, s]))
                m.dCz[i0, i1, i2].set_value(adjust_lb(m.dCz[i0, i1, i2], dCz[z, r_, s]))
                m.dCr[i0, i1, i2].set_value(adjust_lb(m.dCr[i0, i1, i2], dCr[z,
                                                                             r_,
                                                                             s]))
                m.logKa[i0, i1, i2].set_value(adjust_lb(m.logKa[i0, i1, i2],
                                                        logKa[z, r_, s]))
                m.C_Z[i0, i1, i2].set_value(adjust_lb(m.C_Z[i0, i1, i2], C_Z[z,
                                                                             r_,
                                                                             s]))
                m.C_R[i0, i1, i2].set_value(adjust_lb(m.C_R[i0, i1, i2], C_R[z,
                                                                             r_,
                                                                             s]))
                if not(m.dC_Rr[i0, i1, i2].stale):
                    m.dC_Rr[i0, i1, i2].set_value(adjust_lb(m.dC_Rr[i0, i1, i2],
                                                            dC_Rr[z, r_, s]))
                m.lambda_i[i0, i1, i2].set_value(adjust_lb(m.lambda_i[i0, i1,
                                                                      i2],
                                                           lambda_i[z, r_, s]))
                m.lg_den[i0, i1, i2].set_value(adjust_lb(m.lg_den[i0, i1, i2],
                                                         lg_den[z, r_, s]))
                m.mu_i[i0, i1, i2].set_value(adjust_lb(m.mu_i[i0, i1, i2],
                                                       mu_i[z, r_, s]))
                m.mu_den[i0, i1, i2].set_value(adjust_lb(m.mu_den[i0, i1, i2],
                                                         mu_den[z, r_, s]))
                m.CP_i[i0, i1, i2].set_value(adjust_lb(m.CP_i[i0, i1, i2],
                                                       CP_i[z, r_, s]))
                m.Dim_[i0, i1, i2].set_value(adjust_lb(m.Dim_[i0, i1, i2],
                                                       Dim_[z, r_, s]))
                m.Diez_[i0, i1, i2].set_value(adjust_lb(m.Diez_[i0, i1, i2],
                                                        Diez_[z, r_, s]))

    for z in range(len(Zs)):
        for r in range(len(Rs)):
            r_ = 0 if m2 else r
            (i0, i1) = (Zs.at(z+1), Rs.at(r+1))
            m.y[i0, i1].set_value(adjust_lb(m.y[i0, i1], y[z, r_]))
            m.Pt[i0, i1].set_value(adjust_lb(m.Pt[i0, i1], Pt[z, r_]))
            m.T[i0, i1].set_value(adjust_lb(m.T[i0, i1], T[z, r_]))
            m.Ft[i0, i1].set_value(adjust_lb(m.Ft[i0, i1], Ft[z, r_]))
            m.DEN[i0, i1].set_value(adjust_lb(m.DEN[i0, i1], DEN[z, r_]))
            m.k1[i0, i1].set_value(adjust_lb(m.k1[i0, i1], k1[z, r_]))
            m.k2[i0, i1].set_value(adjust_lb(m.k2[i0, i1], k2[z, r_]))
            m.k3[i0, i1].set_value(adjust_lb(m.k3[i0, i1], k3[z, r_]))
            if not(m.dy[i0, i1].stale):
                m.dy[i0, i1].set_value(adjust_lb(m.dy[i0, i1], dy[z, r_]))
            m.dTz[i0, i1].set_value(adjust_lb(m.dTz[i0, i1], dTz[z, r_]))
            m.dTr[i0, i1].set_value(adjust_lb(m.dTr[i0, i1], dTr[z, r_]))
            m.T_Z[i0, i1].set_value(adjust_lb(m.T_Z[i0, i1], T_Z[z, r_]))
            if not(m.dT_Zz[i0, i1].stale):
                m.dT_Zz[i0, i1].set_value(adjust_lb(m.dT_Zz[i0, i1], dT_Zz[z,
                                                                           r_]))
            m.T_Z[i0, i1].set_value(adjust_lb(m.T_Z[i0, i1], T_R[z, r_]))
            if not(m.dT_Rr[i0, i1].stale):
                m.dT_Rr[i0, i1].set_value(adjust_lb(m.dT_Rr[i0, i1], dT_Rr[z,
                                                                           r_]))
            m.lambda_g[i0, i1].set_value(adjust_lb(m.lambda_g[i0, i1],
                                                   lambda_g[z, r_]))
            m.mu[i0, i1].set_value(adjust_lb(m.mu[i0, i1], mu[z, r_]))
            m.CP_g[i0, i1].set_value(adjust_lb(m.CP_g[i0, i1], CP_g[z, r_]))
            m.edC_ZCp[i0, i1].set_value(adjust_lb(m.edC_ZCp[i0, i1], edC_ZCp[z,
                                                                             r_]))
            m.edC_RCp[i0, i1].set_value(adjust_lb(m.edC_RCp[i0, i1], edC_RCp[z,
                                                                             r_]))

    for z in range(len(Zs)):
        for r in range(len(Rs)):
            r_ = 0 if m2 else r
            for s in range(len(Rxs)):
                (i0, i1, i2) = (Zs.at(z+1), Rs.at(r+1), Rxs.at(s+1))
                m.Rate[i0, i1, i2].set_value(adjust_lb(m.Rate[i0, i1, i2],
                                                       Rate[z, r_, s]))
                m.Ke[i0, i1, i2].set_value(adjust_lb(m.Ke[i0, i1, i2], Ke[z, r_,
                                                                          s]))


    for z in range(len(Zs)):
        for r in range(len(Rs)):
            r_ = 0 if m2 else r
            for s0 in range(len(Ss)):
                for s1 in range(len(Ss)):
                    (i0, i1, i2, i3) = (Zs.at(z+1), Rs.at(r+1), Ss.at(s0+1), Ss.at(s1+1))
                    m.lambda_tr[i0, i1, i2,
                                i3].set_value(adjust_lb(m.lambda_tr[i0, i1, i2,
                                                                    i3],
                                                        lambda_tr[z, r_, s0,
                                                                  s1]))
                    m.A_lambda_ij[i0, i1, i2,
                                  i3].set_value(adjust_lb(m.A_lambda_ij[i0, i1,
                                                                        i2, i3],
                                                          A_lambda_ij[z, r_, s0,
                                                                      s1]))
                    m.omegaD[i0, i1, i2, i3].set_value(adjust_lb(m.omegaD[i0,
                                                                          i1,
                                                                          i2,
                                                                          i3],
                                                                 omegaD[z, r_,
                                                                        s0,
                                                                        s1]))
                    m.Dij_[i0, i1, i2, i3].set_value(adjust_lb(m.Dij_[i0, i1,
                                                                      i2, i3],
                                                               Dij_[z, r_, s0,
                                                                    s1]))

    for z in range(len(Zs)):
        idx = Zs.at(z+1)
        m.Re[idx].set_value(adjust_lb(m.Re[idx], Re[z]))
        m.Pr[idx].set_value(adjust_lb(m.Pr[idx], Pr[z]))
        m.U[idx].set_value(adjust_lb(m.U[idx], U[z]))

