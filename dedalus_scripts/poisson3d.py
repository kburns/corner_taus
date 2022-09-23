

import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3


def build_solver(Nx, Ny, Nz, Lx, Ly, Lz, dtype, k_tau_int, k_tau_face, k_lift_int, k_lift_face, k_lift_edge, bc):

    # Bases
    coords = d3.CartesianCoordinates('x', 'y', 'z')
    dist = d3.Distributor(coords, dtype=dtype)
    xb = d3.ChebyshevT(coords['x'], Nx, bounds=(0, Lx))
    yb = d3.ChebyshevT(coords['y'], Ny, bounds=(0, Ly))
    zb = d3.ChebyshevT(coords['z'], Nz, bounds=(0, Lz))
    xb_tau_int = xb.derivative_basis(k_tau_int)
    yb_tau_int = yb.derivative_basis(k_tau_int)
    zb_tau_int = zb.derivative_basis(k_tau_int)
    xb_tau_face = xb.derivative_basis(k_tau_face)
    yb_tau_face = yb.derivative_basis(k_tau_face)
    zb_tau_face = zb.derivative_basis(k_tau_face)
    xb_lift_int = xb.derivative_basis(k_lift_int)
    yb_lift_int = yb.derivative_basis(k_lift_int)
    zb_lift_int = zb.derivative_basis(k_lift_int)
    xb_lift_face = xb.derivative_basis(k_lift_face)
    yb_lift_face = yb.derivative_basis(k_lift_face)
    zb_lift_face = zb.derivative_basis(k_lift_face)
    xb_lift_edge = xb.derivative_basis(k_lift_edge)
    yb_lift_edge = yb.derivative_basis(k_lift_edge)
    zb_lift_edge = zb.derivative_basis(k_lift_edge)

    # RHS fields
    f = dist.Field(name='f', bases=(xb, yb, zb))
    u_T = dist.Field(name='uT', bases=(xb, yb))
    u_B = dist.Field(name='uB', bases=(xb, yb))
    u_N = dist.Field(name='uN', bases=(xb, zb))
    u_S = dist.Field(name='uS', bases=(xb, zb))
    u_E = dist.Field(name='uE', bases=(yb, zb))
    u_W = dist.Field(name='uW', bases=(yb, zb))

    # Variables
    u = dist.Field(name='u', bases=(xb, yb, zb))
    txy = [dist.Field(bases=(xb_tau_int, yb_tau_int)) for i in range(2)]
    txz = [dist.Field(bases=(xb_tau_int, zb_tau_int)) for i in range(2)]
    tyz = [dist.Field(bases=(yb_tau_int, zb_tau_int)) for i in range(2)]
    tx = [dist.Field(bases=xb_tau_face) for i in range(8)]
    ty = [dist.Field(bases=yb_tau_face) for i in range(8)]
    tz = [dist.Field(bases=zb_tau_face) for i in range(8)]
    tc = [dist.Field() for i in range(33)]
    vars = [u] + txy + txz + tyz + tx + ty + tz + tc

    # Tau terms
    tau_u = (d3.Lift(tyz[0], xb_lift_int, -1) + d3.Lift(tyz[1], xb_lift_int, -2) +
             d3.Lift(txz[0], yb_lift_int, -1) + d3.Lift(txz[1], yb_lift_int, -2) +
             d3.Lift(txy[0], zb_lift_int, -1) + d3.Lift(txy[1], zb_lift_int, -2))
    tau_T = (d3.Lift(tx[0], yb_lift_face, -1) + d3.Lift(tx[1], yb_lift_face, -2) +
             d3.Lift(ty[0], xb_lift_face, -1) + d3.Lift(ty[1], xb_lift_face, -2))
    tau_B = (d3.Lift(tx[2], yb_lift_face, -1) + d3.Lift(tx[3], yb_lift_face, -2) +
             d3.Lift(ty[2], xb_lift_face, -1) + d3.Lift(ty[3], xb_lift_face, -2))
    tau_N = (d3.Lift(tx[4], zb_lift_face, -1) + d3.Lift(tx[5], zb_lift_face, -2) +
             d3.Lift(tz[0], xb_lift_face, -1) + d3.Lift(tz[1], xb_lift_face, -2))
    tau_S = (d3.Lift(tx[6], zb_lift_face, -1) + d3.Lift(tx[7], zb_lift_face, -2) +
             d3.Lift(tz[2], xb_lift_face, -1) + d3.Lift(tz[3], xb_lift_face, -2))
    tau_E = (d3.Lift(ty[4], zb_lift_face, -1) + d3.Lift(ty[5], zb_lift_face, -2) +
             d3.Lift(tz[4], yb_lift_face, -1) + d3.Lift(tz[5], yb_lift_face, -2))
    tau_W = (d3.Lift(ty[6], zb_lift_face, -1) + d3.Lift(ty[7], zb_lift_face, -2) +
             d3.Lift(tz[6], yb_lift_face, -1) + d3.Lift(tz[7], yb_lift_face, -2))
    tau_TN = d3.Lift(tc[ 9], xb_lift_edge, -1) + d3.Lift(tc[10], xb_lift_edge, -2)
    tau_TS = d3.Lift(tc[11], xb_lift_edge, -1) + d3.Lift(tc[12], xb_lift_edge, -2)
    tau_TE = d3.Lift(tc[13], yb_lift_edge, -1) + d3.Lift(tc[14], yb_lift_edge, -2)
    tau_TW = d3.Lift(tc[15], yb_lift_edge, -1) + d3.Lift(tc[16], yb_lift_edge, -2)
    tau_BN = d3.Lift(tc[17], xb_lift_edge, -1) + d3.Lift(tc[18], xb_lift_edge, -2)
    tau_BS = d3.Lift(tc[19], xb_lift_edge, -1) + d3.Lift(tc[20], xb_lift_edge, -2)
    tau_BE = d3.Lift(tc[21], yb_lift_edge, -1) + d3.Lift(tc[22], yb_lift_edge, -2)
    tau_BW = d3.Lift(tc[23], yb_lift_edge, -1) + d3.Lift(tc[24], yb_lift_edge, -2)
    tau_NE = d3.Lift(tc[25], zb_lift_edge, -1) + d3.Lift(tc[26], zb_lift_edge, -2)
    tau_SE = d3.Lift(tc[27], zb_lift_edge, -1) + d3.Lift(tc[28], zb_lift_edge, -2)
    tau_NW = d3.Lift(tc[29], zb_lift_edge, -1) + d3.Lift(tc[30], zb_lift_edge, -2)
    tau_SW = d3.Lift(tc[31], zb_lift_edge, -1) + d3.Lift(tc[32], zb_lift_edge, -2)

    # BC operators
    dx = lambda A: d3.Differentiate(A, coords['x'])
    dy = lambda A: d3.Differentiate(A, coords['y'])
    dz = lambda A: d3.Differentiate(A, coords['z'])
    if bc == "dirichlet":
        bc_T = lambda A: A(z=Lz)
        bc_B = lambda A: A(z=0)
        bc_N = lambda A: A(y=Ly)
        bc_S = lambda A: A(y=0)
        bc_E = lambda A: A(x=Lx)
        bc_W = lambda A: A(x=0)
    elif bc == "neumann":
        bc_T = lambda A: dz(A)(z=Lz)
        bc_B = lambda A: -dz(A)(z=0)
        bc_N = lambda A: dy(A)(y=Ly)
        bc_S = lambda A: -dy(A)(y=0)
        bc_E = lambda A: dx(A)(x=Lx)
        bc_W = lambda A: -dx(A)(x=0)
    elif bc == "robin":
        bc_T = lambda A: (A+dz(A))(z=Lz)
        bc_B = lambda A: (A-dz(A))(z=0)
        bc_N = lambda A: (A+dy(A))(y=Ly)
        bc_S = lambda A: (A-dy(A))(y=0)
        bc_E = lambda A: (A+dx(A))(x=Lx)
        bc_W = lambda A: (A-dx(A))(x=0)

    # Problem
    problem = d3.LBVP(vars)

    # Interior
    problem.add_equation((d3.Laplacian(u) + tau_u + tc[0], f))

    # Surface BCs
    problem.add_equation((bc_T(u) + tau_T, u_T))
    problem.add_equation((bc_B(u) + tau_B, u_B))
    problem.add_equation((bc_N(u) + tau_N, u_N))
    problem.add_equation((bc_S(u) + tau_S, u_S))
    problem.add_equation((bc_E(u) + tau_E, u_E))
    problem.add_equation((bc_W(u) + tau_W, u_W))

    # Edge compatibility
    problem.add_equation((bc_N(tau_T) + bc_T(tau_N) + tau_TN, 0))
    problem.add_equation((bc_S(tau_T) + bc_T(tau_S) + tau_TS, 0))
    problem.add_equation((bc_E(tau_T) + bc_T(tau_E) + tau_TE, 0))
    problem.add_equation((bc_W(tau_T) + bc_T(tau_W) + tau_TW, 0))
    problem.add_equation((bc_N(tau_B) + bc_B(tau_N) + tau_BN, 0))
    problem.add_equation((bc_S(tau_B) + bc_B(tau_S) + tau_BS, 0))
    problem.add_equation((bc_E(tau_B) + bc_B(tau_E) + tau_BE, 0))
    problem.add_equation((bc_W(tau_B) + bc_B(tau_W) + tau_BW, 0))
    problem.add_equation((bc_N(tau_E) + bc_E(tau_N) + tau_NE, 0))
    problem.add_equation((bc_S(tau_E) + bc_E(tau_S) + tau_SE, 0))
    problem.add_equation((bc_N(tau_W) + bc_W(tau_N) + tau_NW, 0))
    problem.add_equation((bc_S(tau_W) + bc_W(tau_S) + tau_SW, 0))

    # Interior tau degeneracies
    problem.add_equation((txy[0](x=0), 0))
    problem.add_equation((txy[0](x=Lx), 0))
    problem.add_equation((txy[1](x=0), 0))
    problem.add_equation((txy[1](x=Lx), 0))
    problem.add_equation((txz[0](z=0), 0))
    problem.add_equation((txz[0](z=Lz), 0))
    problem.add_equation((txz[1](z=0), 0))
    problem.add_equation((txz[1](z=Lz), 0))
    problem.add_equation((tyz[0](y=0) + d3.Lift(tc[1], zb_tau_int, -1) + d3.Lift(tc[2], zb_tau_int, -2), 0))
    problem.add_equation((tyz[0](y=Ly) + d3.Lift(tc[3], zb_tau_int, -1) + d3.Lift(tc[4], zb_tau_int, -2), 0))
    problem.add_equation((tyz[1](y=0) + d3.Lift(tc[5], zb_tau_int, -1) + d3.Lift(tc[6], zb_tau_int, -2), 0))
    problem.add_equation((tyz[1](y=Ly) + d3.Lift(tc[7], zb_tau_int, -1) + d3.Lift(tc[8], zb_tau_int, -2), 0))

    # Surface tau degeneracies
    problem.add_equation((tx[0](x=0), 0))
    problem.add_equation((tx[0](x=Lx), 0))
    problem.add_equation((tx[1](x=0), 0))
    problem.add_equation((tx[1](x=Lx), 0))
    problem.add_equation((tx[2](x=0), 0))
    problem.add_equation((tx[2](x=Lx), 0))
    problem.add_equation((tx[3](x=0), 0))
    problem.add_equation((tx[3](x=Lx), 0))
    problem.add_equation((tx[4](x=0), 0))
    problem.add_equation((tx[4](x=Lx), 0))
    problem.add_equation((tx[5](x=0), 0))
    problem.add_equation((tx[5](x=Lx), 0))
    problem.add_equation((tx[6](x=0), 0))
    problem.add_equation((tx[6](x=Lx), 0))
    problem.add_equation((tx[7](x=0), 0))
    problem.add_equation((tx[7](x=Lx), 0))
    problem.add_equation((ty[4](y=0), 0))
    problem.add_equation((ty[4](y=Ly), 0))
    problem.add_equation((ty[5](y=0), 0))
    problem.add_equation((ty[5](y=Ly), 0))
    problem.add_equation((ty[6](y=0), 0))
    problem.add_equation((ty[6](y=Ly), 0))
    problem.add_equation((ty[7](y=0), 0))
    problem.add_equation((ty[7](y=Ly), 0))

    # Corner conditions
    problem.add_equation((bc_W(bc_N(tau_T)) + bc_N(bc_T(tau_W)) + bc_T(bc_W(tau_N)), 0))
    problem.add_equation((bc_E(bc_N(tau_T)) + bc_N(bc_T(tau_E)) + bc_T(bc_E(tau_N)), 0))
    problem.add_equation((bc_W(bc_S(tau_T)) + bc_S(bc_T(tau_W)) + bc_T(bc_W(tau_S)), 0))
    problem.add_equation((bc_E(bc_S(tau_T)) + bc_S(bc_T(tau_E)) + bc_T(bc_E(tau_S)), 0))
    problem.add_equation((bc_W(bc_N(tau_B)) + bc_N(bc_B(tau_W)) + bc_B(bc_W(tau_N)), 0))
    problem.add_equation((bc_E(bc_N(tau_B)) + bc_N(bc_B(tau_E)) + bc_B(bc_E(tau_N)), 0))
    problem.add_equation((bc_W(bc_S(tau_B)) + bc_S(bc_B(tau_W)) + bc_B(bc_W(tau_S)), 0))
    problem.add_equation((bc_E(bc_S(tau_B)) + bc_S(bc_B(tau_E)) + bc_B(bc_E(tau_S)), 0))

    # Gauge
    if bc == "neumann":
        problem.add_equation((d3.Integrate(u), 0))
    else:
        problem.add_equation((tc[0], 0))

    # Solver
    solver = problem.build_solver()
    return solver


if __name__ == "__main__":

    N = 8
    L = 1
    dtype = np.float64
    k_tau_int = 2
    k_tau_face = 0
    k_lift_int = 2
    k_lift_face = 0
    k_lift_edge = 0

    for bc in ["dirichlet", "neumann", "robin"]:

        solver = build_solver(N, N, N, L, L, L, dtype, k_tau_int, k_tau_face, k_lift_int, k_lift_face, k_lift_edge, bc)
        solver.print_subproblem_ranks()

        L_min = solver.subproblems[0].L_min

        L_pool = L_min.A
        P = 1
        if P > 1:
            I, J = L_min.shape
            L_pool = L_pool[:(I//P)*P, :(J//P)*P].reshape(I//P, P, J//P, P).max(axis=(1, 3))

        plt.figure(figsize=(10,10))
        matrix = np.log10(np.abs(L_pool))
        matrix[matrix == - np.inf] = np.nan
        cmap = plt.cm.viridis
        norm = plt.Normalize(vmin=np.nanmin(matrix), vmax=np.nanmax(matrix))
        image = cmap(norm(matrix))
        image[np.isnan(matrix)] = 1
        plt.imsave(f"3d_{bc}.png", image)
        plt.clf()

        LA = L_min.A
        L_sub = []
        n_no_tau = 0
        for i in range(LA.shape[0]):
            if not np.any(LA[i,N**3:]):
                L_sub.append(LA[i,:N**3].copy())
                n_no_tau += 1
        print(n_no_tau)
        L_sub = np.array(L_sub)
        print(L_sub.shape)
        print(np.linalg.cond(L_sub))

        # M = N**3
        # A = LA[:M, :M]
        # B = LA[:M, M:]
        # C = LA[M:, :M]
        # D = LA[M:, M:]
        # Dinv = np.linalg.inv(D)
        # A_SC = A - B @ Dinv @ C
        # print("Schur complement cond:", np.linalg.cond(A_SC))
