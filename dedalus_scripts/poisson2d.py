
import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3


def build_solver(Nx, Ny, Lx, Ly, dtype, k_tau_int, k_lift_int, k_lift_edge, bc, corner_tau, bc_deriv_scale):

    # Bases
    coords = d3.CartesianCoordinates('x', 'y')
    dist = d3.Distributor(coords, dtype=dtype)
    xb = d3.ChebyshevT(coords['x'], Nx, bounds=(0, Lx))
    yb = d3.ChebyshevT(coords['y'], Ny, bounds=(0, Ly))
    xb_tau_int = xb.derivative_basis(k_tau_int)
    yb_tau_int = yb.derivative_basis(k_tau_int)
    xb_lift_int = xb.derivative_basis(k_lift_int)
    yb_lift_int = yb.derivative_basis(k_lift_int)
    xb_lift_edge = xb.derivative_basis(k_lift_edge)
    yb_lift_edge = yb.derivative_basis(k_lift_edge)

    # RHS fields
    f = dist.Field(name='f', bases=(xb, yb))
    uL = dist.Field(name='uL', bases=yb)
    uR = dist.Field(name='uR', bases=yb)
    uB = dist.Field(name='uB', bases=xb)
    uT = dist.Field(name='uT', bases=xb)

    # Variables
    u = dist.Field(name='u', bases=(xb, yb))
    tx = [dist.Field(bases=xb_tau_int) for i in range(2)]
    ty = [dist.Field(bases=yb_tau_int) for i in range(2)]
    tc = [dist.Field() for i in range(9)]
    vars = [u] + tx + ty + tc

    # Tau terms
    tau_u = (d3.Lift(tx[0], yb_lift_int, -1) + d3.Lift(tx[1], yb_lift_int, -2) +
             d3.Lift(ty[0], xb_lift_int, -1) + d3.Lift(ty[1], xb_lift_int, -2))
    tau_L = d3.Lift(tc[4], yb_lift_edge, -1) + d3.Lift(tc[0], yb_lift_edge, -2)
    tau_R = d3.Lift(tc[5], yb_lift_edge, -1) + d3.Lift(tc[1], yb_lift_edge, -2)
    tau_B = d3.Lift(tc[6], xb_lift_edge, -1) + d3.Lift(tc[2], xb_lift_edge, -2)
    tau_T = d3.Lift(tc[7], xb_lift_edge, -1) + d3.Lift(tc[3], xb_lift_edge, -2)

    # BC operators
    dx = lambda A: d3.Differentiate(A, coords['x'])
    dy = lambda A: d3.Differentiate(A, coords['y'])
    if bc == "dirichlet":
        bc_L = lambda A: A(x=0)
        bc_R = lambda A: A(x=Lx)
        bc_B = lambda A: A(y=0)
        bc_T = lambda A: A(y=Ly)
    elif bc == "neumann":
        bc_L = lambda A: -dx(A)(x=0)/Nx**bc_deriv_scale
        bc_R = lambda A: dx(A)(x=Lx)/Nx**bc_deriv_scale
        bc_B = lambda A: -dy(A)(y=0)/Ny**bc_deriv_scale
        bc_T = lambda A: dy(A)(y=Ly)/Ny**bc_deriv_scale
    elif bc == "DN":
        bc_L = lambda A: A(x=0)
        bc_R = lambda A: A(x=Lx)
        bc_B = lambda A: -dy(A)(y=0)/Ny**bc_deriv_scale
        bc_T = lambda A: dy(A)(y=Ly)/Ny**bc_deriv_scale
    elif bc == "robin":
        bc_L = lambda A: (A - dx(A))(x=0)/Nx**bc_deriv_scale
        bc_R = lambda A: (A + dx(A))(x=Lx)/Nx**bc_deriv_scale
        bc_B = lambda A: (A - dy(A))(y=0)/Ny**bc_deriv_scale
        bc_T = lambda A: (A + dy(A))(y=Ly)/Ny**bc_deriv_scale

    # Problem
    problem = d3.LBVP(vars)

    # Interior
    problem.add_equation((d3.Laplacian(u) + tau_u, f))

    # Edge BCs
    if bc == "neumann":
        problem.add_equation((bc_L(u) + tau_L + tc[-1], uL))
        problem.add_equation((bc_R(u) + tau_R + tc[-1], uR))
        problem.add_equation((bc_B(u) + tau_B + tc[-1], uB))
        problem.add_equation((bc_T(u) + tau_T + tc[-1], uT))
    else:
        problem.add_equation((bc_L(u) + tau_L, uL))
        problem.add_equation((bc_R(u) + tau_R, uR))
        problem.add_equation((bc_B(u) + tau_B, uB))
        problem.add_equation((bc_T(u) + tau_T, uT))

    # Corner compatibility
    if corner_tau:
        problem.add_equation((bc_L(tau_B) + bc_B(tau_L), 0))
        problem.add_equation((bc_R(tau_B) + bc_B(tau_R), 0))
        problem.add_equation((bc_L(tau_T) + bc_T(tau_L), 0))
        problem.add_equation((bc_R(tau_T) + bc_T(tau_R), 0))
    else:
        problem.add_equation((bc_L(bc_B(u)) + bc_B(bc_L(u)), bc_L(uB) + bc_B(uL)))
        problem.add_equation((bc_R(bc_B(u)) + bc_B(bc_R(u)), bc_R(uB) + bc_B(uR)))
        problem.add_equation((bc_L(bc_T(u)) + bc_T(bc_L(u)), bc_L(uT) + bc_T(uL)))
        problem.add_equation((bc_R(bc_T(u)) + bc_T(bc_R(u)), bc_R(uT) + bc_T(uR)))

    # Interior tau degeneracies
    problem.add_equation((tx[0](x=0), 0))
    problem.add_equation((tx[0](x=Lx), 0))
    problem.add_equation((tx[1](x=0), 0))
    problem.add_equation((tx[1](x=Lx), 0))

    # Gauge
    if bc == "neumann":
        problem.add_equation((d3.Integrate(u), 0))
    else:
        problem.add_equation((tc[-1], 0))

    # Solver
    solver = problem.build_solver()
    solver.build_matrices()
    return solver


if __name__ == "__main__":

    N = 16
    L = 1
    dtype = np.float64
    k_tau_int = 2
    k_lift_int = 2
    k_lift_edge = 0

    for bc in ["dirichlet", "neumann", "robin"]:

        solver = build_solver(N, N, L, L, dtype, k_tau_int, k_lift_int, k_lift_edge, bc)
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
        plt.imsave(f"2d_{bc}.png", image)
        plt.clf()

        LA = L_min.A
        L_sub = []
        n_no_tau = 0
        for i in range(LA.shape[0]):
            if not np.any(LA[i,N**2:]):
                L_sub.append(LA[i,:N**2].copy())
                n_no_tau += 1
        print(n_no_tau)
        L_sub = np.array(L_sub)
        print(L_sub.shape)
        print(np.linalg.cond(L_sub))

        plt.figure(figsize=(10,10))
        matrix = np.log10(np.abs(L_sub))
        matrix[matrix == - np.inf] = np.nan
        cmap = plt.cm.viridis
        norm = plt.Normalize(vmin=np.nanmin(matrix), vmax=np.nanmax(matrix))
        image = cmap(norm(matrix))
        image[np.isnan(matrix)] = 1
        plt.imsave(f"2d_{bc}_sub.png", image)
        plt.clf()