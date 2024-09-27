using Pkg; Pkg.activate(@__DIR__);
using Revise
using QuadrupedControl
using JLD2
using FiniteDiff
using SparseArrays
using Plots; gr(); theme(:dark); default(:size, (1920, 1080)); scalefontsizes(1.2)
using Statistics
import ControlSystems:lqr, Discrete, are

import MuJoCo

include(joinpath(@__DIR__, "nadia_robot_fixed_foot.jl"))
include(joinpath(@__DIR__, "plot_utils.jl"))
include(joinpath(@__DIR__, "references.jl"))
include("control_utils.jl")
model_fixed = NadiaFixed();
generate_mujoco_stateorder(model_fixed);
vis = Visualizer();
mvis = init_visualizer(model_fixed, vis)
model = Nadia(nc_per_foot = 4)
intf = init_mujoco_interface(model)
intf.sim_rate = intf.m.opt.timestep
data = init_data(model, intf, preferred_monitor=3);

# let
    # global K_pd, K_inf

## Set up standing i.c. and solve for stabilizing control
# let
    global K_pd, K_qp, mpc
K_pd = [zeros(model.nu, 7) diagm(1e1*ones(model.nu)) zeros(model.nu, 6) 1e0diagm(ones(model.nu))]
ang = 40*pi/180;
x_lin = [0; 0; 0.892; 1; 0; 0; 0; zeros(2); -ang; 2*ang; -ang; zeros(3); -ang; 2*ang; -ang; zeros(4); repeat([0; 0; 0; -pi/4], 2); zeros(model.nv)];
data.x = copy(x_lin); set_data!(model, intf, data);
u_lin = vcat(calc_continuous_eq(model, x_lin, K=K_pd, verbose = true)...);
u_lin = [u_lin[1:model.nu]; zeros(model.nc*3); u_lin[model.nu + 1:end]] # Update to include constraint forces on position-velocity kinematics
J_func(model, x) = BlockDiagonal([kinematics_jacobian(model, x)[:, 1:model.nq], kinematics_jacobian(model, x)[:, 1:model.nq]*velocity_kinematics(model, x)])

# Confirm that we do have an eq point with the full constraint
@assert norm(continuous_dynamics(model, x_lin, u_lin[1:model.nu], λ = u_lin[model.nu + 1:end], J_func=J_func, K=K_pd), Inf) < 1e-10

# Linearize dynamics (continuous time), constraints on position and velocity
E_jac = error_jacobian(model, x_lin);
E_jac_T = error_jacobian_T(model, x_lin);
cont_dyn_func(model, x, u, λ) = continuous_dynamics(model, x, u, λ=λ, J_func=J_func, K=K_pd)
A = E_jac_T*FiniteDiff.finite_difference_jacobian(_x -> cont_dyn_func(model, _x, u_lin[1:model.nu], u_lin[model.nu + 1:end]), x_lin)*E_jac;
B = E_jac_T*FiniteDiff.finite_difference_jacobian(_u -> cont_dyn_func(model, x_lin, _u, u_lin[model.nu + 1:end]), u_lin[1:model.nu]);
C = E_jac_T*FiniteDiff.finite_difference_jacobian(_λ -> cont_dyn_func(model, x_lin, u_lin[1:model.nu], _λ), u_lin[model.nu + 1:end]);
J = J_func(model, x_lin)*E_jac

# The dynamics system we are solving for (in x_{k+1} and λ_k)) is
# (I - dt*A)x_{k+1} - J'λ_k = x_k + dt*Bu_k
# s.t. Jx_{k+1} - 1/ρPλ_k = 0
# x ∈ R^58, u ∈ R^23, λ ∈ R^48 and J is 48 × 58 but rank(J) = 24. By default P = I(48)

# Parameters (horizon, timestep, penalty, sizing)
N = 3;
dt = 0.01;
ρ = 1e5;

# Get Pinf from LQR
kkt_sys = [I-dt*A -dt*C; J -1/ρ*dt*I]
Ā = (kkt_sys \ [I(model.nx - 1); zeros(size(J, 1), model.nx - 1)])[1:model.nx - 1, :]
B̄ = (kkt_sys \ [dt*B; zeros(size(J, 1), model.nu)])[1:model.nx - 1, :]

# Calculate ihlqr
Q = diagm([1e3ones(model.nv); 5e0ones(model.nv)])
R = diagm(1e-3ones(model.nu));
P_inf = are(Discrete, Ā, B̄, Q, R)
K_inf = lqr(Discrete, Ā, B̄, Q, R)

# Set up cost hessian
# Q = diagm([1e4ones(model.nv); 1e0ones(model.nv)]);
# R = diagm([1e-2ones(model.nu); 0*ones(model.nc*3*2)]);
Q = diagm([1e3ones(model.nv); 5e0ones(model.nv)]);
R = diagm([1e-3ones(model.nu); 0*ones(model.nc*3*2)]);
Qf = P_inf

nc = size(J, 1)
constraints = Vector{NamedTuple}([
        (C=[zeros(nc, model.nx - 1 + model.nu) -1/ρ*dt*I J], l = zeros(nc), u = zeros(nc))]) # Contact constraint (position and velocity)
        # (C=[zeros(model.nc, model.nx - 1 + model.nu + model.nc*3) kron(I(model.nc), [0 0 1]) zeros(model.nc, model.nx - 1)], # Force constraint on z for the velocity level constraint
        #  l = -u_lin[model.nu + model.nc*3 + 1:end][3:3:end], u = fill(Inf, model.nc))])

        
X_ref, U_ref = quasi_shift_foot_lift(shift_ang = 10, tf = 2, K=K_pd, dt=dt);
mpc = LinMPC(model, x_lin, u_lin, [I dt*B dt*C (dt*A - I)], Q, R, Qf, N, constraints, dt=dt, K_pd=K_pd);

# Solve with OSQP (use X_ref to test against)
using OSQP
mpc.ref = LinearizedQuadRef(model, [X_ref[end]], [U_ref[end]], x_lin, u_lin, dt, nc = model.nc)
reset_data!(model, intf, data)
data.x = copy(X_ref[end])
data.u = zeros(model.nu)
set_data!(model, intf, data)
OSQP.update_settings!(mpc.m, verbose = true)
get_ctrl(model, mpc, data);

# Solve with Clarabel
using Clarabel
qp = mpc.qp
cones = [Clarabel.ZeroConeT(size(qp.A, 1))]
settings = Clarabel.Settings(verbose = true, time_limit = 5)
cSolver = Clarabel.Solver()
Clarabel.setup!(cSolver, qp.H, qp.g, qp.A, qp.l, cones, settings)
solution = Clarabel.solve!(cSolver)

# Compare
norm(mpc.results.x - solution.x, Inf)