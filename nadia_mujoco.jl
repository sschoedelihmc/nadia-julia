using Pkg; Pkg.activate(@__DIR__);
using Revise
using QuadrupedControl
using JLD2
using ForwardDiff
using SparseArrays
using Plots; gr(); theme(:dark); default(:size, (1920, 1080)); scalefontsizes(1.2)
using Statistics
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
data = init_data(model, intf, preferred_monitor=3);

# Set up standing i.c. and solve for stabilizing control
ang = 40*pi/180
x_lin = [0; 0; 0.88978022; 1; zeros(5); -ang; 2*ang; -ang; zeros(3); -ang; 2*ang; -ang; zeros(model.nx - 18)]
u_lin = vcat(calc_continuous_eq(model, x_lin)...)

# Create linear MPC around x_lin, u_lin
dt = 0.01
body_pos, arm_pos, leg_pos, spine_pos = 1e4*ones(6), 1e4*ones(4), 1e4*ones(6), 1e4*ones(3)
body_vel, arm_vel, leg_vel, spine_vel = 1e1*body_pos, 1e1*arm_pos, 1e1*leg_pos, 1e1*spine_pos
arm_u, leg_u, spine_u = 1e-4*ones(4), [1e-4*ones(5); 1e5], 1e-4*ones(3)
Q = spdiagm([body_pos; leg_pos; leg_pos; spine_pos; arm_pos; arm_pos; body_vel; leg_vel; leg_vel; spine_vel; arm_vel; arm_vel])
R = spdiagm([leg_u; leg_u; spine_u; arm_u; arm_u]);
mpc = LinearizedRK4MPC(model, x_lin, u_lin, dt/100, Q, R, 10);

# Simulate using MuJoCo
simulation_time_step = intf.m.opt.timestep*4
end_time = 3
N = Int(floor(end_time/simulation_time_step))
X = [zeros(model.nx) for _ = 1:N];
X_d = [zeros(model.nx) for _ = 1:N];
U = [zeros(model.nu) for _ = 1:N-1];
X[1] = deepcopy(x_lin);
X_d[1] = deepcopy(x_lin);
# X[1][model.nq + 5] = 1.5

# Set ref to the shifted position
ref = quasi_shift_foot_lift()
shift_stand_ref = LinearizedQuadRef(model, ref.X_ref[551:551], ref.U_ref[551:551], x_lin, u_lin, dt, nc = model.nc)
mpc.ref = shift_stand_ref

# Todo check linear dynamics
A_lin, B_lin, lqr_K = mpc.A_lin, mpc.B_lin, mpc.lqr_K
Ā = A_lin - B_lin[:, 1:model.nu]*lqr_K
J = kinematics_velocity_jacobian(model, x_lin)*error_jacobian(model, x_lin)

U, S, V = svd(J) # Careful, U and V are not unitary, only semi-unitary in Julia's version (kinda like compact SVD but instead S is minimum(size(J)))
P_null = I - V[:, 1:12]*V[:, 1:12]'
P_range = U[:, 1:12]' # Project into subspace R(A)

# Use implicit function theorem to solve least squares for λ as a function of X, i.e minimize
# ||P_range*J*Ā*x + P_range*J*B_lin[:, model.nu + 1:end]*λ||²
ls_A = P_range*J*B_lin[:, model.nu + 1:end]
ls_B = P_range*J*Ā

dλ_dx = ls_A'*((ls_A*ls_A') \ (ls_B))

Ā_cl = Ā - B_lin[:, model.nu + 1:end]*dλ_dx






# Run simulation
data.x = copy(X[1]); data.u = u_lin[1:model.nu]; data.t = 0; warmstart!(model, mpc, data); 
for k = 1:N - 1
    # Set MuJoCo state
    data.x .= X[k]
    data.t = k*simulation_time_step
    set_data!(model, intf, data)

    # Calc control
    X_d[k + 1], U[k], _, _, status = get_ctrl(model, mpc, data)

    # Update control
    data.u .= U[k]
    set_data!(model, intf, data)

    # Take a step
    @lock intf.p.lock begin
        for _ = 1:4 # effective control rate of 500 Hz
            MuJoCo.mj_step(intf.m, intf.d)
        end
    end

    # Get data
    get_data!(intf, data)
    X[k + 1] = copy(data.x)

    # if norm(state_error(model, X[k+1], X_d[k+1]), Inf) >= 1
    #     @warn "Deviated too much from reference, exiting"
    #     break
    # end

    # Integrate
    # global X[k + 1] = rk4(model, X[k], U[k], simulation_time_step; gains=model.baumgarte_gains)
end
plot_tracking()
ΔX = [state_error(model, x, x_lin) for x in X];
anim = animate(model_fixed, mvis, [change_order(model_fixed, x, :mujoco, :rigidBodyDynamics) for x in X]; Δt=simulation_time_step, frames_to_skip=50);
setanimation!(mvis, anim)

# Get continuous time As and Bs
using FiniteDiff
E_jac = error_jacobian(model, x_lin)
E_jac_T = error_jacobian_T(model, x_lin)
A = E_jac_T*FiniteDiff.finite_difference_jacobian(_x -> continuous_dynamics(model, _x, u_lin[1:model.nu], u_lin[model.nu + 1:end]), x_lin)*E_jac
B = E_jac_T*FiniteDiff.finite_difference_jacobian(_u -> continuous_dynamics(model, x_lin, _u, u_lin[model.nu + 1:end]), u_lin[1:model.nu])
C = E_jac_T*FiniteDiff.finite_difference_jacobian(_u -> continuous_dynamics(model, x_lin, u_lin[1:model.nu], _u), u_lin[model.nu + 1:end])
J = kinematics_jacobian(model, x_lin)*E_jac

@save joinpath(@__DIR__, "cont_time_dynamics.jld2") A B C J