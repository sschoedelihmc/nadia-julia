using Pkg; Pkg.activate(@__DIR__);
using Revise
using QuadrupedControl
using JLD2
using ForwardDiff
using SparseArrays
using Plots; plotlyjs(); theme(:default)
using Statistics
import MuJoCo

include(joinpath(@__DIR__, "nadia_robot_fixed_foot.jl"))
include("control_utils.jl")
model_fixed = NadiaFixed();
generate_mujoco_stateorder(model_fixed);
model = Nadia(nc_per_foot = 4)
intf = init_mujoco_interface(model)
data = init_data(model, intf, preferred_monitor=3)

# Set up standing i.c. and solve for stabilizing control
ang = 40*pi/180
x_lin = [0; 0; 0.88978022; 1; zeros(5); -ang; 2*ang; -ang; zeros(3); -ang; 2*ang; -ang; zeros(model.nx - 18)]
u_lin = vcat(calc_continuous_eq(model, x_lin)...)

# Create linear MPC around x_lin, u_lin
body_pos, arm_pos, leg_pos, spine_pos = 1e4*ones(6), 1e2*ones(4), 1e4*ones(6), 1e4*ones(3)
body_vel, arm_vel, leg_vel, spine_vel = 1e1*body_pos, 1e1*arm_pos, 1e1*leg_pos, 1e1*spine_pos
Q = spdiagm([body_pos; leg_pos; leg_pos; spine_pos; arm_pos; arm_pos; body_vel; leg_vel; leg_vel; spine_vel; arm_vel; arm_vel])
R = spdiagm(1e-0*ones(model.nu));
mpc = LinearizedRK4MPC(model, x_lin, u_lin, 0.01, Q, R, 10);

# Simulate using MuJoCo
vis = Visualizer();
mvis = init_visualizer(model_fixed, vis)
simulation_time_step = intf.m.opt.timestep*4
end_time = 5.3

N = Int(floor(end_time/simulation_time_step))
X = [zeros(model.nx) for _ = 1:N];
U = [zeros(model.nu) for _ = 1:N];
X[1] = deepcopy(x_lin);
X[1][model.nq + 5] = 1.3; # Perturb i.c.

# Run simulation
for k = 1:N - 1
    # Set MuJoCo state
    data.x .= X[k]
    data.t = k*simulation_time_step
    set_data!(model, intf, data)

    # Calc control
    x_d, U[k], _, _, status = get_ctrl(model, mpc, data)

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

    # Integrate
    # global X[k + 1] = rk4(model, X[k], U[k], simulation_time_step; gains=model.baumgarte_gains)
end
ΔX = [state_error(model, x, x_lin) for x in X];
anim = animate(model_fixed, mvis, [change_order(model_fixed, x, :mujoco, :rigidBodyDynamics) for x in X]; Δt=simulation_time_step, frames_to_skip=50);
setanimation!(mvis, anim)


