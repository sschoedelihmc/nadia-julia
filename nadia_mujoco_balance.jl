using Pkg; Pkg.activate(@__DIR__);
using QuadrupedControl
using JLD2
using ForwardDiff
using SparseArrays
using Plots; plotlyjs(); theme(:dark)
import MuJoCo

include(joinpath(@__DIR__, "nadia_robot_fixed_foot.jl"))
include("control_utils.jl")
model = NadiaFixed();
intf = init_mujoco_interface(model)
data = init_data(model, intf, preferred_monitor=3)

# Load reference
x_ref = load_object("nadia_balance_x_ref.jld2"); x_ref[1:4] = normalize(x_ref[1:4])
u_ref = load_object("nadia_balance_u_ref.jld2")
data.x .= change_order(model, x_ref, :rigidBodyDynamics, :mujoco)
data.u .= change_order(model, u_ref, :rigidBodyDynamics, :mujoco)
set_data!(model, intf, data)

# Quick check
state = MechanismState(model.mech)
set_configuration!(state, x_ref[1:model.nq])
M1 = mass_matrix(state)
M2 = change_order(model, M_func(Nadia(), change_order(model, [configuration(state); zeros(model.nv)], :rigidBodyDynamics, :mujoco)), :mujoco, :rigidBodyDynamics)
kinematics(Nadia(), change_order(model, [configuration(state); zeros(model.nv)], :rigidBodyDynamics, :mujoco))

# Create LQR
dt = intf.m.opt.timestep
ADyn = ForwardDiff.jacobian(x_ -> rbd_rk4(model, x_, u_ref, dt), x_ref)
BDyn = ForwardDiff.jacobian(u_ -> rbd_rk4(model, x_ref, u_, dt), u_ref)

# Reduce quaternion representation to a form we can do math with
E_jac = error_jacobian(Nadia(), change_order(model, x_ref, :rigidBodyDynamics, :mujoco));
E_jac_T = error_jacobian_T(Nadia(), change_order(model, x_ref, :rigidBodyDynamics, :mujoco));
ADynReduced = E(x_ref[1:4])' * ADyn * E(x_ref[1:4])
BDynReduced = E(x_ref[1:4])' * BDyn

# Compute IHLQR optimal feedback gain matrix Kinf
Q = spdiagm([repeat([5e2], 6); repeat([1e-3, 1e-3, 1e3], 3); 1e2; 1e2; repeat([5e1; 5e1; 1e3; 1e3], 2); repeat([1e2], 4);
                repeat([1e1], 6); repeat([1e1, 1e1, 1e1], 3); 1e2; 1e2; repeat([1e1; 1e1; 1e1; 1e1], 2); repeat([1e1], 4)]);
R = spdiagm(1e-0*ones(size(BDynReduced)[2]));

ADynReduced = change_order(model, ADynReduced, :rigidBodyDynamics, :mujoco)
BDynReduced = change_order(model, BDynReduced, :rigidBodyDynamics, :mujoco)
Q = change_order(model, Q, :rigidBodyDynamics, :mujoco)
R = change_order(model, R, :rigidBodyDynamics, :mujoco)

Kinf, Qf = ihlqr(ADynReduced, BDynReduced, Q, R, Q; max_iters = 20000);

# Simulate using MuJoCo
vis = Visualizer();
mvis = init_visualizer(model, vis)
simulation_time_step = intf.m.opt.timestep
end_time = 10.0

N = Int(floor(end_time/simulation_time_step))
X = [zeros(length(x_ref)) for _ = 1:N];
U = [zeros(length(u_ref)) for _ = 1:N];
X[1] = deepcopy(x_ref);
# X[1][nadia.nq + 5] = 1.3; # Perturb i.c.
Kinf_mj = Kinf
# Run simulation
for k = 1:1#N - 1
    # Get error
    global Δx̃ = [qtorp(L(x_ref[1:4])'*X[k][1:4]); X[k][5:end] - x_ref[5:end]]
  
    # add some noise
    Δx̃ += 0.019 * randn(length(Δx̃))

    # Compute controls for this time step
    Δx̃ = change_order(model, Δx̃, :rigidBodyDynamics, :mujoco)
    global U[k] = change_order(model, u_ref, :rigidBodyDynamics, :mujoco) - Kinf_mj*Δx̃

    # Apply to MuJoCo
    data.x .= change_order(model, X[k], :rigidBodyDynamics, :mujoco)
    data.u .= U[k]#change_order(model, U[k], :rigidBodyDynamics, :mujoco)
    set_data!(model, intf, data)

    # Take a step
    @lock intf.p.lock begin
        MuJoCo.mj_step(intf.m, intf.d)
    end

    # Get data
    get_data!(intf, data)
    X[k + 1] = change_order(model, data.x, :mujoco, :rigidBodyDynamics)

    # Integrate
    # global X[k + 1] = rk4(model, X[k], U[k], simulation_time_step; gains=model.baumgarte_gains)
end
anim = animate(model, mvis, X; Δt=simulation_time_step, frames_to_skip=50);
setanimation!(mvis, anim)