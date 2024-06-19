using Pkg; Pkg.activate(@__DIR__)

using JLD2
using MeshCatMechanisms
using RigidBodyDynamics
using RigidBodyDynamics.PDControl

using MeshCat

using StaticArrays
using SparseArrays

import ForwardDiff
import FiniteDiff ### TODO: Delete this later if it doesn't work

include("nadia_robot.jl")
include("control_utils.jl")

##

nadia = Nadia();
vis = Visualizer();
render(vis)

##

# Initialize visualizer with robot meshes
mvis = init_visualizer(nadia, vis)

##

# Load equilibrium reference controls and state
u_ref = load_object("nadia_balance_u_ref.jld2")

# u_ref torques should be in the following order:
# "LEFT_HIP_Z"
# "RIGHT_HIP_Z"
# "SPINE_Z"
# "LEFT_HIP_X"
# "RIGHT_HIP_X"
# "SPINE_X"
# "LEFT_HIP_Y"
# "RIGHT_HIP_Y"
# "SPINE_Y"
# "LEFT_KNEE_Y"
# "RIGHT_KNEE_Y"
# "LEFT_SHOULDER_Y"
# "RIGHT_SHOULDER_Y"
# "LEFT_ANKLE_Y"
# "RIGHT_ANKLE_Y"
# "LEFT_SHOULDER_X"
# "RIGHT_SHOULDER_X"
# "LEFT_ANKLE_X"
# "RIGHT_ANKLE_X"
# "LEFT_SHOULDER_Z"
# "RIGHT_SHOULDER_Z"
# "LEFT_ELBOW_Y"
# "RIGHT_ELBOW_Y"

x_ref = load_object("nadia_balance_x_ref.jld2")
# 4 variables for pelvis orientation (w, x, y, z quaternion)
# 3 variables for pelvis position in world frame
# LEFT_HIP_Z
# RIGHT_HIP_Z
# SPINE_Z
# LEFT_HIP_X
# RIGHT_HIP_X
# SPINE_X
# LEFT_HIP_Y
# RIGHT_HIP_Y
# SPINE_Y
# LEFT_KNEE_Y
# RIGHT_KNEE_Y
# LEFT_SHOULDER_Y
# RIGHT_SHOULDER_Y
# LEFT_ANKLE_Y
# RIGHT_ANKLE_Y
# LEFT_SHOULDER_X
# RIGHT_SHOULDER_X
# LEFT_ANKLE_X
# RIGHT_ANKLE_X
# LEFT_SHOULDER_Z
# RIGHT_SHOULDER_Z
# LEFT_ELBOW_Y
# RIGHT_ELBOW_Y
# 29 variables for reference velocities in the same order 
#   as config but with three variables for orientation 
#   (generally all zeros for an equilibrium position)

##

# Set pose in visualizer
set_configuration!(mvis, x_ref[1:nadia.nq])

##

dt = 1e-3
ADyn = ForwardDiff.jacobian(x_ -> rk4(nadia, x_, u_ref, dt), x_ref)
BDyn = ForwardDiff.jacobian(u_ -> rk4(nadia, x_ref, u_, dt), u_ref)

ADynReduced = E(x_ref[1:4])' * ADyn * E(x_ref[1:4])
BDynReduced = E(x_ref[1:4])' * BDyn

# ADynReduced_nadia_balance_1 = load_object("state_transition_matrix_A_reduced_form.jld2")
# BDynReduced_nadia_balance_1 = load_object("input_matrix_B_reduced_form.jld2")
# maximum(abs.(ADynReduced1 - ADynReduced_nadia_balance_1))
# maximum(abs.(BDynReduced1 - BDynReduced_nadia_balance_1))

# ADynReduced = ADynReduced1
# BDynReduced = BDynReduced1

##

# Compute IHLQR optimal feedback gain matrix Kinf

# Tune these if you want
Q = spdiagm([repeat([5e2], 6); repeat([1e-3, 1e-3, 1e3], 3); 1e2; 1e2; repeat([5e-4; 5e-4; 1e1; 1e-5], 2); repeat([1e-2], 4);
                repeat([1e1], 6); repeat([1e1, 1e1, 1e1], 3); 1e2; 1e2; repeat([1e1; 1e1; 1e1; 1e-4], 2); repeat([1e1], 4)]);
R = spdiagm(1e-2*ones(size(BDynReduced)[2]));

Kinf, Qf = ihlqr(ADynReduced, BDynReduced, Q, R, Q, max_iters = 200000);

# Check eigenvalues of system
eigvals(ADynReduced - BDynReduced*Kinf)

Kinf_nadia_balance_1 = load_object("Kinf_nadia_balance_1.jld2")
maximum(abs.(Kinf - Kinf_nadia_balance_1))

##

# # Create LQR controller
# function lqr_controller!(torques::AbstractVector, t, current_state::MechanismState)
#     global index
#     current_x = [current_state.q[1:end]; current_state.v[1:end]] # [1:end] to extract Vector{Float64} from segmented vector
#     Δx̃ = [qtorp(L(x_ref[1:4])'*current_x[1:4]); current_x[5:end] - x_ref[5:end]]
#     Δu = -Kinf * Δx̃
#     torques[1:end] = [zeros(6); Δu + u_ref]
# end

# copyto!(nadia.state, x_ref)
# nadia.state.v[5] = 1.2 # Set initial pelvis X-velocity to mimic a push disturbance

# t, q, v = simulate(nadia.state, 3.0, lqr_controller!; Δt=0.001, stabilization_gains=nadia.baumgarte_gains);
# animation = Animation(mvis, t, q);
# setanimation!(mvis, animation);

##

simulation_time_step = 0.001
end_time = 3.0

N = Int(floor(end_time/simulation_time_step))
X = [zeros(length(x_ref)) for _ = 1:N];
U = [zeros(length(u_ref)) for _ = 1:N];
X[1] = deepcopy(x_ref);
X[1][nadia.nq + 5] = -1.3; # Perturb i.c.

# Run simulation
for k = 1:N - 1
    # Get error
    global Δx̃ = [qtorp(L(x_ref[1:4])'*X[k][1:4]); X[k][5:end] - x_ref[5:end]]
  
    # Compute controls for this time step
    global U[k] = u_ref - Kinf*Δx̃

    # Integrate
    global X[k + 1] = rk4(nadia, X[k], U[k], simulation_time_step; gains=nadia.baumgarte_gains)
end
anim = animate(nadia, mvis, X, Δt=simulation_time_step, division=100);
setanimation!(mvis, anim)