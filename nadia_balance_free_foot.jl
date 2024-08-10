using Pkg; Pkg.activate(@__DIR__)

using JLD2

using MeshCat
using MeshCatMechanisms
using RigidBodyDynamics
using RigidBodyDynamics.PDControl

using StaticArrays
using SparseArrays

import ForwardDiff

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

# Calculate linearized dynamics
dt = 1e-3
ADyn = ForwardDiff.jacobian(x_ -> rk4(nadia, x_, u_ref, dt), x_ref)
BDyn = ForwardDiff.jacobian(u_ -> rk4(nadia, x_ref, u_, dt), u_ref)

# Reduce quaternion representation to a form we can do math with
ADynReduced = E(x_ref[1:4])' * ADyn * E(x_ref[1:4])
BDynReduced = E(x_ref[1:4])' * BDyn

##

# Compute IHLQR optimal feedback gain matrix Kinf

# Tune these if you want
Q = spdiagm([repeat([5e2], 6); repeat([1e-3, 1e-3, 1e3], 3); 1e2; 1e2; repeat([5e1; 5e1; 1e3; 1e3], 2); repeat([1e2], 4);
                repeat([1e1], 6); repeat([1e1, 1e1, 1e1], 3); 1e2; 1e2; repeat([1e1; 1e1; 1e1; 1e1], 2); repeat([1e1], 4)]);
R = spdiagm(1e-2*ones(size(BDynReduced)[2]));

Kinf, Qf = ihlqr(ADynReduced, BDynReduced, Q, R, Q; max_iters = 200000, verbose=true);

# Check eigenvalues of system
# eigvals(ADynReduced - BDynReduced*Kinf)

##

reference_torques = [
    -1.529996146733067       # "LEFT_HIP_Z"
    9.964441654400309        # "RIGHT_HIP_Z"
    2.4214516921032456       # "SPINE_Z"
    30.725284382117          # "LEFT_HIP_X"
    -102.44994521205273      # "RIGHT_HIP_X"
    -10.165094068114897      # "SPINE_X"
    1.3888702260931227       # "LEFT_HIP_Y"
    30.57276337319791        # "RIGHT_HIP_Y"
    -5.786704258930225       # "SPINE_Y"
    18.379170992769396       # "LEFT_KNEE_Y"
    -87.7703566391235        # "RIGHT_KNEE_Y"
    2.673614241444654        # "LEFT_SHOULDER_Y"
    3.90078753582609         # "RIGHT_SHOULDER_Y"
    0.6569298958706897       # "LEFT_ANKLE_Y"
    31.358889860182014       # "RIGHT_ANKLE_Y"
    7.695142597693708        # "LEFT_SHOULDER_X"
    -4.036609656052576       # "RIGHT_SHOULDER_X"
    0.22082314816501622      # "LEFT_ANKLE_X"
    -0.0021063087760480495   # "RIGHT_ANKLE_X"
    0.4149513856694272       # "LEFT_SHOULDER_Z"
    -0.2869521910778749      # "RIGHT_SHOULDER_Z"
    -0.532071050841635       # "LEFT_ELBOW_Y"
    -0.5439629977475727      # "RIGHT_ELBOW_Y"
]

torques = [
    -0.24816164506842853
    12.05430968128148
    0.7875707412229151
    26.177245225037133
    -105.45508699151762
    -12.929636803036365
    0.016713255622627488
    28.024378358805535
    -7.951043524947613
    17.57222371047697
    -79.21248987248435
    5.6449812176736796
    4.448558679620415
    -2.7037527284340204
    27.65160208391014
    9.88647307825871
    -1.9012962154921587
    -5.152083411295517
    2.099832470888962
    -2.097052687967737
    0.2230599135817925
    -3.5749887762684063
    -3.687034056152998
]

##

simulation_time_step = 0.001
end_time = 3.0

N = Int(floor(end_time/simulation_time_step))
X = [zeros(length(x_ref)) for _ = 1:N];
U = [zeros(length(u_ref)) for _ = 1:N];
X[1] = deepcopy(x_ref);
# X[1][nadia.nq + 5] = 1.3; # Perturb i.c.

# Run simulation
for k = 1:N - 1
    # Get error
    global Δx̃ = [qtorp(L(x_ref[1:4])'*X[k][1:4]); X[k][5:end] - x_ref[5:end]]
  
    # add some noise
    Δx̃ += 0.1 * randn(length(Δx̃))

    # Compute controls for this time step
    global U[k] = u_ref - Kinf*Δx̃

    # Integrate
    global X[k + 1] = rk4(nadia, X[k], U[k], simulation_time_step)
end
anim = animate(nadia, mvis, X; Δt=simulation_time_step, frames_to_skip=50);
setanimation!(mvis, anim)

##

Kinf_str = ""
for gain in Kinf
    Kinf_str *= string(gain) * ", "
end

clipboard(Kinf_str)