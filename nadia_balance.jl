using Pkg; Pkg.activate(@__DIR__)

using MeshCatMechanisms
using RigidBodyDynamics
using RigidBodyDynamics.PDControl

using MeshCat

using StaticArrays
using SparseArrays

import ForwardDiff

include("nadia_robot.jl")
include("control_utils.jl")
include("quaternions.jl")

##

nadia = Nadia();
vis = Visualizer();
render(vis)

##

# Initialize visualizer with robot meshes
mvis = init_visualizer(nadia, vis)

##

# Load equilibrium reference controls and state
u_ref =
[
-1.529996146733067,     # "LEFT_HIP_Z"
9.964441654400309,      # "RIGHT_HIP_Z"
2.4214516921032456,     # "SPINE_Z"
30.725284382117,        # "LEFT_HIP_X"
-102.44994521205273,    # "RIGHT_HIP_X"
-10.165094068114897,    # "SPINE_X"
1.3888702260931227,     # "LEFT_HIP_Y"
30.57276337319791,      # "RIGHT_HIP_Y"
-5.786704258930225,     # "SPINE_Y"
18.379170992769396,     # "LEFT_KNEE_Y"
-87.7703566391235,      # "RIGHT_KNEE_Y"
2.673614241444654,      # "LEFT_SHOULDER_Y"
3.90078753582609,       # "RIGHT_SHOULDER_Y"
0.6569298958706897,     # "LEFT_ANKLE_Y"
31.358889860182014,     # "RIGHT_ANKLE_Y"
7.695142597693708,      # "LEFT_SHOULDER_X"
-4.036609656052576,     # "RIGHT_SHOULDER_X"
0.22082314816501622,    # "LEFT_ANKLE_X"
-0.0021063087760480495, # "RIGHT_ANKLE_X"
0.4149513856694272,     # "LEFT_SHOULDER_Z"
-0.2869521910778749,    # "RIGHT_SHOULDER_Z"
-0.532071050841635,     # "LEFT_ELBOW_Y"
-0.5439629977475727     # "RIGHT_ELBOW_Y"
]

q_ref = 
[0.9851812850604109, 0.14091286404085285, -0.021356269245195184, -0.095421748536162, # pelvis orientation
-0.04426391998996947, -0.1668505364459041, 1.0334898178372895,     # pelvis position
0.06781608404733115, # LEFT_HIP_Z
-0.1922539118626478, # RIGHT_HIP_Z
0.2861350253811403, # SPINE_Z
0.15898265237693734, # LEFT_HIP_X
-0.12064158590993693, # RIGHT_HIP_X
-0.1057141650799965, # SPINE_X
-0.4044177896371911, # LEFT_HIP_Y
-0.3535655955195078, # RIGHT_HIP_Y
0.07352471380726427, # SPINE_Y
1.500022076915403, # LEFT_KNEE_Y
0.7689918571683205, # RIGHT_KNEE_Y
0.2271611304501065, # LEFT_SHOULDER_Y
0.6387021266726293, # RIGHT_SHOULDER_Y
0.22893847881329563, # LEFT_ANKLE_Y
-0.3460980826665025, # RIGHT_ANKLE_Y
0.35167329835530214, # LEFT_SHOULDER_X
-0.2566674753190524, # RIGHT_SHOULDER_X
-0.43626403944917924, # LEFT_ANKLE_X
-0.15679667582212356, # RIGHT_ANKLE_X
0.7430339207647649, # LEFT_SHOULDER_Z
-0.608843261553189, # RIGHT_SHOULDER_Z
-1.2994261459930767, # LEFT_ELBOW_Y
-1.475040588733499] # RIGHT_ELBOW_Y

x_ref = [q_ref; zeros(nadia.nv)]

##

# Set pose in visualizer
visualize!(nadia, mvis, q_ref)

##

# Linearize dynamics about equilibrium

dt = 1e-3
ADyn = ForwardDiff.jacobian(x_->rk4(nadia, x_, u_ref, dt), x_ref);
BDyn = ForwardDiff.jacobian(u_->rk4(nadia, x_ref, u_, dt), u_ref);

ADynReduced = E(x_ref[1:4])' * ADyn * E(x_ref[1:4])
BDynReduced = E(x_ref[1:4])' * BDyn

##

# Compute IHLQR optimal feedback gain matrix Kinf

# Tune these if you want
Q = spdiagm([repeat([5e2], 6); repeat([1e-3, 1e-3, 1e3], 3); 1e2; 1e2; repeat([5e-4; 5e-4; 1e1; 1e-5], 2); repeat([1e-2], 4);    repeat([1e1], 6); repeat([1e1, 1e1, 1e1], 3); 1e2; 1e2; repeat([1e1; 1e1; 1e1; 1e-4], 2); repeat([1e1], 4)]);
R = spdiagm(1e-2*ones(size(BDynReduced)[2]));

Kinf, Qf = ihlqr(ADynReduced, BDynReduced, Q, R, Q, max_iters = 200000);

# Check eigenvalues of system
eigvals(ADynReduced - BDynReduced*Kinf)

##

# # Create LQR controller
# function lqr_controller!(torques::AbstractVector, t, current_state::MechanismState)
#     global index
#     current_x = [current_state.q[1:end]; current_state.v[1:end]] # [1:end] to extract Vector{Float64} from segmented vector
#     Δx̃ = [qtorp(L(x_ref[1:4])'*current_x[1:4]); current_x[5:end] - x_ref[5:end]]
    # Δu = -Kinf * Δx̃
#     torques[1:end] = [zeros(6); Δu + u_ref]
# end

# # state.v[5] = -1.5 # X-velocity of pelvis

# baumgarte_gains = Dict(JointID(right_foot_fixed_joint) => SE3PDGains(PDGains(3000.0, 200.0), PDGains(3000.0, 200.0))) # angular, linear
# t, q, v = simulate(state, end_time, lqr_controller!; stabilization_gains=baumgarte_gains);
# animation = Animation(mvis, t, q);
# setanimation!(mvis, animation);


simulation_time_step = 0.01
end_time = 3.0

N = Int(floor(end_time/simulation_time_step))
X = [zeros(length(x_ref)) for _ = 1:N];
U = [zeros(length(u_ref)) for _ = 1:N];
X[1] = deepcopy(x_ref);
# X[1][length(q_ref) + 5] = 1.3; # Perturb i.c.

# Run simulation
for k = 1:N - 1
    # Get error
    global Δx̃ = [qtorp(L(x_ref[1:4])'*X[k][1:4]); X[k][5:end] - x_ref[5:end]]
  
    # Compute controls for this time step
    global U[k] = u_ref - Kinf*Δx̃

    # Integrate
    global X[k + 1] = rk4(nadia, X[k], U[k], simulation_time_step)
end
anim = animate(nadia, mvis, X, Δt=simulation_time_step, division=1);
setanimation!(mvis, anim)