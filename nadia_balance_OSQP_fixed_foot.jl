using Pkg; Pkg.activate(@__DIR__)

using JLD2
using Plots
using ReLUQP

using MeshCat
using MeshCatMechanisms
using RigidBodyDynamics
using RigidBodyDynamics.PDControl

using StaticArrays
using SparseArrays

using GeometryBasics: Sphere, Vec, Point, Mesh

using ForwardDiff

include("nadia_robot_fixed_foot.jl")
include("control_utils.jl")

##

urdfpath = joinpath(@__DIR__, "nadia_V17_description/urdf/nadiaV17.fullRobot.simpleKnees.extended.cycloidArms.urdf");
nadia = NadiaFixed(urdfpath)
vis = Visualizer()
render(vis)

##

# Initialize visualizer with robot meshes
mvis = init_visualizer(nadia, vis)

##

# Load equilibrium reference controls and state
u_ref = load_object("nadia_balance_u_ref_4_central_foot.jld2")
x_ref = load_object("nadia_balance_x_ref_4_central_foot.jld2")

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

# Set stage costs for state and control input
Q = spdiagm([
        # Pelvis orientation, pelvis translation
        repeat([5e2], 3); repeat([5e3], 3);
        # LEFT_HIP_Z, RIGHT_HIP_Z, SPINE_Z
        100; 100; 1000;
        # LEFT_HIP_X RIGHT_HIP_X SPINE_X
        70; 500; 10000;
        # LEFT_HIP_Y RIGHT_HIP_Y SPINE_Y
        0.5; 5; 1000;
        # LEFT_KNEE_Y RIGHT_KNEE_Y
        50; 1;
        # LEFT_SHOULDER_Y RIGHT_SHOULDER_Y LEFT_ANKLE_Y RIGHT_ANKLE_Y
        0.4; 0.4; 0.1; 10;
        # LEFT_SHOULDER_X RIGHT_SHOULDER_X LEFT_ANKLE_X RIGHT_ANKLE_X
        0.4; 0.4; 0.1; 0.7;
        # LEFT_SHOULDER_Z RIGHT_SHOULDER_Z
        0.4; 0.4;
        # LEFT_ELBOW_Y RIGHT_ELBOW_Y
        1.2; 1.2;

        # Pelvis orientation & translation
        repeat([10], 6);
        # LEFT_HIP_Z, RIGHT_HIP_Z, SPINE_Z
        10; 10; 10;
        # LEFT_HIP_X RIGHT_HIP_X SPINE_X
        10; 10; 10;
        # LEFT_HIP_Y RIGHT_HIP_Y SPINE_Y
        10; 10; 10;
        # LEFT_KNEE_Y RIGHT_KNEE_Y
        5; 15; 
        # LEFT_SHOULDER_Y RIGHT_SHOULDER_Y LEFT_ANKLE_Y RIGHT_ANKLE_Y
        0.3; 0.3; 10; 0.1;
        # LEFT_SHOULDER_X RIGHT_SHOULDER_X LEFT_ANKLE_X RIGHT_ANKLE_X
        0.3; 0.3; 10; 0.5;
        # LEFT_SHOULDER_Z RIGHT_SHOULDER_Z
        0.3; 0.3;
        # LEFT_ELBOW_Y RIGHT_ELBOW_Y
        0.3; 0.3;
    ])

R = spdiagm([
        # LEFT_HIP_Z RIGHT_HIP_Z SPINE_Z
        0.07; 0.07; 0.01;
        # LEFT_HIP_X RIGHT_HIP_X SPINE_X
        0.07; 0.1; 0.01;
        # LEFT_HIP_Y RIGHT_HIP_Z SPINE_Y
        0.07; 0.07; 0.01;
        # LEFT_KNEE_Y RIGHT_KNEE_Y
        0.01; 0.01;
        # LEFT_SHOULDER_Y RIGHT_SHOULDER_Y LEFT_ANKLE_Y RIGHT_ANKLE_Y
        0.5; 0.5; 10; 0.1;
        # LEFT_SHOULDER_X RIGHT_SHOULDER_X LEFT_ANKLE_X RIGHT_ANKLE_X
        0.5; 0.5; 10; 0.5;
        # LEFT_SHOULDER_Z RIGHT_SHOULDER_Z
        0.5; 0.5;
        # LEFT_ELBOW_Y RIGHT_ELBOW_Y
        0.5; 0.5
    ])

Kinf, Qf = ihlqr(ADynReduced, BDynReduced, Q, R, Q; max_iters = 200000, verbose=true);

##

# Define additional constraints for the QP
horizon = 2;
A_torque = kron(I(horizon), [I(nadia.nu) zeros(nadia.nu, nadia.nq+nadia.nv)]);
l_torque = repeat(-nadia.torque_limits-u_ref, horizon);
u_torque = repeat(nadia.torque_limits-u_ref, horizon);

# Setup QP
H, g, A, l, u, g_x0, lu_x0 = gen_condensed_mpc_qp(Ad, Bd, Q, R, Qf, horizon, A_torque, l_torque, u_torque, K);

##

simulation_time_step = 0.001
end_time = 3.0

N = Int(floor(end_time/simulation_time_step))
X = [zeros(length(x_ref)) for _ = 1:N+1];
U = [zeros(length(u_ref)) for _ = 1:N];
X[1] = deepcopy(x_ref);
X[1][nadia.nq + 5] = 0.05; # Perturb i.c.

# Run simulation
for k = 1:N
    # Get error
    global Δx̃ = [qtorp(L(x_ref[1:4])'*X[k][1:4]); X[k][5:end] - x_ref[5:end]]
  
    # add some noise
    # Δx̃ += 0.01 * randn(length(Δx̃))

    # Compute controls for this time step
    global U[k] = u_ref - Kinf*Δx̃

    # Integrate
    global X[k + 1] = rk4(nadia, X[k], U[k], simulation_time_step; gains=nadia.baumgarte_gains)
end
anim = animate(nadia, mvis, X; Δt=simulation_time_step, frames_to_skip=50);
setanimation!(mvis, anim)