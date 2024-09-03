using Pkg; Pkg.activate(@__DIR__)

using JLD2
using Plots

using MeshCat
using MeshCatMechanisms
using RigidBodyDynamics
using RigidBodyDynamics.PDControl

using StaticArrays
using SparseArrays

using GeometryBasics: Sphere, Vec, Point, Mesh

import ForwardDiff

include("nadia_robot_fixed_foot.jl")
include("control_utils.jl")

##

# urdfpath = joinpath(@__DIR__, "nadia_V17_description/urdf/nadiaV17.fullRobot.simpleKnees.cycloidArms.urdf");
urdfpath = joinpath(@__DIR__, "nadia_V17_description/urdf/nadiaV17.fullRobot.simpleKnees.extended.cycloidArms.urdf");
# urdfpath = joinpath(@__DIR__, "nadia_V17_description/urdf/nadiaV17.fullRobot.cycloidArms4DoF.withFourBarLinkage.urdf");
nadia = NadiaFixed(urdfpath)
vis = Visualizer()
render(vis)

##

# Initialize visualizer with robot meshes
mvis = init_visualizer(nadia, vis)

##

# Load equilibrium reference controls and state
# u_ref = load_object("nadia_balance_u_ref.jld2")
# u_ref = load_object("nadia_balance_u_ref_2.jld2")
u_ref = load_object("nadia_balance_u_ref_4_central_foot.jld2")

# u_ref torques should be in the following order:
# "LEFT_HIP_Z"                  1
# "RIGHT_HIP_Z"                 2
# "SPINE_Z"                     3
# "LEFT_HIP_X"                  4
# "RIGHT_HIP_X"                 5
# "SPINE_X"                     6
# "LEFT_HIP_Y"                  7
# "RIGHT_HIP_Y"                 8
# "SPINE_Y"                     9
# "LEFT_KNEE_Y"                 10
# "RIGHT_KNEE_Y"                11
# "LEFT_SHOULDER_Y"             12
# "RIGHT_SHOULDER_Y"            13
# "LEFT_ANKLE_Y"                14
# "RIGHT_ANKLE_Y"               15
# "LEFT_SHOULDER_X"             16
# "RIGHT_SHOULDER_X"            17
# "LEFT_ANKLE_X"                18
# "RIGHT_ANKLE_X"               19
# "LEFT_SHOULDER_Z"             20
# "RIGHT_SHOULDER_Z"            21
# "LEFT_ELBOW_Y"                22
# "RIGHT_ELBOW_Y"               23


# x_ref = load_object("nadia_balance_x_ref.jld2")
# x_ref = load_object("nadia_balance_x_ref_fourbar.jld2")
# x_ref = load_object("nadia_balance_x_ref_2.jld2")
x_ref = load_object("nadia_balance_x_ref_4_central_foot.jld2")
# save_object("nadia_balance_x_ref_fourbar.jld2", x_ref)
# 4 variables for pelvis orientation (w, x, y, z quaternion)
# 3 variables for pelvis position in world frame
# LEFT_HIP_Z            8
# RIGHT_HIP_Z           9
# SPINE_Z               10
# LEFT_HIP_X            11
# RIGHT_HIP_X           12
# SPINE_X               13
# LEFT_HIP_Y            14
# RIGHT_HIP_Y           15
# SPINE_Y               16
# LEFT_KNEE_Y           17
# RIGHT_KNEE_Y          18
# LEFT_SHOULDER_Y       19
# RIGHT_SHOULDER_Y      20
# LEFT_ANKLE_Y          21
# RIGHT_ANKLE_Y         22
# LEFT_SHOULDER_X       23
# RIGHT_SHOULDER_X      24
# LEFT_ANKLE_X          25
# RIGHT_ANKLE_X         26
# LEFT_SHOULDER_Z       27
# RIGHT_SHOULDER_Z      28
# LEFT_ELBOW_Y          29
# RIGHT_ELBOW_Y         30

# PELVIS VELOCITIES     31-36
# LEFT_HIP_Z_VEL        37
# RIGHT_HIP_Z_VEL       38
# SPINE_Z_VEL           39
# LEFT_HIP_X_VEL        40
# RIGHT_HIP_X_VEL       41
# SPINE_X_VEL           42
# LEFT_HIP_Y_VEL        43
# RIGHT_HIP_Y_VEL       44
# SPINE_Y_VEL           45
# LEFT_KNEE_Y_VEL       46
# RIGHT_KNEE_Y_VEL      47
# LEFT_SHOULDER_Y_VEL   48
# RIGHT_SHOULDER_Y_VEL  49
# LEFT_ANKLE_Y_VEL      50
# RIGHT_ANKLE_Y_VEL     51
# LEFT_SHOULDER_X_VEL   52
# RIGHT_SHOULDER_X_VEL  53
# LEFT_ANKLE_X_VEL      54
# RIGHT_ANKLE_X_VEL     55
# LEFT_SHOULDER_Z_VEL   56
# RIGHT_SHOULDER_Z_VEL  57
# LEFT_ELBOW_Y_VEL      58
# RIGHT_ELBOW_Y_VEL     59
# 29 variables for reference velocities in the same order 
#   as config but with three variables for orientation 
#   (generally all zeros for an equilibrium position)


# Set pose in visualizer
# x_ref[11] += 0.1
set_configuration!(mvis, x_ref[1:nadia.nq])

##


# x_bar = deepcopy(x_ref)
# for i in 1:90
#     angle = i*pi/180
#     x_bar[17] = angle
#     x_bar_fourbar = convert_simple_to_four_bar_state(x_bar)
#     set_configuration!(mvis, x_bar_fourbar[1:nadia.nq])
#     sleep(0.1)
# end

# x_test = convert_simple_to_four_bar_state(x_ref)
# x_test[17] = 0.0 # left shell upper
# x_test[18] = 0.0 # left shell linkage upper
# x_test[19] = 0.0 # right shell upper
# x_test[20] = 0.0 # right shell linkage upper
# x_test[22] = 0.0 # left knee shell lower
# x_test[23] = 0.0 # right knee shell lower
# set_configuration!(mvis, x_test[1:nadia.nq])
# u_test = convert_simple_to_four_bar_control_input(u_ref)

##

# a = [-0.02113091, 0.0, -0.33468461]
# b = [0.0, 0.0, -0.38]
# ab = SA[-0.02113091, 0.0, 0.04531539000000001]


# for i in 1:90
#     angle = i*pi/180
#     x_bar = deepcopy(x_ref)
#     x_bar[17] = angle
#     x_test = simple_to_four_bar(x_bar)
#     set_configuration!(mvis, x_test[1:nadia.nq])
#     sleep(0.01)
# end

##

# Calculate linearized dynamics
dt = 1e-3
# ADyn = ForwardDiff.jacobian(x_ -> rk4_simple_to_four_bar(nadia, x_, u_ref, dt), x_ref)
# BDyn = ForwardDiff.jacobian(u_ -> rk4_simple_to_four_bar(nadia, x_ref, u_, dt), u_ref)
ADyn = ForwardDiff.jacobian(x_ -> rk4(nadia, x_, u_ref, dt), x_ref)
BDyn = ForwardDiff.jacobian(u_ -> rk4(nadia, x_ref, u_, dt), u_ref)

# Reduce quaternion representation to a form we can do math with
ADynReduced = E(x_ref[1:4])' * ADyn * E(x_ref[1:4])
BDynReduced = E(x_ref[1:4])' * BDyn

##

# Compute IHLQR optimal feedback gain matrix Kinf

# # Tune these if you want
# These work for the simple knees model with the x_ref and u_ref in nadia_balance_x/u_ref.jld2
# Q = spdiagm([repeat([5e2], 3); repeat([5e3], 3); 1e2; 1e2; 1e3; repeat([1e-1, 1e-1, 1e3], 2); 1; 1; repeat([2e1; 2e1; 1e-4; 1e-4], 2); 1.5*ones(2); 1.5*ones(2);
#                 repeat([1e1], 6); repeat([1e1, 1e1, 1e1], 3); 1.5e1; 1.5e1; repeat([1e1; 1e1; 1e1; 1e1], 2); 1e-1*ones(2); 1e-1*ones(2)])
# R = spdiagm([1e-2*ones(11); 1e1*ones(2); repeat([1e4; 1e4; 1e1; 1e1], 2); 1*ones(2)])

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

# Check eigenvalues of system
eigvals(ADynReduced-BDynReduced*Kinf)

Kinf_str = ""
for gain in Kinf
    Kinf_str *= string(gain) * ", "
end

clipboard(Kinf_str)

maximum(abs.(Kinf))

##

setobject!(vis, Sphere(Point{3, Float64}([-0.004351222979328111, -0.12106285479389704, 0.08180975114086993]), .03))

##

simulation_time_step = 0.001
end_time = 3.0

N = Int(floor(end_time/simulation_time_step))
X = [zeros(length(x_ref)) for _ = 1:N+1];
U = [zeros(length(u_ref)) for _ = 1:N];
X[1] = deepcopy(x_ref);
# X[1][nadia.nq + 5] = 0.1; # Perturb i.c.


# # left_knee_linkage = findbody(nadia.mech, "LEFT_KNEE_LINKAGE_LINK")
# # left_shin = findbody(nadia.mech, "LEFT_SHIN_LINK")
# # left_knee_linkage_joint = Joint("left_knee_linkage_joint", Revolute{Float64}([-0.01711428, 0.0, -0.0883578]))
# # attach!(nadia.mech, left_knee_linkage, left_shin, left_knee_linkage_joint)

# X = [zeros(length(x_ref)+8) for _ = 1:N+1];
# U = [zeros(length(u_ref)+4) for _ = 1:N];
# X[1] = deepcopy(convert_simple_to_four_bar_state(x_ref));
# # X[1][nadia.nq + 5] = 0.1; # Perturb i.c.

# Run simulation
for k = 1:N
    # Get error
    global Δx̃ = [qtorp(L(x_ref[1:4])'*X[k][1:4]); X[k][5:end] - x_ref[5:end]]
  
    # add some noise
    Δx̃ += 0.01 * randn(length(Δx̃))

    # Compute controls for this time step
    global U[k] = u_ref - Kinf*Δx̃
    # global U[k] = u_ref
    # global U[k] = zeros(length(u_ref))

    # Integrate
    global X[k + 1] = rk4(nadia, X[k], U[k], simulation_time_step; gains=nadia.baumgarte_gains)
end
anim = animate(nadia, mvis, X; Δt=simulation_time_step, frames_to_skip=50);
setanimation!(mvis, anim)

##

plot((hcat(U... ) .- u_ref)'[:,4])
# plot(hcat(X...)')

include("scsdata.jl")

taus = zeros(23)
for (jointName, index) in TORQUES_ORDER
    taus[index] = namedTaus[jointName]
end

deltaState = zeros(58)
for (jointName, index) in REDUCED_CONFIG_ORDER
    deltaState[index] = namedDeltaConfigurations[jointName]
end
for (jointName, index) in REDUCED_VELOCITY_ORDER
    deltaState[index] = namedDeltaVelocities[jointName]
end

deltaTorques = zeros(23)
for (jointName, index) in TORQUES_ORDER
    deltaTorques[index] = namedDeltaTorques[jointName]
end

torques = zeros(23)
for (jointName, index) in TORQUES_ORDER
    torques[index] = namedTorques[jointName]
end

clampedTorques = zeros(23)
for (jointName, index) in TORQUES_ORDER
    clampedTorques[index] = namedClampedTorques[jointName]
end
