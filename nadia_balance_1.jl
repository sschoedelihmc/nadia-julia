using Pkg; Pkg.activate(@__DIR__)

using JLD2
using ForwardDiff

using MeshCatMechanisms
using RigidBodyDynamics
using RigidBodyDynamics.OdeIntegrators
using RigidBodyDynamics.PDControl

using MeshCat
using GeometryBasics
using ColorTypes
using CoordinateTransformations

using StaticArrays
using SparseArrays
using Rotations
using LinearAlgebra
import ForwardDiff
import FiniteDiff
include("data.jl")
include("quaternions.jl")

##

vis = Visualizer()
render(vis)

##

delete!(vis)

##

# Create robot and fix right foot to world
urdf = joinpath(@__DIR__, "nadia_V17_description/urdf/nadiaV17.fullRobot.simpleKnees.cycloidArms.urdf");
robot = parse_urdf(urdf, floating=true, remove_fixed_tree_joints=true)

right_foot = findbody(robot, "RIGHT_FOOT_LINK")
world = findbody(robot, "world")
right_foot_fixed_joint = Joint("right_foot_fixed_joint", Fixed{Float64}())
foot_translation = SA[-0.004351222979328111, -0.13006285479389704, 0.08180975114086993]

world_to_joint = Transform3D(
    frame_before(right_foot_fixed_joint),
    default_frame(world),
    foot_translation
)

attach!(robot, world, right_foot, right_foot_fixed_joint, joint_pose = world_to_joint)
# remove_joint!(robot, findjoint(robot, "PELVIS_LINK_to_world"))

mvis = MechanismVisualizer(robot, URDFVisuals(urdf), vis)

##

# Set initial state
function initialize!(state::MechanismState, robot::Mechanism)
    zero!(state)
    for side in ("LEFT_", "RIGHT_")
        set_configuration!(state, findjoint(robot, "$(side)HIP_Z"), HOME_CONFIG["$(side)HIP_Z"])
        set_configuration!(state, findjoint(robot, "$(side)HIP_X"), HOME_CONFIG["$(side)HIP_X"])
        set_configuration!(state, findjoint(robot, "$(side)HIP_Y"), HOME_CONFIG["$(side)HIP_Y"])
        set_configuration!(state, findjoint(robot, "$(side)KNEE_Y"), HOME_CONFIG["$(side)KNEE_Y"])
        set_configuration!(state, findjoint(robot, "$(side)ANKLE_Y"), HOME_CONFIG["$(side)ANKLE_Y"])
        set_configuration!(state, findjoint(robot, "$(side)ANKLE_X"), HOME_CONFIG["$(side)ANKLE_X"])

        set_configuration!(state, findjoint(robot, "$(side)SHOULDER_Y"), HOME_CONFIG["$(side)SHOULDER_Y"])
        set_configuration!(state, findjoint(robot, "$(side)SHOULDER_X"), HOME_CONFIG["$(side)SHOULDER_X"])
        set_configuration!(state, findjoint(robot, "$(side)SHOULDER_Z"), HOME_CONFIG["$(side)SHOULDER_Z"])
        set_configuration!(state, findjoint(robot, "$(side)ELBOW_Y"), HOME_CONFIG["$(side)ELBOW_Y"])
    end
    set_configuration!(state, findjoint(robot, "SPINE_Z"), HOME_CONFIG["SPINE_Z"])
    set_configuration!(state, findjoint(robot, "SPINE_X"), HOME_CONFIG["SPINE_X"])
    set_configuration!(state, findjoint(robot, "SPINE_Y"), HOME_CONFIG["SPINE_Y"])
    set_configuration!(state, joint_to_parent(findbody(robot, "PELVIS_LINK"), robot), HOME_CONFIG["PELVIS_LINK"])
    nothing
end

state = MechanismState(robot)
initialize!(state, robot) # Set robot mechanism state config
set_configuration!(mvis, configuration(state)) # Update config in visualizer

##

# Load balanced reference
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

x_ref = [q_ref; zeros(length(state.v))]

##

set_configuration!(mvis, q_ref)

##

# Linearize dynamics about equilibrium
result = DynamicsResult(robot);

function dynamics(x, torques)
    set_configuration!(state, x[1:length(state.q)])
    set_velocity!(state, x[length(state.q)+1:end])
    dynamics!(result, state, [zeros(6); torques]; )
    return [result.q̇; result.v̇]
end

function rk4(x, torques, dt)
    k1 = dt * dynamics(x, torques)
    k2 = dt * dynamics(x + k1/2, torques)
    k3 = dt * dynamics(x + k2/2, torques)
    k4 = dt * dynamics(x + k3, torques)
    return x + (k1 + 2*k2 + 2*k3 + k4)/6
end

dt = 1e-3
ADyn = FiniteDiff.finite_difference_jacobian(x_ -> rk4(x_, u_ref, dt), x_ref)
BDyn = FiniteDiff.finite_difference_jacobian(u_ -> rk4(x_ref, u_, dt), u_ref)

AReducedDyn = E(x_ref[1:4])' * ADyn * E(x_ref[1:4])
BReducedDyn = E(x_ref[1:4])' * BDyn

save_object("state_transition_matrix_A_reduced_form.jld2", AReducedDyn)
save_object("input_matrix_B_reduced_form.jld2", BReducedDyn)

##

# Compute optimal feedback gains
nx = size(BReducedDyn)[1]
nu = size(BReducedDyn)[2]
Q = spdiagm([repeat([5e2], 6); repeat([1e-3, 1e-3, 1e3], 3); 1e2; 1e2; repeat([5e-4; 5e-4; 1e1; 1e-5], 2); repeat([1e-2], 4);    repeat([1e1], 6); repeat([1e1, 1e1, 1e1], 3); 1e2; 1e2; repeat([1e1; 1e1; 1e1; 1e-4], 2); repeat([1e1], 4)]);
# Q = spdiagm(repeat( [repeat([1e2], 3); 1e2; 1e2; 1e2; repeat([1e1, 1e1, 1e3], 3); 1e2; 1e2; repeat([1e-2; 1e-2; 1e1; 1e-4], 2); repeat([5e-2], 4)], 2) );
R = spdiagm(1e-2*ones(nu));

Kinf = zeros(nu,nx)
Pinf = zeros(nx,nx)
Kprev = zeros(nu,nx)
Pprev = Q

# Compute Kinf, Pinf
riccati_iters = 0
riccati_err = 1e-10
A = AReducedDyn
B = BReducedDyn
for i in 1:50000 # 1:max_iters
    Kinf = (R + B'*Pprev*B)\(B'*Pprev*A);
    Pinf = Q + A'*Pprev*(A - B*Kinf);
    if maximum(abs.(Kinf - Kprev)) < riccati_err
        display("IHLQR converged in " * string(i) * " iterations")
        break
    end
    Kprev = Kinf
    Pprev = Pinf
    riccati_iters += 1
end

eigvals(A-B*Kinf)

##

end_time = 3.0
time_step = 0.0001

x_ref_pelvis_z = [sin(t/12000) for t in 1:floor((end_time*4)/time_step)+5] .- 0.05
x_ref_new = deepcopy(x_ref)

# Create LQR controller
index = 1
function lqr_controller!(torques::AbstractVector, t, current_state::MechanismState)
    global index
    current_x = [current_state.q[1:end]; current_state.v[1:end]] # [1:end] to extract Vector{Float64} from segmented vector
    Δx̃ = [qtorp(L(x_ref[1:4])'*current_x[1:4]); current_x[5:end] - x_ref[5:end]]
    # x_ref_new[7] = x_ref[7] + x_ref_pelvis_z[index]
    # index += 1
    # Δx̃ = [qtorp(L(x_ref[1:4])'*current_x[1:4]); current_x[5:end] - x_ref_new[5:end]]
    Δu = -Kinf * Δx̃
    torques[1:end] = [zeros(6); Δu + u_ref]
end

state = MechanismState(robot)
initialize!(state, robot) # Set robot mechanism state config
set_configuration!(mvis, configuration(state)) # Update config in visualizer

state.v[5] = -1.5 # X-velocity of pelvis

baumgarte_gains = Dict(JointID(right_foot_fixed_joint) => SE3PDGains(PDGains(3000.0, 200.0), PDGains(3000.0, 200.0))) # angular, linear
t, q, v = simulate(state, end_time, lqr_controller!; Δt=time_step, stabilization_gains=baumgarte_gains);
animation = Animation(mvis, t, q);
setanimation!(mvis, animation);

##


function animate!(mvis::MechanismVisualizer, qs; Δt=0.001, division=10)
    anim = MeshCat.Animation(convert(Int, floor(1.0 / (Δt * division))))
    for (t, q) in enumerate(qs[1:division:end])
        MeshCat.atframe(anim, t) do 
            set_configuration!(mvis, q[1:length(state.q)])
            height = x_ref[7] + x_ref_pelvis_z[(t-1)*1000*4 + 1]
            settransform!(vis[:ball], Translation(0, 1.0, height))
        end
    end
    MeshCat.setanimation!(mvis, anim)
    return anim
end

setobject!(vis[:ball], Sphere(Point(0.0, 0.0, 0.0), 0.07), MeshPhongMaterial(color=RGBA(.7, 1, .6, 1)))

anim = animate!(mvis, q, Δt=time_step, division=1000);
setanimation!(mvis, anim)

##

# Check error
for i in 1:length(t)
    println(maximum([qtorp(L(x_ref[1:4])'*q[i][1:4]); [q[i][5:end]; v[i]] - x_ref[5:end]]))
end

##

MeshCat.convert_frames_to_video("/home/sschoedel/Videos/Weeks 1-2/meshcat_nadia_LQR_balance_4.tar")

##

clipboard(Kinf)

str = ""
for (i, gain) in enumerate(Kinf)
    str *= string(gain)
    if i < length(Kinf)
        str *= ", "
    end
    if i % size(Kinf)[1] == 0
        str *= "\n"
    end
end

clipboard(str)