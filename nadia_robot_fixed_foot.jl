# Setup the dynamics and visualization for the Nadia model
using MeshCat
using MeshCatMechanisms
using RigidBodyDynamics
using RigidBodyDynamics.PDControl
using StaticArrays

include("control_utils.jl")

struct NadiaFixed
    mech::Mechanism{Float64}
    state::MechanismState
    dyn_result::DynamicsResult
    statecache::StateCache
    dyn_result_cache::DynamicsResultCache
    baumgarte_gains
    urdfpath::String
    nq::Int
    nv::Int
    fourbarknee::Bool
    function NadiaFixed(urdfpath; fourbarknee=false)

        # Create robot and fix right foot to world (without deleting pelvis)
        mech = parse_urdf(urdfpath, floating=true, remove_fixed_tree_joints=true) 

        right_foot = findbody(mech, "RIGHT_FOOT_LINK")
        world = findbody(mech, "world")
        right_foot_fixed_joint = Joint("right_foot_fixed_joint", Fixed{Float64}())
        # foot_translation = SA[-0.004351222979328111, -0.13006285479389704, 0.08180975114086993]
        foot_translation = SA[-0.002351222979328111, -0.12106285479389704, 0.08180975114086993]

        world_to_joint = Transform3D(
            frame_before(right_foot_fixed_joint),
            default_frame(world),
            foot_translation
        )

        attach!(mech, world, right_foot, right_foot_fixed_joint, joint_pose=world_to_joint)

        # left_knee_linkage = findbody(mech, "LEFT_KNEE_LINKAGE_LINK")
        # left_shin = findbody(mech, "LEFT_SHIN_LINK")
        # left_knee_linkage_joint = Joint("left_knee_linkage_joint", Revolute{Float64}([0.0, 1.0, 0.0]))
        # left_knee_linkage_translation = SA[-0.01711428, 0.0, -0.0883578]
        # left_linkage_to_shin_joint = Transform3D(
        #     frame_before(left_knee_linkage_joint),
        #     default_frame(left_knee_linkage),
        #     left_knee_linkage_translation
        # )
        # left_shin_linkage_translation = SA[0.03830222, 0.0, 0.03213938]
        # left_shin_to_linkage_joint = Transform3D(
        #     default_frame(left_shin),
        #     frame_after(left_knee_linkage_joint),
        #     left_shin_linkage_translation
        # )
        # attach!(mech, left_knee_linkage, left_shin, left_knee_linkage_joint,
        #     joint_pose=left_linkage_to_shin_joint,
        #     successor_pose=left_shin_to_linkage_joint
        # )


        # right_knee_linkage = findbody(mech, "RIGHT_KNEE_LINKAGE_LINK")
        # right_shin = findbody(mech, "RIGHT_SHIN_LINK")
        # right_knee_linkage_joint = Joint("right_knee_linkage_joint", Revolute{Float64}([0.0, 1.0, 0.0]))
        # right_knee_linkage_translation = SA[-0.01711428, 0.0, -0.0883578]
        # right_linkage_to_shin_joint = Transform3D(
        #     frame_before(right_knee_linkage_joint),
        #     default_frame(right_knee_linkage),
        #     right_knee_linkage_translation
        # )
        # # right_shin_linkage_translation = SA[-0.03830222, 0.0, -0.03213938]
        # right_shin_linkage_translation = SA[0.03830222, 0.0, 0.03213938]
        # right_shin_to_linkage_joint = Transform3D(
        #     default_frame(right_shin),
        #     frame_after(right_knee_linkage_joint),
        #     right_shin_linkage_translation
        # )
        # attach!(mech, right_knee_linkage, right_shin, right_knee_linkage_joint,
        #     joint_pose=right_linkage_to_shin_joint,
        #     successor_pose=right_shin_to_linkage_joint
        # )



        # # Stabilization gains for non-tree joints
        # baumgarte_gains = Dict(
        #     JointID(right_foot_fixed_joint) => SE3PDGains(PDGains(3000.0, 200.0), PDGains(3000.0, 200.0)), # angular, linear
        #     JointID(left_knee_linkage_joint) => SE3PDGains(PDGains(3000.0, 200.0), PDGains(3000.0, 200.0)),
        #     JointID(right_knee_linkage_joint) => SE3PDGains(PDGains(3000.0, 200.0), PDGains(3000.0, 200.0))
        # )

        # Stabilization gains for non-tree joints
        baumgarte_gains = Dict(
            JointID(right_foot_fixed_joint) => SE3PDGains(PDGains(3000.0, 200.0), PDGains(3000.0, 200.0)) # angular, linear
        )

        new(mech, MechanismState(mech), DynamicsResult(mech), StateCache(mech), DynamicsResultCache(mech), baumgarte_gains, urdfpath, num_positions(mech), num_velocities(mech), fourbarknee)
    end
end

function dynamics(model::NadiaFixed, x::AbstractVector{T1}, u::AbstractVector{T2}; gains=RigidBodyDynamics.default_constraint_stabilization_gains(Float64)) where {T1, T2}
    T = promote_type(T1, T2)
    state = model.statecache[T]
    dyn_result = model.dyn_result_cache[T]

    # x = [rptoq(x[1:3]); x[4:end]]

    x[1:4] /= norm(x[1:4]) # normalize quaternion

    # Set the mechanism state
    copyto!(state, x)

    # Perform forward dynamics (six zeros because RigidBodyDynamics allows control over the pelvis)
    dynamics!(dyn_result, state, [zeros(6); u]; stabilization_gains=gains)

    return [dyn_result.q̇; dyn_result.v̇]
end

function rk4_simple_to_four_bar(model::NadiaFixed, x_, u_, h; gains=RigidBodyDynamics.default_constraint_stabilization_gains(Float64))
    x = simple_to_four_bar(x_)
    u = simple_to_four_bar_control(u_)
    k1 = dynamics(model, x, u; gains=gains)
    k2 = dynamics(model, x + h/2*k1, u; gains=gains)
    k3 = dynamics(model, x + h/2*k2, u; gains=gains)
    k4 = dynamics(model, x + h*k3, u; gains=gains)
    x_next = x + h/6*(k1 + 2*k2 + 2*k3 + k4)
    return four_bar_to_simple(x_next)
end

function rk4(model::NadiaFixed, x, u, h; gains=RigidBodyDynamics.default_constraint_stabilization_gains(Float64))
    k1 = dynamics(model, x, u; gains=gains)
    k2 = dynamics(model, x + h/2*k1, u; gains=gains)
    k3 = dynamics(model, x + h/2*k2, u; gains=gains)
    k4 = dynamics(model, x + h*k3, u; gains=gains)
    return x + h/6*(k1 + 2*k2 + 2*k3 + k4)
end

function init_visualizer(model::NadiaFixed, vis::Visualizer)
    delete!(vis)
    mvis = MechanismVisualizer(model.mech, URDFVisuals(model.urdfpath), vis)
    return mvis
end


function animate(model::NadiaFixed, mvis::MechanismVisualizer, qs; Δt=0.001, frames_to_skip=50)
    anim = MeshCat.Animation(convert(Int, floor(1.0 / (Δt * frames_to_skip))))
    for (t, q) in enumerate(qs[1:frames_to_skip:end])
        MeshCat.atframe(anim, t) do 
            set_configuration!(mvis, q[1:model.nq])
        end
    end
    MeshCat.setanimation!(mvis, anim)

    return anim
end



function convert_simple_to_four_bar_state(x_no_four_bar)
    left_knee_angle = x_no_four_bar[17]
    right_knee_angle = x_no_four_bar[18]
    left_knee_shell_upper_y_angle = left_knee_angle/2.0
    left_knee_linkage_upper_y_angle = left_knee_angle/2.0
    left_knee_shell_lower_y_angle = left_knee_angle/2.0
    right_knee_shell_upper_y_angle = right_knee_angle/2.0
    right_knee_linkage_upper_y_angle = right_knee_angle/2.0
    right_knee_shell_lower_y_angle = right_knee_angle/2.0
    left_knee_vel = x_no_four_bar[46]
    right_knee_vel = x_no_four_bar[47]
    left_knee_shell_upper_y_vel = left_knee_vel/2.0
    left_knee_linkage_upper_y_vel = left_knee_vel/2.0
    left_knee_shell_lower_y_vel = left_knee_vel/2.0
    right_knee_shell_upper_y_vel = right_knee_vel/2.0
    right_knee_linkage_upper_y_vel = right_knee_vel/2.0
    right_knee_shell_lower_y_vel = right_knee_vel/2.0
    x_four_bar_config = [
        x_no_four_bar[1:16];
        left_knee_shell_upper_y_angle;
        left_knee_linkage_upper_y_angle;
        right_knee_shell_upper_y_angle;
        right_knee_linkage_upper_y_angle;
        x_no_four_bar[19:20];
        left_knee_shell_lower_y_angle;
        right_knee_shell_lower_y_angle;
        x_no_four_bar[23:24];
        x_no_four_bar[21:22];
        x_no_four_bar[27:28];
        x_no_four_bar[25:26];
        x_no_four_bar[29:30];
    ]
    x_four_bar_velocity = [
        x_no_four_bar[31:45];
        left_knee_shell_upper_y_vel;
        left_knee_linkage_upper_y_vel;
        right_knee_shell_upper_y_vel;
        right_knee_linkage_upper_y_vel;
        x_no_four_bar[48:49];
        left_knee_shell_lower_y_vel;
        right_knee_shell_lower_y_vel;
        x_no_four_bar[52:53];
        x_no_four_bar[50:51];
        x_no_four_bar[56:57];
        x_no_four_bar[54:55];
        x_no_four_bar[58:59]
    ]
    return [x_four_bar_config; x_four_bar_velocity]
end



function convert_four_bar_state_to_simple(x_four_bar)
    left_knee_shell_upper_y_angle = x_four_bar[17]
    right_knee_shell_upper_y_angle = x_four_bar[19]
    left_knee_angle = left_knee_shell_upper_y_angle*2.0
    right_knee_angle = right_knee_shell_upper_y_angle*2.0
    left_knee_shell_upper_y_vel = x_four_bar[50]
    right_knee_shell_upper_y_vel = x_four_bar[52]
    left_knee_vel = left_knee_shell_upper_y_vel*2.0
    right_knee_vel = right_knee_shell_upper_y_vel*2.0
    x_config = [
        x_four_bar[1:16];
        left_knee_angle;
        right_knee_angle;
        x_four_bar[21:22];
        x_four_bar[27:28];
        x_four_bar[25:26];
        x_four_bar[31:32];
        x_four_bar[29:30];
        x_four_bar[33:34];
    ]
    x_velocity = [
        x_four_bar[35:49];
        left_knee_vel;
        right_knee_vel;
        x_four_bar[54:55];
        x_four_bar[60:61];
        x_four_bar[58:59];
        x_four_bar[64:65];
        x_four_bar[62:63];
        x_four_bar[66:67]
    ]
    return [x_config; x_velocity]
end



function simple_to_four_bar_control(u_no_four_bar)
    left_knee_y_tau = u_no_four_bar[10]
    right_knee_y_tau = u_no_four_bar[11]
    left_knee_linkage_upper_y_tau = left_knee_y_tau
    right_knee_linkage_upper_y_tau = right_knee_y_tau
    u_four_bar = [
        u_no_four_bar[1:9];
        0.0;
        left_knee_linkage_upper_y_tau;
        0.0;
        right_knee_linkage_upper_y_tau;
        u_no_four_bar[12:13];
        0.0;
        0.0;
        u_no_four_bar[16:17];
        u_no_four_bar[14:15];
        u_no_four_bar[20:21];
        u_no_four_bar[18:19];
        u_no_four_bar[22:23];
    ]
    return u_four_bar
end

function four_bar_to_simple_control(u_four_bar)
    left_knee_linkage_upper_y_tau = u_four_bar[11]
    right_knee_linkage_upper_y_tau = u_four_bar[13]
    left_knee_y_tau = left_knee_linkage_upper_y_tau
    right_knee_y_tau = right_knee_linkage_upper_y_tau
    u_no_four_bar = [
        u_four_bar[1:9];
        left_knee_y_tau;
        right_knee_y_tau;
        u_four_bar[14:15];
        u_four_bar[20:21];
        u_four_bar[18:19];
        u_four_bar[24:25];
        u_four_bar[22:23];
        u_four_bar[26:27];
    ]
    return u_no_four_bar
end




AB = 0.05
BC = 0.11
AD = 0.09
CD = 0.05

knee_linkage_upper_y_default_angle = 2.5123 # radians
knee_shell_upper_y_default_angle = 0.04359 # radians
knee_shell_lower_y_default_angle = 1.8735 # radians
knee_linkage_lower_y_default_angle = 0.6678 # radians

function simple_to_four_bar(x_simple)
    left_knee_angle = x_simple[17]
    right_knee_angle = x_simple[18]
    left_knee_linkage_upper_y_angle = left_knee_angle*(AD/BC)
    right_knee_linkage_upper_y_angle = right_knee_angle*(AD/BC)

    theta_A = knee_linkage_upper_y_default_angle - left_knee_linkage_upper_y_angle
    r = sqrt(AB^2 + AD^2 -2*AB*AD*cos(theta_A))
    alpha_1 = asin((AD/r)*sin(theta_A))
    beta_1 = π - theta_A - alpha_1

    theta_C = acos((BC^2 + CD^2 - r^2)/(2*BC*CD))
    alpha_2 = asin((CD/r)*sin(theta_C))
    beta_2 = π - theta_C - alpha_2

    theta_B = alpha_1 - alpha_2
    theta_D = beta_2 - beta_1

    left_knee_shell_upper_y_angle = -(knee_shell_upper_y_default_angle - theta_B)
    left_knee_shell_lower_y_angle = knee_shell_lower_y_default_angle - theta_C
    

    theta_A = knee_linkage_upper_y_default_angle - right_knee_linkage_upper_y_angle
    r = sqrt(AB^2 + AD^2 -2*AB*AD*cos(theta_A))
    alpha_1 = asin((AD/r)*sin(theta_A))
    beta_1 = π - theta_A - alpha_1

    theta_C = acos((BC^2 + CD^2 - r^2)/(2*BC*CD))
    alpha_2 = asin((CD/r)*sin(theta_C))
    beta_2 = π - theta_C - alpha_2

    theta_B = alpha_1 - alpha_2
    theta_D = beta_2 - beta_1

    right_knee_shell_upper_y_angle = -(knee_shell_upper_y_default_angle - theta_B)
    right_knee_shell_lower_y_angle = knee_shell_lower_y_default_angle - theta_C

    x_four_bar_config = [
        x_simple[1:16];
        left_knee_shell_upper_y_angle;
        left_knee_linkage_upper_y_angle;
        right_knee_shell_upper_y_angle;
        right_knee_linkage_upper_y_angle;
        x_simple[19:20];
        left_knee_shell_lower_y_angle;
        right_knee_shell_lower_y_angle;
        x_simple[23:24];
        x_simple[21:22];
        x_simple[27:28];
        x_simple[25:26];
        x_simple[29:30];
    ]

    left_knee_vel = x_simple[46]
    right_knee_vel = x_simple[47]
    left_knee_linkage_upper_y_vel = left_knee_vel*(AD/BC)
    right_knee_linkage_upper_y_vel = right_knee_vel*(AD/BC)
    
    x_four_bar_vel = [
        x_simple[31:45];
        0.0;
        left_knee_linkage_upper_y_vel;
        0.0;
        right_knee_linkage_upper_y_vel;
        x_simple[48:49];
        0.0;
        0.0;
        x_simple[52:53];
        x_simple[50:51];
        x_simple[56:57];
        x_simple[54:55];
        x_simple[58:59]
    ]
    return [x_four_bar_config; x_four_bar_vel]
end


function four_bar_to_simple(x_four_bar)
    left_knee_linkage_upper_y_angle = x_four_bar[18]
    right_knee_linkage_upper_y_angle = x_four_bar[20]
    left_knee_angle = left_knee_linkage_upper_y_angle*(BC/AD)
    right_knee_angle = right_knee_linkage_upper_y_angle*(BC/AD)

    x_config = [
        x_four_bar[1:16];
        left_knee_angle;
        right_knee_angle;
        x_four_bar[21:22];
        x_four_bar[27:28];
        x_four_bar[25:26];
        x_four_bar[31:32];
        x_four_bar[29:30];
        x_four_bar[33:34];
    ]

    left_knee_linkage_upper_y_vel = x_four_bar[51]
    right_knee_linkage_upper_y_vel = x_four_bar[53]
    left_knee_vel = left_knee_linkage_upper_y_vel*(BC/AD)
    right_knee_vel = right_knee_linkage_upper_y_vel*(BC/AD)

    x_velocity = [
        x_four_bar[35:49];
        left_knee_vel;
        right_knee_vel;
        x_four_bar[54:55];
        x_four_bar[60:61];
        x_four_bar[58:59];
        x_four_bar[64:65];
        x_four_bar[62:63];
        x_four_bar[66:67]
    ]
    return [x_config; x_velocity]
end