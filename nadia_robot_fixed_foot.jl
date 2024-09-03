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
    nu::Int
    torque_limits::Vector{Float64}
    function NadiaFixed(urdfpath)

        # Create robot and fix right foot to world (without deleting pelvis)
        mech = parse_urdf(urdfpath, floating=true, remove_fixed_tree_joints=true) 

        right_foot = findbody(mech, "RIGHT_FOOT_LINK")
        world = findbody(mech, "world")
        right_foot_fixed_joint = Joint("right_foot_fixed_joint", Fixed{Float64}())
        # foot_translation = SA[-0.004351222979328111, -0.13006285479389704, 0.08180975114086993]
        # foot_translation = SA[-0.002351222979328111, -0.12106285479389704, 0.08180975114086993]
        foot_translation = SA[0.0, 0.0, 0.08180975114086993]

        world_to_joint = Transform3D(
            frame_before(right_foot_fixed_joint),
            default_frame(world),
            foot_translation
        )

        attach!(mech, world, right_foot, right_foot_fixed_joint, joint_pose=world_to_joint)

        # Stabilization gains for non-tree joints
        baumgarte_gains = Dict(
            JointID(right_foot_fixed_joint) => SE3PDGains(PDGains(3000.0, 200.0), PDGains(3000.0, 200.0)) # angular, linear
        )

        torque_limits = [ # TODO: set these with values from SCS
            360.0,  # LEFT_HIP_Z
            740.0,  # RIGHT_HIP_Z
            890.0,  # SPINE_Z
            840.0,  # LEFT_HIP_X
            530.0,  # RIGHT_HIP_X
            275.0,  # SPINE_X
            106.0,  # LEFT_HIP_Y
            275.0,  # RIGHT_HIP_Y
            445.0,  # SPINE_Y
            530.0,  # LEFT_KNEE_Y
            300.0,  # RIGHT_KNEE_Y
            840.0,  # LEFT_SHOULDER_Y
            87.0,   # RIGHT_SHOULDER_Y
            87.0,   # LEFT_ANKLE_Y
            890.0,  # RIGHT_ANKLE_Y
            99.0,   # LEFT_SHOULDER_X
            99.0,   # RIGHT_SHOULDER_X
            740.0,  # LEFT_ANKLE_X
            63.0,   # RIGHT_ANKLE_X
            63.0,   # LEFT_SHOULDER_Z
            360.0,  # RIGHT_SHOULDER_Z
            112.0,  # LEFT_ELBOW_Y
            112.0   # RIGHT_ELBOW_Y
        ]

        new(mech, MechanismState(mech), DynamicsResult(mech), StateCache(mech), DynamicsResultCache(mech),
            baumgarte_gains, urdfpath,
            num_positions(mech), num_velocities(mech), 23,
            torque_limits)
    end
end

function dynamics(model::NadiaFixed, x::AbstractVector{T1}, u::AbstractVector{T2}; gains=RigidBodyDynamics.default_constraint_stabilization_gains(Float64)) where {T1, T2}
    T = promote_type(T1, T2)
    state = model.statecache[T]
    dyn_result = model.dyn_result_cache[T]

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

