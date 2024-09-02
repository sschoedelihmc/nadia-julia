# Setup the dynamics and visualization for the Nadia model
using MeshCat
using MeshCatMechanisms
using RigidBodyDynamics
using RigidBodyDynamics.PDControl
using StaticArrays

struct NadiaFixed <: PinnZooModel # TODO fix this hack to make the model play nice with QuadrupedControl.jl
    mech::Mechanism{Float64}
    state::MechanismState
    dyn_result::DynamicsResult
    statecache::StateCache
    dyn_result_cache::DynamicsResultCache
    baumgarte_gains
    urdfpath::String
    nq::Int
    nv::Int
    nx::Int
    nu::Int
    orders::Dict{Symbol, StateOrder}
    conversions::Dict{Tuple{Symbol, Symbol}, ConversionIndices}
    function NadiaFixed()

        # Create robot and fix right foot to world (without deleting pelvis)
        urdfpath = joinpath(@__DIR__, "nadia_V17_description/urdf/nadiaV17.fullRobot.simpleKnees.cycloidArms.urdf");
        mech = parse_urdf(urdfpath, floating=true, remove_fixed_tree_joints=true)

        right_foot = findbody(mech, "RIGHT_FOOT_LINK")
        world = findbody(mech, "world")
        right_foot_fixed_joint = Joint("right_foot_fixed_joint", Fixed{Float64}())
        foot_translation = SA[-0.004351222979328111, -0.13006285479389704, 0.08180975114086993]

        world_to_joint = Transform3D(
            frame_before(right_foot_fixed_joint),
            default_frame(world),
            foot_translation
        )

        attach!(mech, world, right_foot, right_foot_fixed_joint, joint_pose = world_to_joint)

        # Stabilization gains for non-tree joints
        baumgarte_gains = Dict(JointID(right_foot_fixed_joint) => SE3PDGains(PDGains(3000.0, 200.0), PDGains(3000.0, 200.0))) # angular, linear

        # Generate state order for rigidBodyDynamics
        state = MechanismState(mech)
        config_names =  [Symbol(joints(mech)[id].name) for id in state.q_index_to_joint_id]
        vel_names = [Symbol(joints(mech)[id].name) for id in state.v_index_to_joint_id]

        # Fix floating base
        for joint in joints(mech)
            if typeof(joint.joint_type) <: QuaternionFloating
                config_names[state.qranges[joint]] = [:qw, :qx, :qy, :qz, :x, :y, :z]
                vel_names[state.vranges[joint]] = [:wx, :wy, :wz, :vx, :vy, :vz]
            end
        end

        orders = Dict{Symbol, StateOrder}()
        orders[:rigidBodyDynamics] = StateOrder(config_names, vel_names)
        conversions = generate_conversions(orders) # Will be empty to start

        new(mech, MechanismState(mech), DynamicsResult(mech), StateCache(mech), DynamicsResultCache(mech), baumgarte_gains, 
                urdfpath, num_positions(mech), num_velocities(mech), num_positions(mech) + num_velocities(mech), 23, orders, conversions)
    end
end

QuadrupedControl.get_mujoco_xml_path(model::NadiaFixed) = joinpath(@__DIR__, "nadia_V17_description/mujoco/nadiaV17.simpleKnees_scene.xml")

function dynamics(model::NadiaFixed, x::AbstractVector{T1}, u::AbstractVector{T2}; gains=RigidBodyDynamics.default_constraint_stabilization_gains(Float64)) where {T1, T2}
    T = promote_type(T1, T2)
    state = model.statecache[T]
    dyn_result = model.dyn_result_cache[T]

    # Set the mechanism state
    copyto!(state, x)

    # Perform forward dynamics (six zeros because RigidBodyDynamics allows control over the pelvis)
    dynamics!(dyn_result, state, [zeros(6); u]; stabilization_gains=gains)

    return [dyn_result.q̇; dyn_result.v̇]
end

function rbd_rk4(model::NadiaFixed, x, u, h; gains=RigidBodyDynamics.default_constraint_stabilization_gains(Float64))
    k1 = dynamics(model, x, u; gains=gains)
    k2 = dynamics(model, x + h/2*k1, u; gains=gains)
    k3 = dynamics(model, x + h/2*k2, u; gains=gains)
    k4 = dynamics(model, x + h*k3, u; gains=gains)
    return x + h/6*(k1 + 2*k2 + 2*k3 + k4)
end

function init_visualizer(model::NadiaFixed, vis::Visualizer)
    delete!(vis)
    mvis = MechanismVisualizer(model.mech, URDFVisuals(model.urdfpath, package_path=[@__DIR__]), vis)
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