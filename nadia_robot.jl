# Setup the dynamics and visualization for the Nadia model
using RigidBodyDynamics
using MeshCat
using MeshCatMechanisms
using Random
using StaticArrays
using Rotations
using LinearAlgebra
using ForwardDiff
import ForwardDiff as FD

struct Nadia
    mech::Mechanism{Float64}
    statecache::StateCache
    dynrescache::DynamicsResultCache
    nq::Int
    nv::Int
    nx::Int
    nu::Int
    joint_names
    function Nadia(mech)
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
        # remove_joint!(mech, findjoint(mech, "PELVIS_LINK_to_world"))

        # Get mechanism details
        nq = num_positions(mech)
        nv = num_velocities(mech)
        nx = nq + nv
        nu = nq

        new(mech, StateCache(mech), DynamicsResultCache(mech), nq, nv, nx, nu)
    end
end

function dynamics(model::Nadia, x::AbstractVector{T1}, u::AbstractVector{T2}) where {T1, T2}
    T = promote_type(T1, T2)
    state = model.statecache[T]
    dyn_result = model.dynrescache[T]

    # Set the mechanism state
    copyto!(state, x)

    # Perform forward dynamics
    dynamics!(dyn_result, state, [zeros(6); u])

    return [dyn_result.q̇; dyn_result.v̇]
end

function rk4(model::Nadia, x, u, h)
    k1 = dynamics(model, x, u)
    k2 = dynamics(model, x + h/2*k1, u)
    k3 = dynamics(model, x + h/2*k2, u)
    k4 = dynamics(model, x + h*k3, u)
    return x + h/6*(k1 + 2*k2 + 2*k3 + k4)
end

function init_visualizer(model::Nadia, vis::Visualizer, urdfpath)
    delete!(vis)
    mvis = MechanismVisualizer(model.mech, URDFVisuals(urdfpath), vis)
    return mvis
end

function visualize!(model::Nadia, mvis::MechanismVisualizer, q)
    set_configuration!(mvis, q[1:model.nq])
end

function animate!(model::Nadia, mvis::MechanismVisualizer, qs; Δt=0.001)
    anim = MeshCat.Animation(convert(Int, floor(1.0 / Δt)))
    for (t, q) in enumerate(qs)
        MeshCat.atframe(anim, t) do 
            set_configuration!(mvis, q[1:model.nq])
        end
    end
    MeshCat.setanimation!(mvis, anim)

    return anim
end