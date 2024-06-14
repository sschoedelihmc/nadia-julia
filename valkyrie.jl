using Pkg; Pkg.activate(@__DIR__)

using ForwardDiff
using MeshCatMechanisms
using RigidBodyDynamics
using MeshCat
using ValkyrieRobot
using ValkyrieRobot.BipedControlUtil: Side, flipsign_if_right

##

vis = Visualizer()
render(vis)

##

val = Valkyrie();
delete!(vis)
mvis = MechanismVisualizer(
    val.mechanism, 
    URDFVisuals(ValkyrieRobot.urdfpath(), package_path=[dirname(dirname(ValkyrieRobot.urdfpath()))]),
    vis);

##


function initialize!(state::MechanismState, val::Valkyrie)
    zero!(state)
    mechanism = val.mechanism
    for side in instances(Side)
        set_configuration!(state, findjoint(mechanism, "$(side)KneePitch"), [1.205])
        set_configuration!(state, findjoint(mechanism, "$(side)HipPitch"), [-0.49])
        set_configuration!(state, findjoint(mechanism, "$(side)AnklePitch"), [-0.71])
        set_configuration!(state, findjoint(mechanism, "$(side)ShoulderPitch"), [0.300196631343025])
        set_configuration!(state, findjoint(mechanism, "$(side)ShoulderRoll"), [flipsign_if_right(-1.25, side)])
        set_configuration!(state, findjoint(mechanism, "$(side)ElbowPitch"), [flipsign_if_right(-0.785398163397448, side)])
        set_configuration!(state, findjoint(mechanism, "$(side)ForearmYaw"), [1.571])
    end
    set_configuration!(state, val.basejoint, [1; 0; 0; 0; 0; 0; 1.025])
    nothing
end
state = MechanismState(val.mechanism)
initialize!(state, val)
set_configuration!(mvis, configuration(state))

##

t, q, v = simulate(state, 2.0);

animation = Animation(mvis, t, q)
setanimation!(mvis, animation)