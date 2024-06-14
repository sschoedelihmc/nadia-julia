using Pkg; Pkg.activate(@__DIR__)

using ForwardDiff
using MeshCatMechanisms
using RigidBodyDynamics
using MeshCat

vis = Visualizer()
render(vis)

urdf = joinpath(dirname(pathof(MeshCatMechanisms)), "..", "test", "urdf", "Acrobot.urdf")
robot = parse_urdf(urdf)
delete!(vis)

mvis = MechanismVisualizer(robot, URDFVisuals(urdf), vis)
set_configuration!(mvis, [0.0, 0.0])

state = MechanismState(robot, randn(2), randn(2))
t, q, v = simulate(state, 5.0);

animation = Animation(mvis, t, q)
setanimation!(mvis, animation)

lower_arm = bodies(robot)[end]
body_frame = default_frame(lower_arm)
setelement!(mvis, body_frame)

radius = 0.05
name = "my_point"
setelement!(mvis, Point3D(body_frame, 0.2, 0.2, 0.2), radius, name)

