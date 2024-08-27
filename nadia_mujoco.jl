using Pkg; Pkg.activate(@__DIR__);
using QuadrupedControl
using JLD2

include(joinpath(@__DIR__, "nadia_robot.jl"))
model = Nadia();
intf = init_mujoco_interface(model)
data = init_data(model, intf, preferred_monitor=3)

# Load reference
x_ref = load_object("nadia_balance_x_ref.jld2")
data.x .= change_order(model, x_ref, :rigidBodyDynamics, :mujoco)
set_data!(model, intf, data)


