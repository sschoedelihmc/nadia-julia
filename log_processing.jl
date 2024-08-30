using Pkg; Pkg.activate(@__DIR__);
using Revise
using QuadrupedControl
using JLD2
using ForwardDiff
using SparseArrays
using Plots; gr(); theme(:dark); default(:size, (1920, 1080)); scalefontsizes(1.2)
using Statistics
import MuJoCo
using CSV
using DataFrames

include(joinpath(@__DIR__, "nadia_robot_fixed_foot.jl"))
include(joinpath(@__DIR__, "plot_utils.jl"))
include(joinpath(@__DIR__, "references.jl"))
include("control_utils.jl")
model_fixed = NadiaFixed();
generate_mujoco_stateorder(model_fixed);
vis = Visualizer();
mvis = init_visualizer(model_fixed, vis)
model = Nadia(nc_per_foot = 4)
intf = init_mujoco_interface(model)
data = init_data(model, intf, preferred_monitor=3);

df = DataFrame(CSV.File(joinpath(@__DIR__, "NadiaSimpleKneesWalkingInPlaceLog2/data.scs2.csv")));

# Load data
X_ref = [[0; 0; 0; 1; zeros(model.nx -4)] for _ = 1:nrow(df)]

# Config
for (i, jnt_name) in enumerate(model_fixed.orders[:mujoco].config_names)
    if i < 8
        [X_ref[k][i] = df[k, " root.nadia.q_PELVIS_LINK_"*String((jnt_name == :qw) ? :qs : jnt_name)] for k in 1:nrow(df)]
    else
        [X_ref[k][i] = df[k, " root.nadia.q_"*String(jnt_name)] for k in 1:nrow(df)]
    end
end

# Velocity
nadia_scs2_vels = [:x, :y, :z, :wX, :wY, :wZ]
for (i, jnt_name) in enumerate(model_fixed.orders[:mujoco].vel_names)
    if i < 7
        [X_ref[k][model.nq + i] = df[k, " root.nadia.qd_PELVIS_LINK_"*String(nadia_scs2_vels[i])] for k in 1:nrow(df)]
    else
        [X_ref[k][model.nq + i] = df[k, " root.nadia.qd_"*String(jnt_name)] for k in 1:nrow(df)]
    end
end

# Torque
U_ref = [zeros(model.nu) for _ = 1:nrow(df)]
for (i, jnt_name) in enumerate(model_fixed.orders[:mujoco].torque_names)
    [U_ref[k][i] = df[k, " root.nadia.tau_"*String(jnt_name)] for k in 1:nrow(df)]
end

# Check config-velocity match
dq = [state_error(model, X_ref[k + 1][1:model.nq], X_ref[k][1:model.nq])/diff(df[:, "root.time[sec]"])[k] for k in 1:nrow(df) - 1]
v = [x[model.nq + 1:end] for x in X_ref[2:end]]
norm(v, Inf)
norm(dq, Inf)
norm(hcat((dq - v)...), 2)/sqrt(length(v)) # RMSE

# TODO: Check dynamics residual and populate contact forces
t = df[:, "root.time[sec]"]
res = []

# visualize trajectory
anim = animate(model_fixed, mvis, [change_order(model_fixed, x, :mujoco, :rigidBodyDynamics) for x in X_ref]; Δt=mean(diff(df[:, "root.time[sec]"])), frames_to_skip=50);
setanimation!(mvis, anim)

# Save
dt = 0.003
joinpath(@__DIR__, "in_place_walking.jld2") X_ref U_ref dt

open("test.txt", "w") do file
    for name in names(df) 
        if occursin("tau_", name)
            write(file, name)
            write(file, "\n")
        end
    end
end