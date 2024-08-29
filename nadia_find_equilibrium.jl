import Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()
using LinearAlgebra, Plots
import ForwardDiff as FD
using MeshCat
using Test
using Plots
using JLD2

include("nadia_robot_fixed_foot.jl") # this loads in our continuous time dynamics function xdot = dynamics(nadia, x, u)
include("control_utils.jl")

##

# --------these three are global variables------------
urdfpath = joinpath(@__DIR__, "nadia_V17_description/urdf/nadiaV17.fullRobot.simpleKnees.extended.cycloidArms.urdf");
nadia = NadiaFixed(urdfpath; fourbarknee=false)
vis = Visualizer()
render(vis)

##

mvis = init_visualizer(nadia, vis)

##

x_guess = load_object("nadia_balance_x_ref_3_central_foot.jld2")
u_guess = load_object("nadia_balance_u_ref_3_central_foot.jld2")

##

# x_guess[1] = 1.0 # pelvis quat w
# x_guess[2] = 0.0 # pelvis quat x
# x_guess[3] = 0.0 # pelvis quat y
# x_guess[4] = 0.0 # pelvis quat z
# x_guess[5] = 0.0 # pelvis trans x
x_guess[6] = -0.05 # pelvis trans y
# x_guess[7] = 1.065 # pelvis trans z

x_guess[10] = 0.0 # spine z
x_guess[13] = 0.0 # spine x
x_guess[16] = 0.0 # spine y

# leg_θ = 0.2

x_guess[12] = .175 # right hip x
# x_guess[15] = -0.4 # right hip y
# x_guess[9] = 0.0 # right hip z
# x_guess[18] = 0.75 # right knee y
x_guess[26] = -.205 # right ankle x
# x_guess[22] = -0.35 # right ankle y

x_guess[11] = 0.5 # left hip x
x_guess[14] = -0.6 # left hip y
x_guess[17] = 1.9 # left knee y
x_guess[25] = 0.0 # left ankle x

x_guess[20] = 0.64 # right shoulder y
x_guess[24] = -0.57 # right shoulder x
x_guess[28] = -0.61 # right shoulder z
x_guess[30] = -1.2 # right elbow y

x_guess[19] = 0.64 # left shoulder y
x_guess[23] = 0.57 # left shoulder x
x_guess[27] = 0.61 # left shoulder z
x_guess[29] = -1.2 # left elbow y

set_configuration!(mvis, x_guess[1:nadia.nq])

##

# indexing stuff
const idx_x = 1:59 # length(x_guess)
const idx_u = 60:82 # length(u_guess)
const idx_c = 83:141 # length(x_guess) for enforcing state equilibrium

# y = [x;u] then Newton's method will solve for z = [x;u;λ], or z = [y;λ]

function nadia_cost(y::Vector)::Real
    @assert length(y) == idx_u[end]
    x = y[idx_x]
    u = y[idx_u]
    
    x_spine_x = x[13]
    x_spine_y = x[16]
    x_no_spine = [x[1:12]; x[14:15]; x[17:end]]
    x_guess_no_spine = [x_guess[1:12]; x_guess[14:15]; x_guess[17:end]]
    u_right_ankle_y = u[15]
    u_right_ankle_x = u[19]
    
    return 50*norm(x_no_spine - x_guess_no_spine)^2 + 0.5*1e-3*norm(u[1:14])^2 +
            0.5*1e-3*norm(u[16:18])^2 + 0.5*1e-3*norm(u[20:end])^2 +
            3e-3*u_right_ankle_y^2 + 5*u_right_ankle_x^2 +
            5000*x_spine_x^2 + 5000*x_spine_y^2
end

function nadia_constraint(y::Vector)::Vector
    @assert length(y) == idx_u[end]
    x = y[idx_x]
    u = y[idx_u]
    
    return dynamics(nadia, x, u)
end

function nadia_kkt(z::Vector)::Vector
    @assert length(z) == idx_c[end]
    x = z[idx_x]
    u = z[idx_u]
    λ = z[idx_c]
    
    y = [x;u]
    
    [
        FD.gradient(nadia_cost,y) + FD.jacobian(nadia_constraint, y)'*λ;
        nadia_constraint(y)
    ]
end

function nadia_kkt_jac(z::Vector)::Matrix
    @assert length(z) == idx_c[end]
    x = z[idx_x]
    u = z[idx_u]
    λ = z[idx_c]
    
    y = [x;u]
    
    A = FD.jacobian(nadia_constraint, y)
    H = FD.hessian(nadia_cost, y)
    ρ = 1e-4
    [(H + ρ*I) A'; A -ρ*I]
end

##


function nadia_merit(z)
    # merit function for the balancing problem 
    @assert length(z) == idx_c[end]
    r = nadia_kkt(z)
    return norm(r[1:idx_u[end]]) + 1e4*norm(r[idx_c[1]:end])
end

z0 = [x_guess; u_guess; zero(x_guess)]
Z = newtons_method(z0, nadia_kkt, nadia_kkt_jac, nadia_merit; tol = 1e-6, verbose = false, max_iters = 100)

Z[end][1:4] /= norm(Z[end][1:4])
x_eq = Z[end][idx_x]
u_eq = Z[end][idx_u]

set_configuration!(mvis, x_guess[1:nadia.nq])
set_configuration!(mvis, x_eq[1:nadia.nq])

##

save_object("nadia_balance_x_ref_4_central_foot.jld2", x_eq)
save_object("nadia_balance_u_ref_4_central_foot.jld2", u_eq)

##

# z0 = [x_guess; u_guess; zero(x_guess)]
# Z = newtons_method(z0, nadia_kkt, nadia_kkt_jac, nadia_merit; tol = 1e-6, verbose = true, max_iters = 100)
R = norm.(nadia_kkt.(Z))

display(plot(1:length(R)-1, R[1:end-1], yaxis=:log, xlabel="iteration", ylabel="|r|"))

# x, u = Z[end][idx_x], Z[end][idx_u]
# norm(dynamics(nadia, x, u))