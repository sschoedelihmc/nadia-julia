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

x_guess = load_object("nadia_balance_x_ref_fourbar.jld2")
u_guess = load_object("nadia_balance_u_ref.jld2")

set_configuration!(mvis, x_guess[1:nadia.nq])

# x_guess = [qtorp(x_guess[1:4]); x_guess[5:end]]

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
            5000*x_spine_x^2 + 500*x_spine_y^2
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

save_object("nadia_balance_x_ref_2.jld2", x_eq)
save_object("nadia_balance_u_ref_2.jld2", u_eq)

##

# z0 = [x_guess; u_guess; zero(x_guess)]
# Z = newtons_method(z0, nadia_kkt, nadia_kkt_jac, nadia_merit; tol = 1e-6, verbose = true, max_iters = 100)
# R = norm.(nadia_kkt.(Z))

# display(plot(1:length(R), R, yaxis=:log, xlabel="iteration", ylabel="|r|"))

# x, u = Z[end][idx_x], Z[end][idx_u]
# norm(dynamics(nadia, x, u))