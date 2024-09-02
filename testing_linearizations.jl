using Pkg; Pkg.activate(@__DIR__);
using Revise
using QuadrupedControl
using JLD2
using FiniteDiff
using SparseArrays
using Plots; gr(); theme(:dark); default(:size, (1920, 1080)); scalefontsizes(1.2)
using Statistics
import ControlSystems:lqr, Discrete, are

import MuJoCo

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

# Set up standing i.c. and solve for stabilizing control
ang = 40*pi/180;
x_lin = [0; 0; 0.88978022; 1; 0; 0; 0; zeros(2); -ang; 2*ang; -ang; zeros(3); -ang; 2*ang; -ang; zeros(4); repeat([0; 0; 0; -pi/4], 2); zeros(model.nv)];
data.x = copy(x_lin)
set_data!(model, intf, data);

u_lin = vcat(calc_continuous_eq(model, x_lin)...);

## Continuous time
# Linearize dynamics
E_jac = error_jacobian(model, x_lin);
E_jac_T = error_jacobian_T(model, x_lin);
A = E_jac_T*FiniteDiff.finite_difference_jacobian(_x -> continuous_dynamics(model, _x, u_lin[1:model.nu], u_lin[model.nu + 1:end]), x_lin)*E_jac;
B = E_jac_T*FiniteDiff.finite_difference_jacobian(_u -> continuous_dynamics(model, x_lin, _u, u_lin[model.nu + 1:end]), u_lin[1:model.nu]);
C = E_jac_T*FiniteDiff.finite_difference_jacobian(_u -> continuous_dynamics(model, x_lin, u_lin[1:model.nu], _u), u_lin[model.nu + 1:end]);
J = kinematics_jacobian(model, x_lin)*E_jac;

# Calculate ihlqr gain

# Look at closed loop eigvals

## Discrete time
dt = 0.01
K_pd = get_PD_gains(model)*0
A_lin = E_jac_T*FiniteDiff.finite_difference_jacobian(x_ -> rk4(model, x_, u_lin[1:model.nu], u_lin[model.nu + 1:end], dt, K=K_pd), x_lin)*E_jac
B_lin = E_jac_T*FiniteDiff.finite_difference_jacobian(u_ -> rk4(model, x_lin, u_[1:model.nu], u_[model.nu + 1:end], dt, K=K_pd), u_lin)

J = kinematics_velocity_jacobian(model, x_lin)*E_jac
P = zeros(12, 24); P[CartesianIndex.(1:12, [1, 2, 3, 4, 6, 9, 13, 14, 15, 16, 18, 21])] .= 1;

# Controllability
ranks = []; for len = 1:58
    C_gram = hcat([A_lin^k*[B_lin[:, 1:model.nu] B_lin[:, model.nu + 1:end]*P'] for k = 1:len]...)
    push!(ranks, rank(C_gram))
    cond(C_gram)
end

## Test with unconstrained version
K_inf = lqr(Discrete, A_lin, [B_lin[:, 1:model.nu] B_lin[:, model.nu + 1:end]*P'], diagm(ones(model.nx - 1)), 1e-3diagm(ones(model.nu + 12)))

A_cl = A_lin - [B_lin[:, 1:model.nu] B_lin[:, model.nu + 1:end]*P']*K_inf
println(sort(abs.(eigvals(A_cl))))

x_ref = quasi_shift_foot_lift()
let 
    x = copy(x_ref)
    global res
    res = []
    Δx =  state_error(model, x, x_lin)
    for k = 0:1000
        # Δx = state_error(model, x, x_lin)
        push!(res, copy(Δx))
        x = apply_Δx(model, x_lin, A_cl*Δx )
        Δx = A_cl*Δx
        data.x = copy(x)
        set_data!(model, intf, data)
        sleep(0.001)
    end
    
end

## Test with constrained version (nullspace projection, configuration)
J = P*kinematics_jacobian(model, x_lin)*E_jac
svd_J = svd(J, full=true)
P_null = svd_J.V[:, 13:end]'

# Project dynamics
A_null = P_null*A_lin*P_null'
B_null = P_null*B_lin[:, 1:model.nu]

# Controllability
ranks = []; for len = 1:46
    C_gram = hcat([A_null^k*B_null for k = 1:len]...)
    push!(ranks, rank(C_gram))
    cond(C_gram)
end

# Gains (get a symplectic pencil issue if R not around 1e-3, didn't try tuning Q)
K_inf = lqr(Discrete, A_null, B_null, diagm(ones(model.nx - 13)), 1e-7diagm(ones(model.nu)))

A_cl = A_null - B_null*K_inf
println(sort(abs.(eigvals(A_cl))))
println(sort(real.(eigvals(A_cl))))

x_ref = quasi_shift_foot_lift()
let 
    x = copy(x_ref)
    global res
    res = []
    Δx =  P_null*state_error(model, x, x_lin)
    for k = 0:10000
        # Δx = state_error(model, x, x_lin)
        push!(res, copy(Δx))
        x = apply_Δx(model, x_lin, P_null'*A_cl*Δx )
        Δx = A_cl*Δx
        data.x = copy(x)
        set_data!(model, intf, data)
        # sleep(0.01)
    end
end

## Test with constrained version (nullspace projection, velocity)
J = P*kinematics_velocity_jacobian(model, x_lin)*E_jac
svd_J = svd(J, full=true)
P_null = svd_J.V[:, 13:end]'

# Project dynamics
A_null = P_null*A_lin*P_null'
B_null = P_null*B_lin[:, 1:model.nu]

# Controllability
ranks = []; for len = 1:46
    C_gram = hcat([A_null^k*B_null for k = 1:len]...)
    push!(ranks, rank(C_gram))
    cond(C_gram)
end

# Gains
K_inf = lqr(Discrete, A_null, B_null, diagm(P_null*[ones(model.nv); ones(model.nv)]), 1e+2diagm(ones(model.nu)))

A_cl = A_null - B_null*K_inf
println(sort(abs.(eigvals(A_cl))))

x_ref = quasi_shift_foot_lift()
let 
    x = copy(x_ref)
    global res
    res = []
    Δx =  P_null*state_error(model, x, x_lin)
    @showprogress for k = 0:10000000
        # Δx = state_error(model, x, x_lin)
        push!(res, copy(Δx))
        x = apply_Δx(model, x_lin, P_null'*A_cl*Δx )
        Δx = A_cl*Δx
        data.x = copy(x)
        set_data!(model, intf, data)
        # sleep(0.01)
    end
end

# KKT system with continuous time + backward-euler + primal-dual + penalty
J_kkt = kinematics_jacobian(model, x_lin)*E_jac
ρ = 1e5
kkt_sys = [I - dt*A J_kkt'; J_kkt -1/ρ*I]
kkt_lhs_x = [I(model.nx - 1); zeros(size(J_kkt, 1), model.nx - 1)]
kkt_lhs_u = [dt*B; zeros(size(J_kkt, 1), model.nu)]
A_kkt = (kkt_sys \ kkt_lhs_x)[1:model.nx - 1, :]
B_kkt = (kkt_sys \ kkt_lhs_u)[1:model.nx - 1, :]
A_schur = (I - dt*A + ρ*J_kkt'*J_kkt) \ I
B_schur = (I - dt*A + ρ*J_kkt'*J_kkt) \ (dt*B)
cond(kkt_sys)

# Controllability
ranks = []; for len = 1:58
    C_gram = hcat([A_schur^k*B_schur for k = 1:len]...)
    push!(ranks, rank(C_gram))
    cond(C_gram)
end

## KKT with nullspace version
ρ = 1e3
# J_kkt = 
svd_J = svd(J_kkt', full = true)
P_null = svd_J.V[:, rank(J_kkt') + 1:end]*svd_J.V[:, rank(J_kkt') + 1:end]'
kkt_sys = [I - dt*A J_kkt'; J_kkt -1/ρ*P_null]
kkt_lhs_x = [I(model.nx - 1); zeros(size(J_kkt, 1), model.nx - 1)]
kkt_lhs_u = [dt*B; zeros(size(J_kkt, 1), model.nu)]
A_kkt = (kkt_sys \ kkt_lhs_x)[1:model.nx - 1, :]
B_kkt = (kkt_sys \ kkt_lhs_u)[1:model.nx - 1, :]
cond(kkt_sys)

# Controllability
ranks = []; for len = 1:58
    C_gram = hcat([A_schur^k*B_schur for k = 1:len]...)
    push!(ranks, rank(C_gram))
    cond(C_gram)
end

cond((I - dt*A))

## Symmetric KKT version
ρ = 1e5
kkt_sys = [(I - dt*A)'*(I - dt*A) J_kkt'; J_kkt -1/ρ*I]
cond(kkt_sys)