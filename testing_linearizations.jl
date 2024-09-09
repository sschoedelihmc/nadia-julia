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
let
    global K_pd, K_inf
ang = 40*pi/180;
K_pd = [zeros(model.nu, 7) diagm(0*ones(model.nu)) zeros(model.nu, 6) 1e1diagm(ones(model.nu))]
x_lin = [0; 0; 0.88978022; 1; 0; 0; 0; zeros(2); -ang; 2*ang; -ang; zeros(3); -ang; 2*ang; -ang; zeros(4); repeat([0; 0; 0; -pi/4], 2); zeros(model.nv)];
data.x = copy(x_lin);
u_lin = vcat(calc_continuous_eq(model, x_lin, K=K_pd, verbose = true)...);
u_lin = [u_lin[1:model.nu]; zeros(model.nc*3); u_lin[model.nu + 1:end]] # Update to include constraint forces on position-velocity kinematics
J_func(model, x) = BlockDiagonal([kinematics_jacobian(model, x)[:, 1:model.nq], kinematics_jacobian(model, x_lin)[:, 1:model.nq]*velocity_kinematics(model, x_lin)])

# Confirm that we do have an eq point with the full constraint
@assert norm(continuous_dynamics(model, x_lin, u_lin[1:model.nu], λ = u_lin[model.nu + 1:end], J_func=J_func, K=K_pd), Inf) < 1e-10

# Linearize dynamics (continuous time), constraints on position and velocity
E_jac = error_jacobian(model, x_lin);
E_jac_T = error_jacobian_T(model, x_lin);
cont_dyn_func(model, x, u, λ) = continuous_dynamics(model, x, u, λ=λ, J_func=J_func, K=K_pd)
A = E_jac_T*FiniteDiff.finite_difference_jacobian(_x -> cont_dyn_func(model, _x, u_lin[1:model.nu], u_lin[model.nu + 1:end]), x_lin)*E_jac;
B = E_jac_T*FiniteDiff.finite_difference_jacobian(_u -> cont_dyn_func(model, x_lin, _u, u_lin[model.nu + 1:end]), u_lin[1:model.nu]);
C = E_jac_T*FiniteDiff.finite_difference_jacobian(_λ -> cont_dyn_func(model, x_lin, u_lin[1:model.nu], _λ), u_lin[model.nu + 1:end]);
J = J_func(model, x_lin)*E_jac

# The dynamics system we are solving for (in x_{k+1} and λ_k)) is
# (I - dt*A)x_{k+1} - J'λ_k = x_k + dt*Bu_k
# s.t. Jx_{k+1} - 1/ρPλ_k = 0
# x ∈ R^58, u ∈ R^23, λ ∈ R^48 and J is 48 × 58 but rank(J) = 24. By default P = I(48)
dt = intf.m.opt.timestep
ρ = 1e5
P = I(size(J, 1)) # Regularizes everything, adds in artificial DoFs
# P = svd(J').V[:, 25:end]*svd(J').V[:, 25:end]' # Only regularizes things not in the constraint space
kkt_sys = [I-dt*A -dt*C; J -1/ρ*dt*P]
cond(kkt_sys)
Ā = (kkt_sys \ [I(model.nx - 1); zeros(size(J, 1), model.nx - 1)])[1:model.nx - 1, :]
B̄ = (kkt_sys \ [dt*B; zeros(size(J, 1), model.nu)])[1:model.nx - 1, :]

# Check Controllability
ranks = []; for len = 1:58
    C_gram = hcat([Ā^k*B̄ for k = 1:len]...)
    push!(ranks, rank(C_gram))
    cond(C_gram)
end; 
maximum(ranks)

# Calculate ihlqr
Q = diagm([1e3ones(model.nv); 5e0ones(model.nv)])
R = diagm(1e-3ones(model.nu));
# K_inf, P_inf = ihlqr(Ā, B̄, Q, R, Q, max_iters = 1000000)
P_inf = are(Discrete, Ā, B̄, Q, R)
K_inf = lqr(Discrete, Ā, B̄, Q, R)

# Check linear system eigenvalues
A_cl = Ā - B̄*K_inf
println(sort(abs.(eigvals(A_cl))))

# Do the same thing with a scaled version
using MATLAB
A_sc, B_sc, z2x = matlab_prescale(A, [B C])
B_sc, C_sc = B_sc[:, 1:model.nu], B_sc[:, model.nu + 1:end]
@assert norm(A_sc - inv(z2x)*A*z2x, Inf) < 1e-16
@assert norm(B_sc - inv(z2x)*B, Inf) < 1e-16
@assert norm(C_sc - inv(z2x)*C, Inf) < 1e-16
kkt_sys_sc = [I - dt*A_sc -dt*C_sc; J*z2x -1/ρ*dt*P]
cond(kkt_sys_sc)
Ā_sc = (kkt_sys_sc \ [I(model.nx - 1); zeros(size(J, 1), model.nx - 1)])[1:model.nx - 1, :]
B̄_sc = (kkt_sys_sc \ [dt*B_sc; zeros(size(J, 1), model.nu)])[1:model.nx - 1, :]
@assert norm(Ā_sc - inv(z2x)*Ā*z2x, Inf) < 1e-10 norm(Ā_sc - inv(z2x)*Ā*z2x, Inf)
@assert norm(B̄_sc - inv(z2x)*B̄, Inf) < 1e-12

# Calculate ihlqr
Q_sc = z2x*diagm([1e3ones(model.nv); 5e0ones(model.nv)])*z2x
R = diagm(1e-3ones(model.nu));
P_sc_inf = are(Discrete, Ā_sc, B̄_sc, Q_sc, R)
norm(P_sc_inf - z2x*P_inf*z2x, Inf)
K_sc_inf = lqr(Discrete, Ā_sc, B̄_sc, Q_sc, R)
norm(K_sc_inf - K_inf*z2x, Inf)
# @assert norm(K_sc_inf - K_inf*z2x, Inf) < 1e-4 norm(K_sc_inf - K_inf*z2x, Inf)

# Check linear system eigenvalues
A_sc_cl = Ā_sc - B̄_sc*K_sc_inf
println(sort(abs.(eigvals(A_sc_cl))))
@assert norm(A_sc_cl - inv(z2x)*A_cl*z2x, Inf) < 1e-6

# Simulate on the nonlinear system
x2z = inv(z2x)
let 
    x_ref = quasi_shift_foot_lift(shift_ang = 5)
    u_ref = vcat(calc_continuous_eq(model, x_ref)...);
    data.x = copy(x_ref)
    set_data!(model, intf, data)
    global res, U
    res = []
    U = []
    for k = 0:25000
        Δx =  state_error(model, data.x, x_lin) # State error
        u_lqr = -K_sc_inf*x2z*Δx
        # u_lqr = -K_inf*Δx

       @lock intf.p.lock begin # Simulate
            for k_inner = 1:1
                Δx =  state_error(model, data.x, x_lin) # State error

                u = u_lin[1:model.nu] - K_pd*x_lin - K_pd[:, 2:end]*Δx + u_lqr
                data.u = u#u_lin[1:model.nu] - K_pd*x_lin + K_pd*data.x - K_inf*Δx# + K_inf*state_error(model, x_ref, x_lin) # Calc control
                set_data!(model, intf, data) # Int sim
                MuJoCo.mj_step(intf.m, intf.d)
                get_data!(intf, data) # Get sim result

                push!(res, copy(Δx))
                push!(U, u)
            end
        end

        if norm(Δx[1:model.nq], Inf) > 1
            println("too far from desired state")
            break
        end

        push!(res, copy(Δx))
        push!(U, -K_inf*Δx)
        # sleep(0.0001)
    end
    
end
end

#### RANDOM NULLSPACE ATTEMPTS, etc



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