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
set_data!(model, intf, data);
u_lin = vcat(calc_continuous_eq(model, x_lin, K=K_pd)...);

# Linearize dynamics (continuous time), constraints on position and velocity
E_jac = error_jacobian(model, x_lin);
E_jac_T = error_jacobian_T(model, x_lin);
A = E_jac_T*FiniteDiff.finite_difference_jacobian(_x -> continuous_dynamics(model, _x, u_lin[1:model.nu], u_lin[model.nu + 1:end], K=K_pd), x_lin)*E_jac;
B = E_jac_T*FiniteDiff.finite_difference_jacobian(_u -> continuous_dynamics(model, x_lin, _u, u_lin[model.nu + 1:end], K=K_pd), u_lin[1:model.nu]);
C = E_jac_T*FiniteDiff.finite_difference_jacobian(_u -> continuous_dynamics(model, x_lin, u_lin[1:model.nu], _u, K=K_pd), u_lin[model.nu + 1:end]);
J = [kinematics_jacobian(model, x_lin)*E_jac; kinematics_velocity_jacobian(model, x_lin)*E_jac]

# Redo J to not be rank deficient
# P = zeros(12, 24); P[CartesianIndex.(1:12, [1, 2, 3, 4, 6, 9, 13, 14, 15, 16, 18, 21])] .= 1;
# J = [kinematics_jacobian(model, x_lin)*E_jac; kinematics_velocity_jacobian(model, x_lin)*E_jac]

# The dynamics system we are solving for (in x_{k+1} and λ_k)) is
# (I - dt*A)x_{k+1} - J'λ_k = x_k + dt*Bu_k
# s.t. Jx_{k+1} - 1/ρPλ_k = 0
# x ∈ R^58, u ∈ R^23, λ ∈ R^48 and J is 48 × 58 but rank(J) = 24. By default P = I(48)
dt = intf.m.opt.timestep
ρ = 1e1
P = I(size(J, 1)) # Regularizes everything, adds in artificial DoFs
P = svd(J').V[:, 25:end]*svd(J').V[:, 25:end]' # Only regularizes things not in the constraint space
kkt_sys = [I-dt*A J'; J -1/ρ*P]
cond(kkt_sys)
Ā = (kkt_sys \ [I(model.nx - 1); zeros(size(J, 1), model.nx - 1)])[1:model.nx - 1, :]
B̄ = (kkt_sys \ [dt*B; zeros(size(J, 1), model.nu)])[1:model.nx - 1, :]

# Check Controllability
ranks = []; for len = 1:58
    C_gram = hcat([Ā^k*B̄ for k = 1:len]...)
    push!(ranks, rank(C_gram))
    cond(C_gram)
end

# Calculate ihlqr
P_range = I - svd(J).V[:, 1:24]*svd(J).V[:, 1:24]'
Q = diagm([1e3ones(model.nv); 5e0ones(model.nv)])
Q = 0.5*(Q + Q')
R = diagm(1e-3ones(model.nu));
K_inf = lqr(Discrete, Ā, B̄, Q, R)
# K_inf, P_inf = ihlqr(Ā, B̄, Q, R, Q, max_iters = 100000)

# Simulate on the linear system
A_cl = Ā - B̄*K_inf
println(sort(abs.(eigvals(A_cl))))
# let 
#     x =  quasi_shift_foot_lift()
#     global res, U
#     res = []
#     U = []
#     Δx =  state_error(model, x, x_lin)
#     for k = 0:1000
#         # Δx = state_error(model, x, x_lin)
#         push!(res, copy(Δx))
#         push!(U, u_lin[1:model.nu]-K_inf*Δx)
#         x = apply_Δx(model, x_lin, A_cl*Δx )
#         Δx = A_cl*Δx
#         data.x = copy(x)
#         set_data!(model, intf, data)
#         sleep(0.0001)
#     end
    
# end

# Simulate on the nonlinear system
let 
    x_ref = quasi_shift_foot_lift()
    u_ref = vcat(calc_continuous_eq(model, x_ref)...);
    data.x = copy(x_ref)
    set_data!(model, intf, data)
    global res, U
    res = []
    U = []
    for k = 0:100000
        Δx =  state_error(model, data.x, x_lin) # State error
        u = u_lin[1:model.nu] - K_pd*x_lin - (K_pd[:, 2:end] + K_inf)*Δx
        data.u = u#u_lin[1:model.nu] - K_pd*x_lin + K_pd*data.x - K_inf*Δx# + K_inf*state_error(model, x_ref, x_lin) # Calc control
        set_data!(model, intf, data) # Int sim
        @lock intf.p.lock begin # Simulate
            MuJoCo.mj_step(intf.m, intf.d)
        end
        get_data!(intf, data) # Get sim result

        if norm(Δx[1:model.nq], Inf) > 1
            break
        end

        push!(res, copy(Δx))
        push!(U, -K_inf*Δx)
        # sleep(0.0001)
    end
    
end
end