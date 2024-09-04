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

# let
    # global K_pd, K_inf

# Set up standing i.c. and solve for stabilizing control
ang = 40*pi/180;
K_pd = [zeros(model.nu, 7) diagm(1e1*ones(model.nu)) zeros(model.nu, 6) 5e0diagm(ones(model.nu))]
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

# The dynamics system we are solving for (in x_{k+1} and λ_k)) is
# (I - dt*A)x_{k+1} - J'λ_k = x_k + dt*Bu_k
# s.t. Jx_{k+1} - 1/ρPλ_k = 0
# x ∈ R^58, u ∈ R^23, λ ∈ R^48 and J is 48 × 58 but rank(J) = 24. By default P = I(48)

# Parameters (horizon, timestep, penalty, sizing)
N = 50;
dt = 0.01;
ρ = 1e5;
nΔx, nΔu, nc = model.nx - 1, model.nu + model.nc*3*2, size(J, 1)
nΔz = nΔx*(N + 1) + nΔu*N

# Define indexing variables
Δxi = [(nΔu+nΔx)*(i-1) .+ (1:nΔx)  for i = 1:N+1];
Δui = [(nΔu+nΔx)*(i-1) .+ (nΔx .+ (1:nΔu))  for i = 1:N];
ci = [(nΔx)*(i-1) .+ (1:nΔx)  for i = 1:N + 1];

# Set up cost hessian
Q = diagm([1e4ones(model.nv); 1e-1ones(model.nv)]);
R = diagm([1e-4ones(model.nu); 0*ones(model.nc*3*2)]);
Qf = diagm([1e5ones(model.nv); 25e0ones(model.nv)]); # Todo maybe init with ihlqr
# Cost function
qp_H = spzeros(nΔz, nΔz);
qp_g = zeros(nΔz);
for i = 1:N
    # Δx_ref, Δu_ref, _ = index_ref(ref, (i - 1)*ref.dt)
    qp_H[Δxi[i],Δxi[i]] = Q
    # g[Δxi[i]] .= -Q*Δx_ref # Assume no Δ reference on start (would normally be -Q*Δxref_i)
    qp_H[Δui[i],Δui[i]] = R
    # g[Δui[i]] .= -R*Δu_ref # Assume no Δ reference on start (would normally be -R*Δuref_i)
end
# Δx_ref, _ = index_ref(ref, N*ref.dt)
qp_H[Δxi[N+1],Δxi[N+1]] = Qf;
# g[Δxi[N+1]] .= Qf*Δx_ref # Assume no Δ reference on start (would normally be -Qf*Δxref_{N+1})

# Set up dynamics (backwards-euler using linearized continuous dynamics)
qp_A_dyn = spzeros((N+1)*nΔx, nΔz);
qp_b_dyn = zeros((N+1)*nΔx);
qp_A_dyn[ci[1],Δxi[1]] = I(nΔx); # Initial condition constraint
for i = 1:N
    qp_A_dyn[ci[i+1],Δxi[i]] = I(nΔx)
    qp_A_dyn[ci[i+1],Δui[i]] = [dt*B J']
    qp_A_dyn[ci[i+1],Δxi[i+1]] = (I - dt*A)
end
qp_b_dyn[ci[1]] .= 0; # Initial condition constraint (assume no Δ during setup)

# Set up contact constraints
qp_A_foot = spzeros(N*nc, nΔz);
qp_b_foot = zeros(N*nc);
cfoot = [(k - 1)*nc .+ (1:nc) for k = 1:N];
for i = 1:N
    qp_A_foot[cfoot[i],Δui[i]] = [zeros(nc, model.nu) 1/ρ*I(nc)]
    qp_A_foot[cfoot[i],Δxi[i + 1]] = J
end

# Assemble the problem
qp_A = [qp_A_dyn; qp_A_foot];
qp_b = [qp_b_dyn; qp_b_foot];
kkt_sys = [qp_H qp_A'; qp_A spzeros(size(qp_A, 1), size(qp_A, 1))];

kkt_sys_factor = qr(kkt_sys);
println(rank(kkt_sys_factor), " ", size(kkt_sys_factor))

# Test
Δx = state_error(model, quasi_shift_foot_lift(), x_lin);
kkt_lhs = [qp_g; qp_b]; kkt_lhs[nΔz .+ (1:nΔx)] = Δx;
res = kkt_sys_factor \ kkt_lhs;

# Simulate on the nonlinear system
let 
    x_ref = quasi_shift_foot_lift(shift_ang = 2.5)
    u_ref = vcat(calc_continuous_eq(model, x_ref)...);
    data.x = copy(x_ref)
    set_data!(model, intf, data)
    global res, U
    res = []
    U = []
    @showprogress for k = 0:1000
        Δx =  state_error(model, data.x, x_lin) # State error
        # Update and solve QP
        kkt_lhs[nΔz .+ (1:nΔx)] = Δx
        result = kkt_sys_factor \ kkt_lhs
        @lock intf.p.lock begin # Simulate
            for k_inner = 1:4
                Δx =  state_error(model, data.x, x_lin) # State error

                u = u_lin[1:model.nu] - K_pd*x_lin - K_pd[:, 2:end]*Δx + result[Δui[1]][1:model.nu]
                data.u = u#u_lin[1:model.nu] - K_pd*x_lin + K_pd*data.x - K_inf*Δx# + K_inf*state_error(model, x_ref, x_lin) # Calc control
                set_data!(model, intf, data) # Int sim
                MuJoCo.mj_step(intf.m, intf.d)
                get_data!(intf, data) # Get sim result

                push!(res, copy(Δx))
                push!(U, u)
            end
        end
        if norm(Δx[1:model.nq], Inf) > 1
            break
        end
    end
    plot(intf.m.opt.timestep.*(1:length(res)), [norm(r[1:model.nv]) for r in res])
end
# end