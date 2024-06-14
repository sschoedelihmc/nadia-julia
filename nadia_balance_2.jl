using Pkg; Pkg.activate(@__DIR__);
using JLD2
using SparseArrays

include("mpc_utils.jl")
include("data.jl")
include("nadia_robot.jl")
include("quaternions.jl")

# Setup model and visualizer

# Create the Mechanism (defaults to floating base)
urdf = joinpath(@__DIR__, "nadia_V17_description/urdf/nadiaV17.fullRobot.simpleKnees.cycloidArms.urdf");
robot = parse_urdf(urdf, floating=true, remove_fixed_tree_joints=true)

nadia = Nadia(robot);
vis = Visualizer();
mvis = init_visualizer(nadia, vis, urdf)

##

# Load balanced reference
u_ref =
[
-1.529996146733067,     # "LEFT_HIP_Z"
9.964441654400309,      # "RIGHT_HIP_Z"
2.4214516921032456,     # "SPINE_Z"
30.725284382117,        # "LEFT_HIP_X"
-102.44994521205273,    # "RIGHT_HIP_X"
-10.165094068114897,    # "SPINE_X"
1.3888702260931227,     # "LEFT_HIP_Y"
30.57276337319791,      # "RIGHT_HIP_Y"
-5.786704258930225,     # "SPINE_Y"
18.379170992769396,     # "LEFT_KNEE_Y"
-87.7703566391235,      # "RIGHT_KNEE_Y"
2.673614241444654,      # "LEFT_SHOULDER_Y"
3.90078753582609,       # "RIGHT_SHOULDER_Y"
0.6569298958706897,     # "LEFT_ANKLE_Y"
31.358889860182014,     # "RIGHT_ANKLE_Y"
7.695142597693708,      # "LEFT_SHOULDER_X"
-4.036609656052576,     # "RIGHT_SHOULDER_X"
0.22082314816501622,    # "LEFT_ANKLE_X"
-0.0021063087760480495, # "RIGHT_ANKLE_X"
0.4149513856694272,     # "LEFT_SHOULDER_Z"
-0.2869521910778749,    # "RIGHT_SHOULDER_Z"
-0.532071050841635,     # "LEFT_ELBOW_Y"
-0.5439629977475727     # "RIGHT_ELBOW_Y"
]

q_ref = 
[0.9851812850604109, 0.14091286404085285, -0.021356269245195184, -0.095421748536162, # pelvis orientation
-0.04426391998996947, -0.1668505364459041, 1.0334898178372895,     # pelvis position
0.06781608404733115, # LEFT_HIP_Z
-0.1922539118626478, # RIGHT_HIP_Z
0.2861350253811403, # SPINE_Z
0.15898265237693734, # LEFT_HIP_X
-0.12064158590993693, # RIGHT_HIP_X
-0.1057141650799965, # SPINE_X
-0.4044177896371911, # LEFT_HIP_Y
-0.3535655955195078, # RIGHT_HIP_Y
0.07352471380726427, # SPINE_Y
1.500022076915403, # LEFT_KNEE_Y
0.7689918571683205, # RIGHT_KNEE_Y
0.2271611304501065, # LEFT_SHOULDER_Y
0.6387021266726293, # RIGHT_SHOULDER_Y
0.22893847881329563, # LEFT_ANKLE_Y
-0.3460980826665025, # RIGHT_ANKLE_Y
0.35167329835530214, # LEFT_SHOULDER_X
-0.2566674753190524, # RIGHT_SHOULDER_X
-0.43626403944917924, # LEFT_ANKLE_X
-0.15679667582212356, # RIGHT_ANKLE_X
0.7430339207647649, # LEFT_SHOULDER_Z
-0.608843261553189, # RIGHT_SHOULDER_Z
-1.2994261459930767, # LEFT_ELBOW_Y
-1.475040588733499] # RIGHT_ELBOW_Y

x_ref = [q_ref; zeros(nadia.nv)]

visualize!(nadia, mvis, x_ref)

##

render(mvis)

##

# Calculate linearized discrete dynamics at equilibrium position
h = 0.01;
Ad = FD.jacobian(x_->rk4(nadia, x_, u_ref, h), x_ref);
Bd = FD.jacobian(u_->rk4(nadia, x_ref, u_, h), u_ref);

AdReduced = E(x_ref[1:4])' * Ad * E(x_ref[1:4])
BdReduced = E(x_ref[1:4])' * Bd

# Set up cost matrices (hand-tuned)
Q = spdiagm([1e3*ones(12); repeat([1e1; 1e1; 1e3], 3); 1e1*ones(8); 1e2*ones(12); repeat([1; 1; 1e2], 3); 1*ones(8)]);
R = spdiagm(1e-3*ones(length(u_ref)));

# Calculate infinite-horizon LQR cost-to-go and gain matrices
K, Qf = ihlqr(AdReduced, BdReduced, Q, R, Q, max_iters = 1000);

# # Define additional constraints for the QP (just torques for Atlas)
# horizon = 2;
# A_torque = kron(I(horizon), [I(length(u_ref)) zeros(length(u_ref), atlas.nx)]);
# l_torque = repeat(-atlas.torque_limits - u_ref, horizon);
# u_torque = repeat(atlas.torque_limits - u_ref, horizon);

# # Setup QP
# H, g, A, l, u, g_x0, lu_x0 = gen_condensed_mpc_qp(Ad, Bd, Q, R, Qf, horizon, A_torque, l_torque, u_torque, K);

# # Setup solver
# m = ReLUQP.setup(H, g, A, l, u, verbose = false, eps_primal=1e-2, eps_dual=1e-2, max_iters=10, iters_btw_checks=1);

# Simulate
N = 300;
X = [zeros(nadia.nx) for _ = 1:N];
U = [zeros(length(u_ref)) for _ = 1:N];
X[1] = deepcopy(x_ref);
X[1][nadia.nq + 5] = 1.3; # Perturb i.c.

# # Warmstart solver
# Δx = X[1] - x_ref;
# ReLUQP.update!(m, g = g + g_x0*Δx, l = l + lu_x0*Δx, u = u + lu_x0*Δx);
# m.opts.max_iters = 4000;
# m.opts.check_convergence = false;
# ReLUQP.solve(m);
# m.opts.max_iters = 10;

# Run simulation
for k = 1:N - 1
    # Get error
    global Δx̃ = [qtorp(L(x_ref[1:4])'*X[k][1:4]); X[k][5:end] - x_ref[5:end]]
    # global Δx = X[k] - x_ref

    # # Update solver
    # ReLUQP.update!(m, g = g + g_x0*Δx, l = l + lu_x0*Δx, u = u + lu_x0*Δx)

    # # Solve and get controls
    # results = ReLUQP.solve(m)
    # global U[k] = results.x[1:length(u_ref)] - K*Δx
    global U[k] = u_ref - K*Δx̃

    # Integrate
    # global X[k + 1] = rk4(atlas, X[k], clamp.(u_ref + U[k], -atlas.torque_limits, atlas.torque_limits), h)
    global X[k + 1] = rk4(nadia, X[k], U[k], h)
end
animate!(nadia, mvis, X, Δt=h);