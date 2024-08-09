using Pkg; Pkg.activate(@__DIR__)

using JLD2

using MeshCat
using MeshCatMechanisms
using RigidBodyDynamics
using RigidBodyDynamics.PDControl

using StaticArrays
using SparseArrays

import ForwardDiff

include("nadia_robot.jl")
include("control_utils.jl")

##

nadia = Nadia();
vis = Visualizer();
render(vis)

##

# Initialize visualizer with robot meshes
mvis = init_visualizer(nadia, vis)

##

# Load equilibrium reference controls and state
u_ref = load_object("nadia_balance_u_ref.jld2")

# u_ref torques should be in the following order:
# "LEFT_HIP_Z"
# "RIGHT_HIP_Z"
# "SPINE_Z"
# "LEFT_HIP_X"
# "RIGHT_HIP_X"
# "SPINE_X"
# "LEFT_HIP_Y"
# "RIGHT_HIP_Y"
# "SPINE_Y"
# "LEFT_KNEE_Y"
# "RIGHT_KNEE_Y"
# "LEFT_SHOULDER_Y"
# "RIGHT_SHOULDER_Y"
# "LEFT_ANKLE_Y"
# "RIGHT_ANKLE_Y"
# "LEFT_SHOULDER_X"
# "RIGHT_SHOULDER_X"
# "LEFT_ANKLE_X"
# "RIGHT_ANKLE_X"
# "LEFT_SHOULDER_Z"
# "RIGHT_SHOULDER_Z"
# "LEFT_ELBOW_Y"
# "RIGHT_ELBOW_Y"

x_ref = load_object("nadia_balance_x_ref.jld2")
# 4 variables for pelvis orientation (w, x, y, z quaternion)
# 3 variables for pelvis position in world frame
# LEFT_HIP_Z
# RIGHT_HIP_Z
# SPINE_Z
# LEFT_HIP_X
# RIGHT_HIP_X
# SPINE_X
# LEFT_HIP_Y
# RIGHT_HIP_Y
# SPINE_Y
# LEFT_KNEE_Y
# RIGHT_KNEE_Y
# LEFT_SHOULDER_Y
# RIGHT_SHOULDER_Y
# LEFT_ANKLE_Y
# RIGHT_ANKLE_Y
# LEFT_SHOULDER_X
# RIGHT_SHOULDER_X
# LEFT_ANKLE_X
# RIGHT_ANKLE_X
# LEFT_SHOULDER_Z
# RIGHT_SHOULDER_Z
# LEFT_ELBOW_Y
# RIGHT_ELBOW_Y
# 29 variables for reference velocities in the same order 
#   as config but with three variables for orientation 
#   (generally all zeros for an equilibrium position)

##

# Set pose in visualizer
set_configuration!(mvis, x_ref[1:nadia.nq])

##

# Calculate linearized dynamics
dt = 1e-3
ADyn = ForwardDiff.jacobian(x_ -> rk4(nadia, x_, u_ref, dt), x_ref)
BDyn = ForwardDiff.jacobian(u_ -> rk4(nadia, x_ref, u_, dt), u_ref)

# Reduce quaternion representation to a form we can do math with
ADynReduced = E(x_ref[1:4])' * ADyn * E(x_ref[1:4])
BDynReduced = E(x_ref[1:4])' * BDyn

##

# Compute IHLQR optimal feedback gain matrix Kinf

# Tune these if you want
Q = spdiagm([repeat([5e2], 6); repeat([1e-3, 1e-3, 1e3], 3); 1e2; 1e2; repeat([5e1; 5e1; 1e3; 1e3], 2); repeat([1e2], 4);
                repeat([1e1], 6); repeat([1e1, 1e1, 1e1], 3); 1e2; 1e2; repeat([1e1; 1e1; 1e1; 1e1], 2); repeat([1e1], 4)]);
R = spdiagm(1e-2*ones(size(BDynReduced)[2]));

Kinf, Qf = ihlqr(ADynReduced, BDynReduced, Q, R, Q; max_iters = 200000, verbose=true);

# Check eigenvalues of system
# eigvals(ADynReduced - BDynReduced*Kinf)

##

current_state = [
    0.9851315002584348
0.14117015201703584
-0.02084838312153866
-0.0956674463925425
-0.044234901400114726
-0.166853379121665
1.0736214580783963
0.06694415402125188
-0.1926155777175038
0.28678055262946034
0.15870995262179505
-0.12119899327305118
-0.10648237906165749
-0.40569136542906614
-0.3548134013645995
0.07257892275662743
1.5006102322161259
0.7695499208515111
0.22737074287654313
0.6387032621664348
0.22924432722452515
-0.3482433992429211
0.3517578062494244
-0.2564253818171134
-0.4345912506093079
-0.15832353750235753
0.7415924615165517
-0.6079673776970418
-1.2967592918332347
-1.4721197341678536
-0.19744414650531628
-0.07532273114738434
-0.055112892456926674
-0.00885431023015928
-0.02071632039215511
-0.07091842108185788
-0.10555883429034962
-0.034956953575896635
0.04664836062409165
0.2520276284263501
0.30821284753292216
0.26598046428792793
0.12698300174068267
-0.02118708310758901
0.08086011046028284
-0.00605600621700032
0.03900001119582415
-0.13218407172794233
-0.05004694474757597
0.21573194241708243
0.11844516663473713
-0.1284932595640506
-0.06023061921101689
1.0406338307526366
-0.712232690402183
0.2363488069029041
-0.04299349182233441
0.2869528819155084
0.2841893400202096
]

reference_state = [
    0.9851812850604109
    0.14091286404085285
    -0.021356269245195184
    -0.095421748536162
    -0.04426391998996947
    -0.1668505364459041
    1.0334898178372895
    0.06781608404733115
    -0.1922539118626478
    0.2861350253811403
    0.15898265237693734
    -0.12064158590993693
    -0.1057141650799965
    -0.4044177896371911
    -0.3535655955195078
    0.07352471380726427
    1.500022076915403
    0.7689918571683205
    0.2271611304501065
    0.6387021266726293
    0.22893847881329563
    -0.3460980826665025
    0.35167329835530214
    -0.2566674753190524
    -0.43626403944917924
    -0.15679667582212356
    0.7430339207647649
    -0.608843261553189
    -1.2994261459930767
    -1.475040588733499
    0.0
    0.0
    0.0
    0.0
    0.0
    0.0
    0.0
    0.0
    0.0
    0.0
    0.0
    0.0
    0.0
    0.0
    0.0
    0.0
    0.0
    0.0
    0.0
    0.0
    0.0
    0.0
    0.0
    0.0
    0.0
    0.0
    0.0
    0.0
    0.0
]

delta_state = [
    2.0678008649781195E-4
4.892256609257301E-4
-3.2386994498956106E-4
2.901858985474709E-5
-2.8426757608945863E-6
0.040131640241106714
-8.719300260792706E-4
-3.616658548559748E-4
6.455272483200658E-4
-2.7269975514229716E-4
-5.574073631142568E-4
-7.682139816609923E-4
-0.0012735757918750368
-0.0012478058450917096
-9.457910506368433E-4
5.881553007229812E-4
5.580636831905572E-4
2.0961242643663391E-4
1.1354938055374575E-6
3.058484112295201E-4
-0.0021453165764185878
8.450789412223214E-5
2.4209350193898915E-4
0.0016727888398713198
-0.0015268616802339707
-0.0014414592482132615
8.758838561472304E-4
0.002666854159842025
0.002920854565645392
-0.19744414650531628
-0.07532273114738434
-0.055112892456926674
-0.00885431023015928
-0.02071632039215511
-0.07091842108185788
-0.10555883429034962
-0.034956953575896635
0.04664836062409165
0.2520276284263501
0.30821284753292216
0.26598046428792793
0.12698300174068267
-0.02118708310758901
0.08086011046028284
-0.00605600621700032
0.03900001119582415
-0.13218407172794233
-0.05004694474757597
0.21573194241708243
0.11844516663473713
-0.1284932595640506
-0.06023061921101689
1.0406338307526366
-0.712232690402183
0.2363488069029041
-0.04299349182233441
0.2869528819155084
0.2841893400202096
]


delta_torques = [
    1.2818345016646384
2.08986802688117
-1.6338809508803305
-4.548039157079866
-3.0051417794648896
-2.7645427349214686
-1.3721569704704952
-2.5483850143923723
-2.164339266017388
-0.8069472822924261
8.557866766639144
2.971366976229026
0.547771143794325
-3.3606826243047103
-3.7072877762718712
2.191330480565003
2.1353134405604175
-5.372906559460533
2.1019387796650104
-2.5120040736371645
0.5100121046596674
-3.042917725426771
-3.1430710584054253
]

reference_torques = [
    -1.529996146733067       # "LEFT_HIP_Z"
    9.964441654400309        # "RIGHT_HIP_Z"
    2.4214516921032456       # "SPINE_Z"
    30.725284382117          # "LEFT_HIP_X"
    -102.44994521205273      # "RIGHT_HIP_X"
    -10.165094068114897      # "SPINE_X"
    1.3888702260931227       # "LEFT_HIP_Y"
    30.57276337319791        # "RIGHT_HIP_Y"
    -5.786704258930225       # "SPINE_Y"
    18.379170992769396       # "LEFT_KNEE_Y"
    -87.7703566391235        # "RIGHT_KNEE_Y"
    2.673614241444654        # "LEFT_SHOULDER_Y"
    3.90078753582609         # "RIGHT_SHOULDER_Y"
    0.6569298958706897       # "LEFT_ANKLE_Y"
    31.358889860182014       # "RIGHT_ANKLE_Y"
    7.695142597693708        # "LEFT_SHOULDER_X"
    -4.036609656052576       # "RIGHT_SHOULDER_X"
    0.22082314816501622      # "LEFT_ANKLE_X"
    -0.0021063087760480495   # "RIGHT_ANKLE_X"
    0.4149513856694272       # "LEFT_SHOULDER_Z"
    -0.2869521910778749      # "RIGHT_SHOULDER_Z"
    -0.532071050841635       # "LEFT_ELBOW_Y"
    -0.5439629977475727      # "RIGHT_ELBOW_Y"
]

torques = [
    -0.24816164506842853
    12.05430968128148
    0.7875707412229151
    26.177245225037133
    -105.45508699151762
    -12.929636803036365
    0.016713255622627488
    28.024378358805535
    -7.951043524947613
    17.57222371047697
    -79.21248987248435
    5.6449812176736796
    4.448558679620415
    -2.7037527284340204
    27.65160208391014
    9.88647307825871
    -1.9012962154921587
    -5.152083411295517
    2.099832470888962
    -2.097052687967737
    0.2230599135817925
    -3.5749887762684063
    -3.687034056152998
]

##

maximum(abs.(qtorp(L(reference_state[1:4])'*current_state[1:4]) - delta_state[1:3]))

Δx = [qtorp(L(reference_state[1:4])'*current_state[1:4]); current_state[5:end] - reference_state[5:end]]
maximum(abs.(Δx[1:3] - delta_state[1:3]))
maximum(abs.(Δx[4:end] - delta_state[4:end]))

Δu = (-Kinf * Δx)
maximum(abs.(Δu - delta_torques))

maximum(abs.(Δu + u_ref - torques))

##

simulation_time_step = 0.001
end_time = 3.0

N = Int(floor(end_time/simulation_time_step))
X = [zeros(length(x_ref)) for _ = 1:N];
U = [zeros(length(u_ref)) for _ = 1:N];
X[1] = deepcopy(x_ref);
X[1][nadia.nq + 5] = 1.3; # Perturb i.c.

# Run simulation
for k = 1:N - 1
    # Get error
    global Δx̃ = [qtorp(L(x_ref[1:4])'*X[k][1:4]); X[k][5:end] - x_ref[5:end]]
  
    # add some noise
    Δx̃ += 0.1 * randn(length(Δx̃))

    # Compute controls for this time step
    global U[k] = u_ref - Kinf*Δx̃

    # Integrate
    global X[k + 1] = rk4(nadia, X[k], U[k], simulation_time_step; gains=nadia.baumgarte_gains)
end
anim = animate(nadia, mvis, X; Δt=simulation_time_step, frames_to_skip=50);
setanimation!(mvis, anim)

##

Kinf_str = ""
for gain in Kinf
    Kinf_str *= string(gain) * ", "
end

clipboard(Kinf_str)