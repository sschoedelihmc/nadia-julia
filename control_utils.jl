using LinearAlgebra
using BlockDiagonals
using ProgressMeter

# Infinite horizon LQR solver
# function ihlqr(A, B, Q, R, Qf; max_iters = 1000, tol = 1e-8, verbose=false)
#     P = Qf
#     K = zero(B')
#     K_prev = deepcopy(K)
#     @showprogress for i = 1:max_iters
#         K = (R .+ B'*P*B) \ (B'*P*A)
#         push!(Ks, copy(K))
#         P = Q + A'P*(A - B*K)
#         push!(Ps, copy(P))
        
#         if norm(K - K_prev, 2) < tol
#             if verbose
#                 display("ihlqr converged in " * string(i) * " iterations")
#             end
#             return K, P
#         end
#         K_prev = deepcopy(K)
#     end
#     @error "ihlqr didn't converge", norm(K - (R .+ B'*P*B) \ (B'*P*A), 2)
#     return K, P
# end

function get_PD_gains(model::PinnZooModel) # Taken from NadiaHighLevelControllerParameters, getDesiredJointBehaviorForWalkingNotLoaded
    leg_p = diagm(zeros(12))
    spine_p = diagm([0; 100; 100])
    arm_p = diagm(repeat([3; 3; 3; 2.5], 2))

    leg_d = diagm(repeat([0.05; 10; 10; 15; 15; 10], 2))
    spine_d = diagm([0.05; 6; 6])
    arm_d = diagm(repeat([7.5; 4; 4; 4], 2))

    K_pd = [zeros(model.nu, 7) BlockDiagonal([leg_p, spine_p, arm_p]) zeros(model.nu, 6) BlockDiagonal([leg_d, spine_d, arm_d])]
    return K_pd
end

# Quaternion stuff

function hat(v)
    return [0 -v[3] v[2];
            v[3] 0 -v[1];
            -v[2] v[1] 0]
end

function L(q)
    s = q[1]
    v = q[2:4]
    L = [s    -v';
         v  s*I+hat(v)]
    return L
end

T = Diagonal([1; -ones(3)])

H = [zeros(1,3); I]

function G(q)
    G = L(q)*H
end

function E(q)
    E = BlockDiagonal([G(q), 1.0*I(55)])
end

function qtoQ(q)
    return H'*T*L(q)*T*L(q)*H
end

function rptoq(ϕ)
    (1/sqrt(1+ϕ'*ϕ))*[1; ϕ]
end

function qtorp(q)
    q[2:4]/q[1]
end