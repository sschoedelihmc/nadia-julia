using LinearAlgebra
using BlockDiagonals

# Infinite horizon LQR solver
function ihlqr(A, B, Q, R, Qf; max_iters = 1000, tol = 1e-8, verbose=false)
    P = Qf
    K = zero(B')
    K_prev = deepcopy(K)
    for i = 1:max_iters
        K = (R .+ B'*P*B) \ (B'*P*A)
        P = Q + A'P*(A - B*K)
        if norm(K - K_prev, 2) < tol
            if verbose
                display("ihlqr converged in " * string(i) * " iterations")
            end
            return K, P
        end
        K_prev = deepcopy(K)
    end
    @error "ihlqr didn't converge", norm(K - (R .+ B'*P*B) \ (B'*P*A), 2)
    return K, P
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

function qtoQ(q)
    return H'*T*L(q)*T*L(q)*H
end

function G(q)
    G = L(q)*H
end

function rptoq(ϕ)
    (1/sqrt(1+ϕ'*ϕ))*[1; ϕ]
end

function qtorp(q)
    q[2:4]/q[1]
end

function E(q)
    E = BlockDiagonal([G(q), 1.0*I(55)])
end