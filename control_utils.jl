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



function linesearch(z::Vector, Δz::Vector, merit_fx::Function;
    max_ls_iters = 10)::Float64 # optional argument with a default
    ϕ0 = merit_fx(z)
    α = 1.0 
    for i = 1:max_ls_iters
        if merit_fx(z + α*Δz) < ϕ0 
            return α
        else
            α = α/2
        end
    end
    error("linesearch failed")
end

function newtons_method(z0::Vector, res_fx::Function, res_jac_fx::Function, merit_fx::Function;
        tol = 1e-10, max_iters = 50, verbose = false)::Vector{Vector{Float64}}

    # - z0, initial guess 
    # - res_fx, residual function 
    # - res_jac_fx, Jacobian of residual function wrt z 
    # - merit_fx, merit function for use in linesearch 

    # optional arguments 
    # - tol, tolerance for convergence. Return when norm(residual)<tol 
    # - max iter, max # of iterations 
    # - verbose, bool telling the function to output information at each iteration

    # return a vector of vectors containing the iterates 
    # the last vector in this vector of vectors should be the approx. solution 

    # return the history of guesses as a vector
    Z = [zeros(length(z0)) for i = 1:max_iters]
    Z[1] = z0 

    for i = 1:(max_iters - 1)

        # evaluate current residual 
        r = res_fx(Z[i])
        norm_r = norm(r)
        if verbose 
            print("iter: $i    |r|: $norm_r   ")
        end

        # check convergence with norm of residual < tol 
        if norm_r < tol
            return Z[1:i]
        end

        # caculate Newton step (don't forget the negative sign)
        J = res_jac_fx(Z[i])
        Δz = -J\r 

        # linesearch and update z 
        α = linesearch(Z[i], Δz, merit_fx)
        Z[i+1] = Z[i] + α*Δz
        if verbose
            print("α: $α \n")
        end

    end
    error("Newton's method did not converge")
end





function gen_sparse_mpc_qp(Ad, Bd, Q, R, Qf, horizon; A_add = Nothing, l_add = Nothing, u_add = Nothing)
    nx, nu = size(Ad, 1), size(Bd, 2)

    # Cost
    H = blockdiag([blockdiag(R, Q) for k = 1:horizon - 1]..., R, sparse(Qf))
    g = zeros(size(H, 1))

    # Dynamics constraint
    A = kron(I(horizon), [Bd -I])
    A[nx + 1:end, nu + 1:end - nx] += kron(I(horizon - 1), [Ad zeros(nx, nu)])
    l = zeros(size(A, 1))
    u = zeros(size(A, 1))

    # If there are additional constraints, append them
    if A_add != Nothing
        A = [A; A_add]
        l = [l; l_add]
        u = [u; u_add]
    end

    return Matrix(H), g, A, l, u
end

function gen_condensed_mpc_qp(Ad, Bd, Q, R, Qf, horizon, A_add, l_add, u_add, K = Nothing)
    nx, nu = size(Ad, 1), size(Bd, 2)

    # Gain matrix defaults to zero
    if K === Nothing
        K = zeros(nu, nx)
    end

    # Get the sparse problem (without additional constraints)
    H_sp, g_sp, _ = gen_sparse_mpc_qp(Ad, Bd, Q, R, Qf, horizon)

    # Define F and G such that z = Fz̄ + Gx0 to transform the sparse problem into the dense one
    # z = [u0; x1; u1; x2; u2; ...]
    # z̄ = [Δu0; Δu1; Δu2; ...] where uk = -Kxk + Δuk
    F = kron(Diagonal(I, horizon), [I; Bd])
    for k = 1:horizon - 1
        F += kron(diagm(-k => ones(horizon - k)), 
                [ -K*(Ad - Bd*K)^(k - 1)*Bd;
                    (Ad - Bd*K)^k*Bd        ])
    end

    G = vcat([[-K*(Ad - Bd*K)^(k - 1); (Ad - Bd*K)^k] for k = 1:horizon]...)

    # Convert the sparse problem cost
    H = F'*H_sp*F
    g_x0 = F'*H_sp*G # Initial condition becomes part of cost
    g = g_x0*zeros(nx) + F'*g_sp

    # Add additional constraints if they exist
    A = A_add*F
    lu_x0 = -A_add*G # Initial condition is also part of the constraints

    return H, g, A, l_add, u_add, g_x0, lu_x0
end