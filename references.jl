using Printf
using BlockDiagonals
### WARNING: quick and dirty global functions that operate on variables created in nadia_mujoco_stand.jl

function shifted_one_foot_lift(;shift_ang = 10, lift_ang = 1, dt = 0.01, tf = 0.5, K = zeros(model.nu, model.nx))
    # Set up standing i.c. and solve for stabilizing control
    bend_ang = 40*pi/180
    x_lin = [0; 0; 0.892; 1; zeros(5); -bend_ang; 2*bend_ang; -bend_ang; zeros(3); -bend_ang; 2*bend_ang; -bend_ang; zeros(4); 
            repeat([0; 0; 0; -pi/4], 2); zeros(model.nv)]
    u_lin = vcat(calc_continuous_eq(model, x_lin, K=K)...)
    u_lin = [u_lin[1:model.nu]; zeros(model.nc*3); u_lin[model.nu + 1:end]] 
    foot_locs = kinematics(model, x_lin)
    foot_center = [mean(foot_locs[1:3:end]), mean(foot_locs[2:3:end]), mean(foot_locs[3:3:end])]

    # Create shift state
    bend_ang = 40*pi/180
    shift_ang = -shift_ang*pi/180 # 14 gets CoM in foot center
    x_shift = [0; 0; 0.892; 1; zeros(4); shift_ang; -bend_ang; 2*bend_ang; -bend_ang; - shift_ang; zeros(1);
                shift_ang; -bend_ang; 2*bend_ang; -bend_ang; - shift_ang; zeros(3); repeat([0; 0; 0; -pi/4], 2); zeros(model.nv)]
    foot_locs = kinematics(model, x_shift)
    x_shift[1:3] = x_shift[1:3] + (foot_center - [mean(foot_locs[1:3:end]), mean(foot_locs[2:3:end]), mean(foot_locs[3:3:end])])
    u_shift = vcat(calc_continuous_eq(model, x_shift, K=K_pd, verbose = true)...);
    u_shift = [u_shift[1:model.nu]; zeros(model.nc*3); u_shift[model.nu + 1:end]]

    # Create lift state (left)
    lift_ang = bend_ang + lift_ang*pi/180
    x_lift_left = copy(x_shift);
    x_lift_left[16:18] .= [-lift_ang; 2*lift_ang; -lift_ang]

    # Create lift state (right)
    x_lift_right = copy(x_shift);
    x_lift_right[10:12] .= [-lift_ang; 2*lift_ang; -lift_ang]

    # Create a reference from shift -> lift -> shift
    profile(t) = (-cos(t*2*pi) + 1)/2
    X = [[x_shift for _ = 1:Int(0.2/dt)]..., [(1 - t)*x_shift + t*x_lift_left for t in profile.(LinRange(0, 1, Int(tf/dt + 1)))]..., [x_shift for _ in 1:Int(1.0/dt)]...]
    X = solve_ref_velocity_first_order(model, X, dt)

    # Contact modes
    CMode = [(kinematics(model, x)[3:3:end] - kinematics(model, x_lin)[3:3:end]) .< 1e-5 for x in X]

    # Check against jacobians (todo fix to check only when active using contact modes)
    cp_viol = [(kinematics(model, x) - kinematics(model, x_lin)).*kron(mode, ones(3)) for (x, mode) in zip(X, CMode)]
    cv_viol = [kinematics_velocity(model, x).*kron(mode, ones(3)) for (x, mode) in zip(X, CMode)]

    # Solve for controls (first order)
    U = [zeros(model.nu + model.nc*3*2) for _ = 1:length(X) - 1]
    B = B_func(model)
    J_func_both(model, x) = BlockDiagonal([kinematics_jacobian(model, x)[:, 1:model.nq], kinematics_jacobian(model, x)[:, 1:model.nq]*velocity_kinematics(model, x)])
    J_func_left(model, x) = BlockDiagonal([kinematics_jacobian(model, x)[1:12, 1:model.nq], kinematics_jacobian(model, x)[1:12, 1:model.nq]*velocity_kinematics(model, x)])
    J_func_right(model, x) = BlockDiagonal([kinematics_jacobian(model, x)[13:end, 1:model.nq], kinematics_jacobian(model, x)[13:end, 1:model.nq]*velocity_kinematics(model, x)])
    U[1] = copy(u_shift)
    max_err = 0
    for k = 2:length(X) - 1
        xk = X[k]
        x_next = X[k + 1]
        Δx_d = state_error(model, x_next, xk)

        λ_inds, J_func = model.nu .+ (1:48), J_func_both
        if CMode[k][1] == 1 && CMode[k][5] == 0 # No forces on right foot
            λ_inds = model.nu .+ [1:12; 25:36]
            J_func = J_func_left
        elseif CMode[k][1] == 0 && CMode[k][5] == 1 # No forces on left foot
            λ_inds = model.nu .+ [13:24; 37:48]
            J_func = J_func_right
        end

        residual(u) = 
            Δx_d - dt*error_jacobian_T(model, xk)*continuous_dynamics(model, xk, u[1:model.nu], J_func=J_func, λ = u[model.nu + 1:end], K=K)

        # Perform a Newton step on the least squares problen (this residual is linear in the control so this is the best you can do)
        u_guess = [U[k-1][1:model.nu]; U[k-1][λ_inds]]
        dr_du = FiniteDiff.finite_difference_jacobian(residual, u_guess)

        A = (dr_du*dr_du' + 1e-11*I) # This is always singular since J_func gives a redundant jacobian
        u_guess = u_guess - dr_du' * (A \ residual(u_guess))

        U[k][1:model.nu] = u_guess[1:model.nu]
        U[k][λ_inds] = u_guess[model.nu + 1:end]
        max_err = max(max_err, norm(residual(u_guess), Inf))
    end

    # Report on errors for sanity checking
    @info @sprintf "errors: constraint pos = %1.2e constraint vel = %1.2e ctrl = %1.2e" norm(cp_viol, Inf) norm(cv_viol, Inf) max_err

    return X, U, CMode
end

function quasi_shift_foot_lift(;shift_ang = 5, dt = 0.01, tf = 3, K = zeros(model.nu, model.nx))
    # Set up standing i.c. and solve for stabilizing control
    bend_ang = 40*pi/180
    x_lin = [0; 0; 0.892; 1; zeros(5); -bend_ang; 2*bend_ang; -bend_ang; zeros(3); -bend_ang; 2*bend_ang; -bend_ang; zeros(4); 
            repeat([0; 0; 0; -pi/4], 2); zeros(model.nv)]
    u_lin = vcat(calc_continuous_eq(model, x_lin, K=K)...)
    u_lin = [u_lin[1:model.nu]; zeros(model.nc*3); u_lin[model.nu + 1:end]] 
    foot_locs = kinematics(model, x_lin)
    foot_center = [mean(foot_locs[1:3:end]), mean(foot_locs[2:3:end]), mean(foot_locs[3:3:end])]

    # Create shift state
    shift_ang = -shift_ang*pi/180 # 14 gets CoM in foot center
    x_shift = [0; 0; 0.892; 1; zeros(4); shift_ang; -bend_ang; 2*bend_ang; -bend_ang; - shift_ang; zeros(1);
                shift_ang; -bend_ang; 2*bend_ang; -bend_ang; - shift_ang; zeros(3); repeat([0; 0; 0; -pi/4], 2); zeros(model.nv)]
    foot_locs = kinematics(model, x_shift)
    x_shift[1:3] = x_shift[1:3] + (foot_center - [mean(foot_locs[1:3:end]), mean(foot_locs[2:3:end]), mean(foot_locs[3:3:end])])

    u_shift = vcat(calc_continuous_eq(model, x_shift, K=K_pd, verbose = true)...);
    u_shift = [u_shift[1:model.nu]; zeros(model.nc*3); u_shift[model.nu + 1:end]]

    # Create a reference from x_lin to x_shift
    profile(t) = cos(t*pi + pi)/2 + 1/2
    X = [[x_lin for _ = 1:Int(0.2/dt)]..., [(1 - t)*x_lin + t*x_shift for t in profile.(LinRange(0, 1, Int(tf/dt + 1)))]..., x_shift]
    X = solve_ref_velocity_first_order(model, X, dt)

    # Check against jacobians
    cp_viol = [kinematics(model, x) - kinematics(model, x_lin) for x in X]
    cv_viol = [kinematics_velocity(model, x) for x in X]

    # Solve for controls (first order)
    U = [zeros(model.nu + model.nc*3*2) for _ = 1:length(X) - 1]
    B = B_func(model)
    J_func(model, x) = BlockDiagonal([kinematics_jacobian(model, x)[:, 1:model.nq], kinematics_jacobian(model, x)[:, 1:model.nq]*velocity_kinematics(model, x)])
    U[1] = copy(u_lin)
    max_err = 0
    for k = 2:length(X) - 1
        xk = X[k]
        x_next = X[k + 1]
        Δx_d = state_error(model, x_next, xk)

        residual(u) = 
            Δx_d - dt*error_jacobian_T(model, xk)*continuous_dynamics(model, xk, u[1:model.nu], J_func=J_func, λ = u[model.nu + 1:end], K=K)

        # Perform a Newton step on the least squares problen (this residual is linear in the control so this is the best you can do)
        dr_du = FiniteDiff.finite_difference_jacobian(residual, U[k - 1])

        A = (dr_du*dr_du' + 1e-11*I) # This is always singular since J_func gives a redundant jacobian
        U[k] = U[k - 1] - dr_du' * (A \ residual(U[k - 1]))
        max_err = max(max_err, norm(residual(U[k]), Inf))
    end

    # Report on errors for sanity checking
    @info @sprintf "errors: constraint pos = %1.2e constraint vel = %1.2e ctrl = %1.2e" norm(cp_viol, Inf) norm(cv_viol, Inf) max_err


    return X, U
end

function playback_ref(X_ref, dt)
    for x in X_ref
        data.x = copy(x)
        set_data!(model, intf, data)
        sleep(dt)
    end
end