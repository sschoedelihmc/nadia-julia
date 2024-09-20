using Printf
using BlockDiagonals
### WARNING: quick and dirty global functions that operate on variables created in nadia_mujoco_stand.jl

function quasi_shift_foot_lift(;shift_ang = 5, dt = 0.01, tf = 3, K = zeros(model.nu, model.nx))
    # Set up standing i.c. and solve for stabilizing control
    bend_ang = 40*pi/180
    x_lin = [0; 0; 0.892; 1; zeros(5); -bend_ang; 2*bend_ang; -bend_ang; zeros(3); -bend_ang; 2*bend_ang; -bend_ang; zeros(4); 
            repeat([0; 0; 0; -pi/4], 2); zeros(model.nv)]
    u_lin = vcat(calc_continuous_eq(model, x_lin, K=K)...)
    u_lin = [u_lin[1:model.nu]; zeros(model.nc*3); u_lin[model.nu + 1:end]] 
    foot_locs = kinematics(model, x_lin)
    foot_center = [mean(foot_locs[1:3:end]), mean(foot_locs[2:3:end]), mean(foot_locs[3:3:end])]

    # Create shift and lift state
    shift_ang = -shift_ang*pi/180 # 14 gets CoM in foot center
    x_shift = [0; 0; 0.892; 1; zeros(4); shift_ang; -bend_ang; 2*bend_ang; -bend_ang; - shift_ang; zeros(1);
                shift_ang; -bend_ang; 2*bend_ang; -bend_ang; - shift_ang; zeros(3); repeat([0; 0; 0; -pi/4], 2); zeros(model.nv)]
    foot_locs = kinematics(model, x_shift)
    x_shift[1:3] = x_shift[1:3] + (foot_center - [mean(foot_locs[1:3:end]), mean(foot_locs[2:3:end]), mean(foot_locs[3:3:end])])

    u_shift = vcat(calc_continuous_eq(model, x_shift, K=K_pd, verbose = true)...);
    u_shift = [u_shift[1:model.nu]; zeros(model.nc*3); u_shift[model.nu + 1:end]]

    # Create a reference from x_lin to x_shift
    profile(t) = cos(t*pi + pi)/2 + 1/2
    X = [[x_lin for _ = 1:50]..., [(1 - t)*x_lin + t*x_shift for t in profile.(LinRange(0, 1, Int(tf/dt + 1)))]..., x_shift]
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

        # Perform a Newton step (this residual is linear in the control so this is the best you can do)
        dr_du = FiniteDiff.finite_difference_jacobian(residual, U[k - 1])
        A = (dr_du*dr_du' + 1e-11*I) # This is always singular since J_func gives a redundant jacobian
        U[k] = U[k - 1] - dr_du' * (A \ residual(U[k - 1]))
        max_err = max(max_err, norm(residual(U[k]), Inf))
    end

    # Report on errors for sanity checking
    @info @sprintf "errors: constraint pos = %1.2e constraint vel = %1.2e ctrl = %1.2e" norm(cp_viol, Inf) norm(cv_viol, Inf) max_err


    return X, U
end

# function quasi_shift_foot_lift()
#     # Set up standing i.c. and solve for stabilizing control
#     bend_ang = 40*pi/180
#     x_lin = [0; 0; 0.88978022; 1; zeros(5); -bend_ang; 2*bend_ang; -bend_ang; zeros(3); -bend_ang; 2*bend_ang; -bend_ang; zeros(model.nx - 18)]
#     u_lin = vcat(calc_continuous_eq(model, x_lin)...)

#     # Create shift and lift state
#     shift_ang = 10*pi/180 # 14 gets CoM in foot center
#     x_shift = [0; -0.1652048243975755; 0.8694956181773023; 1; zeros(4); shift_ang; -bend_ang; 2*bend_ang; -bend_ang; - shift_ang; zeros(1);
#                 shift_ang; -bend_ang; 2*bend_ang; -bend_ang; - shift_ang; zeros(model.nx - 19)]
#     lift_ang = 50*pi/180
#     x_lift = [0; -0.1652048243975755; 0.8694956181773023; 1; zeros(4); shift_ang; -lift_ang; 2*lift_ang; -lift_ang; - shift_ang; zeros(1);
#                 shift_ang; -bend_ang; 2*bend_ang; -bend_ang; - shift_ang; zeros(model.nx - 19)]
#     u_lift = vcat(calc_continuous_eq(model, x_lin, ))

#     # Create reference trajectory
#     sigmoid(t) = 1/(1 + exp(-t))
#     # profile(steps) = sigmoid.(LinRange(-7,7, steps))
#     profile(steps) = LinRange(0,1, steps)
#     dt = 0.01
#     stand_steps, shift_steps, lift_steps = 50, 500, 500
#     X_ref = [[copy(x_lin) for _ = 1:stand_steps]...,
#         [(1-t)*x_lin + t*x_shift for t in profile(shift_steps)]...,
#         [copy(x_shift) for _ = 1:shift_steps]...,
#         [(1-t)*x_shift + t*x_lift for t in profile(lift_steps)]...,
#         [(1-t)*x_lift + t*x_shift for t in profile(lift_steps)]...,
#         [copy(x_shift) for _ = 1:shift_steps]...,
#         [(1-t)*x_shift + t*x_lin for t in profile(shift_steps)]...,
#         [copy(x_lin) for _ = 1:stand_steps]...]
#     # solve_ref_velocity_first_order(model, X_ref, dt);
#     plot(hcat(X_ref...)')
#     CMode_ref = [kinematics(model, x)[3:3:end] .< 0.002 for x in X_ref]
#     U_ref = [copy(u_lin) for _ = 1:length(X_ref)]
#     for k = 1:length(U_ref)
#         for foot = 1:model.nc
#             if !CMode_ref[k][foot]
#                 U_ref[k][1:6] .= 0
#                 U_ref[k][model.nu + 3*(foot - 1) .+ (1:3)] .= 0
#             end
#         end
#     end
#     ref = LinearizedQuadRef(model, X_ref, [u_lin for _ = 1:length(X_ref)], x_lin, u_lin, dt, nc = model.nc, CMode_ref = CMode_ref, periodic=true)
#     return ref
# end

function playback_ref()
    for x in X_ref
        data.x = copy(x)
        set_data!(model, intf, data)
        sleep(dt)
    end
end