### WARNING: quick and dirty global functions that operate on variables created in nadia_mujoco_stand.jl

function quasi__shift_foot_lift()
    # Set up standing i.c. and solve for stabilizing control
    bend_ang = 40*pi/180
    x_lin = [0; 0; 0.88978022; 1; zeros(5); -bend_ang; 2*bend_ang; -bend_ang; zeros(3); -bend_ang; 2*bend_ang; -bend_ang; zeros(model.nx - 18)]
    u_lin = vcat(calc_continuous_eq(model, x_lin)...)

    # Create shift and lift state
    shift_ang = 10*pi/180 # 14 gets CoM in foot center
    x_shift = [0; -0.1652048243975755; 0.8694956181773023; 1; zeros(4); shift_ang; -bend_ang; 2*bend_ang; -bend_ang; - shift_ang; zeros(1);
                shift_ang; -bend_ang; 2*bend_ang; -bend_ang; - shift_ang; zeros(model.nx - 19)]
    lift_ang = 50*pi/180
    x_lift = [0; -0.1652048243975755; 0.8694956181773023; 1; zeros(4); shift_ang; -lift_ang; 2*lift_ang; -lift_ang; - shift_ang; zeros(1);
                shift_ang; -bend_ang; 2*bend_ang; -bend_ang; - shift_ang; zeros(model.nx - 19)]
    u_lift = vcat(calc_continuous_eq(model, x_lin, ))

    # Create reference trajectory
    sigmoid(t) = 1/(1 + exp(-t))
    # profile(steps) = sigmoid.(LinRange(-7,7, steps))
    profile(steps) = LinRange(0,1, steps)
    dt = 0.01
    stand_steps, shift_steps, lift_steps = 50, 500, 500
    X_ref = [[copy(x_lin) for _ = 1:stand_steps]...,
        [(1-t)*x_lin + t*x_shift for t in profile(shift_steps)]...,
        [copy(x_shift) for _ = 1:shift_steps]...,
        [(1-t)*x_shift + t*x_lift for t in profile(lift_steps)]...,
        [(1-t)*x_lift + t*x_shift for t in profile(lift_steps)]...,
        [copy(x_shift) for _ = 1:shift_steps]...,
        [(1-t)*x_shift + t*x_lin for t in profile(shift_steps)]...,
        [copy(x_lin) for _ = 1:stand_steps]...]
    # solve_ref_velocity_first_order(model, X_ref, dt);
    plot(hcat(X_ref...)')
    CMode_ref = [kinematics(model, x)[3:3:end] .< 0.002 for x in X_ref]
    U_ref = [copy(u_lin) for _ = 1:length(X_ref)]
    for k = 1:length(U_ref)
        for foot = 1:model.nc
            if !CMode_ref[k][foot]
                U_ref[k][1:6] .= 0
                U_ref[k][model.nu + 3*(foot - 1) .+ (1:3)] .= 0
            end
        end
    end
    ref = LinearizedQuadRef(model, X_ref, [u_lin for _ = 1:length(X_ref)], x_lin, u_lin, dt, nc = model.nc, CMode_ref = CMode_ref, periodic=true)
    return ref
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