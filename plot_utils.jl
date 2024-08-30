
# function generate_plots()
body_inds = [1:6]
leg_inds = [6 .+ (1:12)]
spine_inds = [6 + 12 .+ (1:3)]
arm_inds = [6 + 12 + 3 .+ (1:8)]

# Plot position actual vs reference
function plot_tracking() #uses globals
    ΔX_ref = [state_error(model, x, x_lin) for x in X_d if norm(x) > 1e-10];
    ΔX_act = [state_error(model, x, x_lin) for x in X[1:length(ΔX_ref)]];
    t = (0:simulation_time_step:((N-1)*simulation_time_step))[1:length(ΔX_ref)]
    plots = []; for r = 1:6, c = 1:5
        i = (c - 1)*6 + r
        if i > model.nv; break; end
        p = plot(t, [x[i] for x in ΔX_act], label="")
        p = plot(p,t,  [x[i] for x in ΔX_ref], label="", title=model_fixed.orders[:mujoco].vel_names[i])
        push!(plots, p)
    end

    p = plot(plots..., layout=(6, 5));
    savefig("position_tracking.png")

    plots = []; for r = 1:6, c = 1:5
        i = (c - 1)*6 + r + model.nv
        if i > model.nv*2; break; end
        p = plot(t, [x[i] for x in ΔX_act], label="")
        p = plot(p,t,  [x[i] for x in ΔX_ref], label="", title=model_fixed.orders[:mujoco].vel_names[i - model.nv])
        push!(plots, p)
    end

    p = plot(plots..., layout=(6, 5));
    savefig("velocity_tracking.png")

    t = (0:simulation_time_step:((N-1)*simulation_time_step))[1:length(ΔX_ref)-1]
    plots = []; for r = 1:6, c = 1:4
        i = (c - 1)*6 + r
        if i > model.nu; break; end
        p = plot(t, [x[i] for x in U[1:length(ΔX_ref)-1]], label="", title=model_fixed.orders[:mujoco].torque_names[i])
        p = plot(p,t,  [x[i]*0 + u_lin[i] for x in U[1:length(ΔX_ref)-1]], label="")
        push!(plots, p)
    end

    p = plot(plots..., layout=(6, 4));
    savefig("controls.png")
end
# end