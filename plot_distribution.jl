#! /usr/bin/env -S julia --color=yes --startup-file=no
# -*- coding: utf-8 -*-
# Visualize the distribution function f_s(v₁, v₂) at different time steps.

include("MantisWrappers.jl")
using .MantisWrappers
using GLMakie
using Random

include("parameters.jl")

function evaluate_on_grid(coeffs, v_grid)
    field = build_field(coeffs)
    nv = length(v_grid)
    F = zeros(nv, nv)
    for (j, v2) in enumerate(v_grid)
        for (i, v1) in enumerate(v_grid)
            loc = locate_particle(v1, v2)
            isnothing(loc) && continue
            fv, _ = evaluate(field, loc)
            F[i, j] = fv[1][1]
        end
    end
    return F
end

function main()
    Random.seed!(42)
    v_particles = zeros(N_PARTICLES, 2)
    v_particles[:, 1] .= σ₁ * randn(N_PARTICLES)
    v_particles[:, 2] .= σ₂ * randn(N_PARTICLES)
    w_particles = fill(1.0 / N_PARTICLES, N_PARTICLES)
    f_coeffs = zeros(n_dofs)

    l2_project!(f_coeffs, v_particles, w_particles)

    snapshot_steps = [0, N_STEPS ÷ 4, N_STEPS ÷ 2, N_STEPS]
    coeff_snapshots = Dict{Int,Vector{Float64}}()
    coeff_snapshots[0] = copy(f_coeffs)

    r_vec = zeros(n_dofs)
    L_vec = zeros(n_dofs)
    G = zeros(N_PARTICLES, 2)
    dot_v = zeros(N_PARTICLES, 2)

    for step in 1:N_STEPS
        f_s = build_field(f_coeffs)
        compute_r!(r_vec, f_s)
        L_vec .= M_lu \ r_vec
        compute_G!(G, v_particles, L_vec)
        compute_collision!(dot_v, v_particles, w_particles, G)
        v_particles .+= DT .* dot_v

        l2_project!(f_coeffs, v_particles, w_particles)

        if step in snapshot_steps
            coeff_snapshots[step] = copy(f_coeffs)
        end

        step % 25 == 0 && println("Step $step/$N_STEPS")
    end

    # Evaluate f_s on a uniform grid for visualization
    nv = 201
    v_grid = LinRange(V_MIN + 1e-10, V_MAX - 1e-10, nv)

    snap_keys = sort(collect(keys(coeff_snapshots)))
    fig = Figure(; size=(1200, 900))

    for (idx, s) in enumerate(snap_keys)
        row, col = fldmod1(idx, 2)
        ax = Axis(fig[row, col];
            title="t = $(round(s * DT; digits=4))",
            xlabel="v₁", ylabel="v₂", aspect=DataAspect())
        F = evaluate_on_grid(coeff_snapshots[s], v_grid)
        heatmap!(ax, collect(v_grid), collect(v_grid), F';
            colormap=:inferno)
        xlims!(ax, V_MIN, V_MAX)
        ylims!(ax, V_MIN, V_MAX)
    end

    save("distribution_fs.png", fig)
    println("Saved distribution_fs.png")
end

main()
