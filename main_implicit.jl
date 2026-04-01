#! /usr/bin/env -S julia --color=yes --startup-file=no
# -*- coding: utf-8 -*-
#
# Implicit timestepping for Landau collision using the discrete gradient method.
#
# The discrete gradient (Gonzalez midpoint, Eq 58)

include("MantisWrappers.jl")
using .MantisWrappers
using GLMakie
using Random

include("parameters.jl")
println("V ∈ [$V_MIN, $V_MAX],  p=$P_DEG, k=$K_REG,  $N_ELEM×$N_ELEM elements")
println("N_particles=$N_PARTICLES,  σ₁=$σ₁, σ₂=$σ₂,  Δt=$DT,  N_steps=$N_STEPS")

const MAX_PICARD = 5
const PICARD_TOL = 1e-10

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

    # Initialize particles from an anisotropic Maxwellian
    Random.seed!(42)
    v_particles = zeros(N_PARTICLES, 2)
    v_particles[:, 1] .= σ₁ * randn(N_PARTICLES)
    v_particles[:, 2] .= σ₂ * randn(N_PARTICLES)
    w_particles = fill(1.0 / N_PARTICLES, N_PARTICLES)
    f_coeffs = zeros(n_dofs)

    l2_project!(f_coeffs, v_particles, w_particles)
    f_s = build_field(f_coeffs)

    entropy_history = Float64[]
    push!(entropy_history, compute_entropy(f_s))
    println("Initial entropy S_h = $(entropy_history[end])")

    # Preallocated buffers
    r_vec = zeros(n_dofs)
    L_vec = zeros(n_dofs)
    G = zeros(N_PARTICLES, 2)         # entropy gradient at midpoint
    G_bar = zeros(N_PARTICLES, 2)     # discrete-gradient-corrected G
    dot_v = zeros(N_PARTICLES, 2)
    v_old = zeros(N_PARTICLES, 2)
    v_mid = zeros(N_PARTICLES, 2)
    v_prev = zeros(N_PARTICLES, 2)
    dv = zeros(N_PARTICLES, 2)
    f_coeffs_mid = zeros(n_dofs)
    f_coeffs_buf = zeros(n_dofs)

    snapshot_steps = [0, N_STEPS ÷ 4, N_STEPS ÷ 2, N_STEPS]
    coeff_snapshots = Dict{Int,Vector{Float64}}()
    coeff_snapshots[0] = copy(f_coeffs)

    w = w_particles[1]  # uniform weight = 1/N

    # Time-stepping loop
    for step in 1:N_STEPS
        v_old .= v_particles
        S_old = entropy_history[end]
        n_iter = MAX_PICARD

        for iter in 1:MAX_PICARD
            v_prev .= v_particles

            # Midpoint velocities (Eq 60)
            @. v_mid = (v_old + v_particles) / 2
            @. dv = v_particles - v_old

            # L² project at midpoint → entropy gradient (Eq 36-37)
            l2_project!(f_coeffs_mid, v_mid, w_particles)
            f_s_mid = build_field(f_coeffs_mid)
            compute_r!(r_vec, f_s_mid)
            L_vec .= M_lu \ r_vec
            compute_G!(G, v_mid, L_vec)

            # Gonzalez discrete gradient correction (Eq 58)
            #
            # The entropy gradient is ∂S_h/∂v_α = -w · G_α.
            # The discrete gradient replaces G with:
            #   Ḡ_α = G_α^mid - (c/w) · Δv_α
            # where c ensures S(v_{n+1}) - S(v_n) = Δv · p̄  exactly.
            dv_sq = sum(abs2, dv)
            G_bar .= G

            if dv_sq > 1e-30
                l2_project!(f_coeffs_buf, v_particles, w_particles)
                S_new = compute_entropy(build_field(f_coeffs_buf))

                dv_dot_G = 0.0
                @inbounds for α in axes(G, 1)
                    dv_dot_G += G[α,1]*dv[α,1] + G[α,2]*dv[α,2]
                end
                dv_dot_G *= w

                c = (S_new - S_old + dv_dot_G) / dv_sq
                inv_w = 1.0 / w
                @. G_bar = G - (c * inv_w) * dv
            end

            # Collision operator at midpoint with discrete gradient (Eq 59)
            compute_collision!(dot_v, v_mid, w_particles, G_bar)

            # Update: v^{n+1} = v^n + Δt · RHS
            @. v_particles = v_old + DT * dot_v

            # Convergence check
            rel_err = sum(abs2, v_particles .- v_prev) /
                      (sum(abs2, v_particles) + 1e-30)
            if rel_err < PICARD_TOL
                n_iter = iter
                break
            end
        end

        l2_project!(f_coeffs, v_particles, w_particles)
        f_s = build_field(f_coeffs)
        push!(entropy_history, compute_entropy(f_s))

        if step in snapshot_steps
            coeff_snapshots[step] = copy(f_coeffs)
        end

        if step % 25 == 0
            println("Step $step/$N_STEPS  S = $(round(entropy_history[end]; digits=6))  ($(n_iter) Picard iter)")
        end
    end

    begin # Visualization: f_s(v₁, v₂) heatmaps + H-function
        nv = 201
        v_grid = LinRange(V_MIN + 1e-10, V_MAX - 1e-10, nv)

        snap_keys = sort(collect(keys(coeff_snapshots)))
        fig = Figure(; size=(1200, 1100))

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

        n_rows = cld(length(snap_keys), 2)
        H_history = -entropy_history
        ax_H = Axis(fig[n_rows+1, 1:2];
            xlabel="time step", ylabel="H_h = -S_h",
            title="Boltzmann H-function (implicit, should decrease monotonically)")
        lines!(ax_H, 0:length(H_history)-1, H_history; color=:red, linewidth=2)
        save("landau_collision_implicit.png", fig)
        println("Saved landau_collision_implicit.png")
    end
end

main()
