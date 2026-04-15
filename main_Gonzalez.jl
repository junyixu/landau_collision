#! /usr/bin/env -S julia --color=yes --startup-file=no
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Time integration via the Gonzalez discrete gradient method.
# Implements Equations (59)–(61) of Jeyakumar et al.:
#
#   (v^{n+1} - v^n)/Δt = G̃_γσ(v_mid) ∇̄_σ S_h(v^n, v^{n+1})          (60)
#
# where the midpoint approximation for the dissipation matrix is (Eq. 61):
#   G̃_γσ(v_n, v_{n+1}) = G_γσ((v_n + v_{n+1})/2)
#
# and the Gonzalez discrete gradient (Eq. 59) for S_h is:
#   ∇̄S_h(z_n, z_{n+1}) = ∇S_h(z_{mid})
#       + (z_{n+1} - z_n) * [S_h(z_{n+1}) - S_h(z_n) - (z_{n+1}-z_n)·∇S_h(z_mid)]
#                            / |z_{n+1} - z_n|²
#
# This guarantees exact discrete conservation of momentum and energy, and
# monotone dissipation of entropy (discrete H-theorem), Sections 4.1–4.2.
#
# The implicit equation is solved by fixed-point (Picard) iteration, using an
# explicit Euler step as the initial guess. Each iteration costs one O(N²)
# collision sweep at the current midpoint estimate.

include("MantisWrappers.jl")
using .MantisWrappers
using GLMakie
using Random
using LinearAlgebra

include("parameters.jl")
println("V ∈ [$V_MIN, $V_MAX],  p=$P_DEG, k=$K_REG,  $N_ELEM×$N_ELEM elements")
println("N_particles=$N_PARTICLES,  σ₁=$σ₁, σ₂=$σ₂,  Δt=$DT,  N_steps=$N_STEPS")


# Compute ∂S_h/∂v_α = -w_α G_α for every particle at positions v_parts.
# Uses Eq. (38): ∂S_h/∂v_α = -Σ_k w_α L_k ∇φ_k(v_α).
# Stores result in dS[α, :] (N×2).  Returns the projected field f_s.
function compute_entropy_gradient!(dS, v_parts, w_parts, f_coeffs_buf)
    l2_project!(f_coeffs_buf, v_parts, w_parts)
    f_s = build_field(f_coeffs_buf)
    r_vec = zeros(n_dofs)
    compute_r!(r_vec, f_s)
    L_vec = M_lu \ r_vec
    G = zeros(size(v_parts))
    compute_G!(G, v_parts, L_vec)
    for α in axes(v_parts, 1)
        dS[α, 1] = -w_parts[α] * G[α, 1]
        dS[α, 2] = -w_parts[α] * G[α, 2]
    end
    return f_s
end


# Fixed-point iteration for one Gonzalez discrete gradient time step.
#
# Arguments
#   v1        : (N×2) in/out — initialised to Euler guess, converges to v^{n+1}
#   v0        : (N×2) particle velocities at time n (not modified)
#   w_parts   : (N,)  particle weights (constant)
#   S0        : entropy S_h(v^n) (precomputed to avoid repeated L² projection)
#   dt        : time step
#   max_iter  : maximum Picard iterations (default 5; small Δt converges quickly)
function step_gonzalez!(v1, v0, w_parts, S0, dt;
                        max_iter=5)
    N = size(v0, 1)
    v_mid     = similar(v0)
    dv        = similar(v0)
    dS_mid    = zeros(N, 2)
    G_eff     = zeros(N, 2)
    dot_v     = zeros(N, 2)
    f_buf     = zeros(n_dofs)

    for _ in 1:max_iter
        v_mid .= 0.5 .* (v0 .+ v1)
        dv    .= v1 .- v0

        # 1. ∂S_h/∂v_α at the current midpoint estimate
        compute_entropy_gradient!(dS_mid, v_mid, w_parts, f_buf)

        # 2. S_h(v1) for the Gonzalez correction scalar
        l2_project!(f_buf, v1, w_parts)
        S1 = compute_entropy(build_field(f_buf))

        # 3. Gonzalez scalar correction (Eq. 59)
        #    c = [S_h(v1) - S_h(v0) - Δv·∇S_h(v_mid)] / |Δv|²
        dot_dv_dS = 0.0
        nrm2_dv   = 0.0
        for α in 1:N
            dot_dv_dS += dv[α, 1] * dS_mid[α, 1] + dv[α, 2] * dS_mid[α, 2]
            nrm2_dv   += dv[α, 1]^2               + dv[α, 2]^2
        end
        correction = nrm2_dv > 1e-30 ? (S1 - S0 - dot_dv_dS) / nrm2_dv : 0.0

        # 4. Discrete gradient ∇̄S_h,α = ∂S_h/∂v_α|_mid + c·Δv_α
        #    Map to the G-convention used by compute_collision!:
        #      G_eff[α] = -(∇̄S_h,α) / w_α
        #    so that compute_collision! yields Σ_α w_α U(G_eff,γ - G_eff,α)
        #      = Σ_α U(∇̄S_h,α - (w_α/w_γ)∇̄S_h,γ) = (G̃ · ∇̄S_h)_γ  ✓
        for α in 1:N
            inv_w = 1.0 / w_parts[α]
            G_eff[α, 1] = -(dS_mid[α, 1] + correction * dv[α, 1]) * inv_w
            G_eff[α, 2] = -(dS_mid[α, 2] + correction * dv[α, 2]) * inv_w
        end

        # 5. Apply the midpoint dissipation matrix G̃(v_mid) to ∇̄S_h
        #    using the existing compute_collision! at midpoint positions
        compute_collision!(dot_v, v_mid, w_parts, G_eff)

        # 6. Update estimate of v^{n+1}
        v1 .= v0 .+ dt .* dot_v
    end
end


function main()
    Random.seed!(42)
    v_particles = zeros(N_PARTICLES, 2)
    v_particles[:, 1] .= σ₁ * randn(N_PARTICLES)
    v_particles[:, 2] .= σ₂ * randn(N_PARTICLES)
    w_particles = fill(1.0 / N_PARTICLES, N_PARTICLES)
    f_coeffs    = zeros(n_dofs)

    l2_project!(f_coeffs, v_particles, w_particles)
    f_s = build_field(f_coeffs)

    entropy_history = Float64[]
    push!(entropy_history, compute_entropy(f_s))
    println("Initial entropy S_h = $(entropy_history[end])")

    # Preallocate work arrays for the Euler initial-guess step
    r_vec  = zeros(n_dofs)
    L_vec  = zeros(n_dofs)
    G      = zeros(N_PARTICLES, 2)
    dot_v  = zeros(N_PARTICLES, 2)
    v1     = copy(v_particles)

    snapshot_steps = Set([0, N_STEPS ÷ 4, N_STEPS ÷ 2, N_STEPS])
    snapshots = Dict{Int,Matrix{Float64}}()
    snapshots[0] = copy(v_particles)

    for step in 1:N_STEPS
        S0 = entropy_history[end]

        # ------------------------------------------------------------------
        # Initial guess: explicit Euler step at current positions (same as
        # main.jl), evaluated at v^n to seed the Picard iteration cheaply.
        # ------------------------------------------------------------------
        compute_r!(r_vec, f_s)
        L_vec .= M_lu \ r_vec
        compute_G!(G, v_particles, L_vec)
        compute_collision!(dot_v, v_particles, w_particles, G)
        v1 .= v_particles .+ DT .* dot_v

        # ------------------------------------------------------------------
        # Gonzalez discrete gradient fixed-point iteration (Eqs. 59–61)
        # ------------------------------------------------------------------
        step_gonzalez!(v1, v_particles, w_particles, S0, DT)
        v_particles .= v1

        l2_project!(f_coeffs, v_particles, w_particles)
        f_s = build_field(f_coeffs)
        push!(entropy_history, compute_entropy(f_s))

        step in snapshot_steps && (snapshots[step] = copy(v_particles))
        step % 25 == 0 &&
            println("Step $step/$N_STEPS  S = $(round(entropy_history[end]; digits=6))")
    end

    begin # Visualization
        fig = Figure(; size=(1200, 800))

        snap_keys = sort(collect(keys(snapshots)))
        for (idx, s) in enumerate(snap_keys)
            row, col = fldmod1(idx, 2)
            ax = Axis(fig[row, col];
                title="t = $(round(s * DT; digits=4))",
                xlabel="v₁", ylabel="v₂", aspect=DataAspect())
            pts = snapshots[s]
            scatter!(ax, pts[:, 1], pts[:, 2]; markersize=2, color=:blue, alpha=0.3)
            xlims!(ax, V_MIN, V_MAX)
            ylims!(ax, V_MIN, V_MAX)
        end

        n_rows = cld(length(snap_keys), 2)
        H_history = -entropy_history
        ax_H = Axis(fig[n_rows+1, 1:2];
            xlabel="time step", ylabel="H_h = -S_h",
            title="Boltzmann H-function — Gonzalez discrete gradient (monotone decrease expected)")
        lines!(ax_H, 0:length(H_history)-1, H_history; color=:red, linewidth=2)
        save("landau_collision_gonzalez_2d.png", fig)
    end
end

main()
