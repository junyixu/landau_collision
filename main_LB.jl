#! /usr/bin/env -S julia --color=yes --startup-file=no
# -*- coding: utf-8 -*-
#
# Implicit-midpoint particle discretisation of the conservative 2D
# Lenard-Bernstein operator (Jeyakumar et al. 2024, generalised to 2D velocity
# space). Same FEM scaffolding as the 2D Landau (Gonzalez) code so an LB run on
# an identical mesh isolates whether the honeycomb artifact is mesh/projection-
# driven or Landau-kernel-specific.
#
#   julia --project=. main_LB.jl parameters_LB2D_v3.jl
#   julia --project=. main_LB.jl parameters_LB2D_v3.jl --N_STEPS=200 --suffix=foo
#
# ARGS[1]   : preset file defining `PARAMS::SimParameters`
# ARGS[2:]  : --key=value scalar overrides
#
# Diagnostics streamed to CSV every step:
#   iter        : Picard / Anderson iterations
#   residual    : ‖G(v) − v‖₂
#   fp_minus_fs : ‖f_s − f_p‖₂  (histogram-based projection error)
#   neg_part    : ∫ max(−f_s, 0) dv  (Gibbs negative-part L¹)

include("MantisWrappers.jl")
using .MantisWrappers

using GLMakie
using Random
using Serialization
using LinearAlgebra
using LinearAlgebra: ldiv!, mul!

# ---- Checkpoint / resume ----------------------------------------------------
function checkpoint_path(suffix::String, step::Int)
    return "checkpoint_$(suffix)_step$(lpad(step, 5, '0')).jls"
end

function save_checkpoint(
    suffix::String,
    step::Int,
    v_particles,
    w_particles,
    f_coeffs,
    entropy_history,
    energy_history,
    momentum_history,
    iter_history,
    res_history,
    fp_l2_history,
    neg_history,
    rng_state,
)
    fname = checkpoint_path(suffix, step)
    open(fname, "w") do io
        serialize(
            io,
            (;
                step,
                v_particles,
                w_particles,
                f_coeffs,
                entropy_history,
                energy_history,
                momentum_history,
                iter_history,
                res_history,
                fp_l2_history,
                neg_history,
                rng_state,
            ),
        )
    end
    println("Saved $fname")
    return fname
end

function load_checkpoint(suffix::String, step)
    fname = if step === :auto || (step isa Integer && step <= 0)
        files = filter(readdir()) do f
            startswith(f, "checkpoint_$(suffix)_step") && endswith(f, ".jls")
        end
        isempty(files) && error("No checkpoint files for suffix=$suffix")
        sort(files)[end]
    else
        checkpoint_path(suffix, Int(step))
    end
    isfile(fname) || error("Checkpoint not found: $fname")
    println("Loading checkpoint: $fname")
    return open(deserialize, fname)
end

# One Picard map for the implicit midpoint scheme (2D LB).
#   v_mid   = 0.5 * (v0 + v_in)
#   f_s_mid = L² projection of {(v_mid_α, w_α)}
#   A, B    = 3×3 multiplier solve (momentum + energy conservation)
#   v̇(v_α)  = -ν (∇f_s/f_s + A + B v_α)
#   v_out   = v0 + dt · v̇(v_mid_α)
function picard_map!(ws::Workspace, v_out, v_in, v0, w_parts, dt, v_mid, f_coeffs, g, dot_v)
    @. v_mid = 0.5 * (v0 + v_in)
    l2_project!(ws, f_coeffs, v_mid, w_parts)
    eval_loggrad_at_particles!(ws, g, v_mid, f_coeffs)
    n, U1, U2, Q = compute_moments(v_mid, w_parts)
    A1, A2, B = compute_drift_multipliers(v_mid, w_parts, g, n, U1, U2, Q)
    compute_LB_velocity!(dot_v, v_mid, g, A1, A2, B, ws.p.nu)
    @. v_out = v0 + dt * dot_v
    return nothing
end

# Anderson-accelerated fixed-point iteration. State is the N×2 velocity matrix;
# all linear-algebra ops use its flat `vec()` view (length 2N), so the solver
# is dimension-agnostic. `use_anderson=false` ⇒ damped Picard.
function step_anderson!(
    ws::Workspace,
    v1,
    v0,
    w_parts,
    dt,
    v_mid,
    f_coeffs,
    g,
    dot_v,
    Gv,
    r_curr,
    r_prev,
    Gv_prev,
    v_old,
    ΔF,
    ΔG;
    m=5,
    max_iter=1000,
    tol=1e-12,
    abs_floor=1e-10,
    stag_window=50,
    stag_rel_tol=0.01,
    damp_decay_start=200,
    damp_decay_factor=0.5,
    restart_factor=Inf,
    damping=0.5,
    reg_factor=1e-10,
    verbose=false,
    use_anderson::Bool=true,
)
    v1_v = vec(v1)
    Gv_v = vec(Gv)
    r_v = vec(r_curr)
    rp_v = vec(r_prev)
    Gp_v = vec(Gv_prev)
    vold_v = vec(v_old)

    history = 0
    nrm_r = 0.0
    nrm_best = Inf
    n_restart = 0

    Gv_best = copy(Gv)
    nrm_best_window = Inf

    for k in 1:max_iter
        vold_v .= v1_v
        picard_map!(ws, Gv, v1, v0, w_parts, dt, v_mid, f_coeffs, g, dot_v)
        @. r_v = Gv_v - v1_v
        nrm_r = norm(r_v)

        if nrm_r < nrm_best
            nrm_best = nrm_r
            Gv_best .= Gv
        end

        eff_tol = max(tol * (norm(v1_v) + 1e-30), abs_floor)
        if nrm_r < eff_tol
            v1 .= Gv
            verbose && println("    k=$k  ‖r‖=$nrm_r  history=$history  [converged]")
            return k, nrm_r, n_restart
        end

        if k > stag_window && k % stag_window == 0
            rel_improve = (nrm_best_window - nrm_best) / (nrm_best_window + 1e-30)
            if rel_improve < stag_rel_tol
                v1 .= Gv_best
                verbose && println("    k=$k  stagnated  nrm_best=$nrm_best  Δ_rel=$rel_improve")
                return k, nrm_best, n_restart
            end
            nrm_best_window = nrm_best
        end

        just_restarted = false
        if k > 1 && nrm_r > restart_factor * nrm_best
            history = 0
            n_restart += 1
            just_restarted = true
        end

        verbose && println("    k=$k  ‖r‖=$nrm_r  history=$history" * (just_restarted ? "  [restart]" : ""))

        damping_eff = k > damp_decay_start ? damping * damp_decay_factor : damping

        if k == 1 || just_restarted || !use_anderson
            @. v1_v = damping_eff * Gv_v + (1 - damping_eff) * vold_v
        else
            if history < m
                history += 1
                new_col = history
            else
                @views ΔF[:, 1:(m - 1)] .= ΔF[:, 2:m]
                @views ΔG[:, 1:(m - 1)] .= ΔG[:, 2:m]
                new_col = m
            end
            @views ΔF[:, new_col] .= r_v .- rp_v
            @views ΔG[:, new_col] .= Gv_v .- Gp_v

            ΔFv = @view ΔF[:, 1:history]
            ΔGv = @view ΔG[:, 1:history]
            ATA = ΔFv' * ΔFv
            ATr = ΔFv' * r_v
            diag_mean = 0.0
            @inbounds for j in 1:history
                diag_mean += ATA[j, j]
            end
            diag_mean /= history
            λ2 = reg_factor * diag_mean + 1e-30
            @inbounds for j in 1:history
                ATA[j, j] += λ2
            end
            γ = ATA \ ATr
            v1 .= Gv
            mul!(v1_v, ΔGv, γ, -1.0, 1.0)
            @. v1_v = damping_eff * v1_v + (1 - damping_eff) * vold_v
        end

        rp_v .= r_v
        Gp_v .= Gv_v
    end

    v1 .= Gv_best
    @warn "Solver did not converge" max_iter tol abs_floor nrm_r nrm_best n_restart
    return max_iter, nrm_best, n_restart
end

function compute_momentum(v_parts, w_parts)
    p1 = sum(w_parts[α] * v_parts[α, 1] for α in axes(v_parts, 1))
    p2 = sum(w_parts[α] * v_parts[α, 2] for α in axes(v_parts, 1))
    return (p1, p2)
end

function compute_energy(v_parts, w_parts)
    return 0.5 * sum(w_parts[α] * (v_parts[α, 1]^2 + v_parts[α, 2]^2) for α in axes(v_parts, 1))
end

function save_fs_snapshot(ws::Workspace, suffix::String, step::Int, f_coeffs::AbstractVector)
    fname = "fs_snapshot_$(suffix)_step$(lpad(step, 5, '0')).csv"
    open(fname, "w") do io
        println(io, "# bp1=", join(ws.bp1, ","))
        println(io, "# bp2=", join(ws.bp2, ","))
        println(io, "# n_dofs=", ws.n_dofs)
        println(io, "coeff")
        for c in f_coeffs
            println(io, c)
        end
    end
    return fname
end

# Per-snapshot dashboard: f_s(v₁_fixed, v₂) slices + log10|f_s| heatmap with a
# negative-region mask overlay (the honeycomb probe).
function plot_fs_diagnostics(
    ws::Workspace,
    f_coeffs::AbstractVector,
    suffix::String,
    step::Int;
    v1_slices::Vector{Float64}=[0.0, 0.5, 1.0],
    n_v2::Int=400,
    n_grid::Int=200,
)
    p = ws.p
    f_s = build_field(ws, f_coeffs)

    v2_grid = collect(range(p.bp2[1], p.bp2[end]; length=n_v2))
    fig = Figure(; size=(1300, 900))
    ax_slice = Axis(
        fig[1, 1:2]; xlabel="v₂", ylabel="f_s", title="f_s slices along v₂  (suffix=$suffix, step=$step)"
    )
    palette = [:blue, :red, :green, :purple]
    for (i, v1f) in enumerate(v1_slices)
        vals = [
            begin
                loc = locate_particle(ws, v1f, v2)
                isnothing(loc) ? 0.0 : (evaluate(ws, f_s, loc)[1][1][1])
            end for v2 in v2_grid
        ]
        lines!(
            ax_slice, v2_grid, vals; color=palette[mod1(i, length(palette))], linewidth=2, label="v₁ = $v1f"
        )
    end
    hlines!(ax_slice, [0.0]; color=:black, linestyle=:dash, linewidth=1)
    axislegend(ax_slice; position=:rt)

    v1_grid = collect(range(p.bp1[1], p.bp1[end]; length=n_grid))
    v2_grid_h = collect(range(p.bp2[1], p.bp2[end]; length=n_grid))
    F = evaluate_on_grid(ws, f_s, v1_grid, v2_grid_h)

    F_log = similar(F)
    @inbounds for I in eachindex(F)
        a = abs(F[I])
        F_log[I] = a > 1e-30 ? log10(a) : -30.0
    end
    ax_h = Axis(fig[2, 1]; xlabel="v₁", ylabel="v₂", title="log10|f_s|", aspect=DataAspect())
    hm = heatmap!(ax_h, v1_grid, v2_grid_h, F_log; colormap=:viridis)
    Colorbar(fig[2, 1, Right()], hm)

    neg_mask = map(x -> x < 0.0 ? 1.0 : NaN, F)
    ax_n = Axis(
        fig[2, 2]; xlabel="v₁", ylabel="v₂", title="negative-region mask (red = f_s < 0)", aspect=DataAspect()
    )
    hm2 = heatmap!(ax_n, v1_grid, v2_grid_h, F_log; colormap=:viridis)
    Colorbar(fig[2, 2, Right()], hm2)
    heatmap!(ax_n, v1_grid, v2_grid_h, neg_mask; colormap=[:transparent, :red], colorrange=(0.0, 1.0))

    png_name = "fs_diag_$(suffix)_step$(lpad(step, 5, '0')).png"
    save(png_name, fig)
    println("Saved $png_name")
    return png_name
end

function plot_run_dashboard(
    ws::Workspace,
    entropy_history,
    energy_history,
    momentum_history,
    iter_history,
    res_history,
    fp_l2_history,
    neg_history,
    suffix::String,
)
    p = ws.p
    steps = 0:(p.N_STEPS)

    E0 = energy_history[1]
    P0 = momentum_history[1]
    E_err = [abs(energy_history[n + 1] - E0) / max(abs(E0), 1e-30) for n in steps]
    P_err = [
        hypot(momentum_history[n + 1][1] - P0[1], momentum_history[n + 1][2] - P0[2]) /
        max(hypot(P0[1], P0[2]), 1e-30) for n in steps
    ]

    fig = Figure(; size=(1200, 1500))

    ax_S = Axis(
        fig[1, 1];
        xlabel="step",
        ylabel="H_h",
        title="Entropy H_h = -∫f log f dv  (monotone increase expected)",
    )
    lines!(ax_S, collect(steps), entropy_history; color=:red, linewidth=2)

    ax_E = Axis(
        fig[2, 1]; xlabel="step", ylabel="rel. error", title="Energy conservation error", yscale=log10
    )
    lines!(ax_E, collect(steps), max.(E_err, 1e-18); color=:blue, linewidth=2)

    ax_P = Axis(
        fig[3, 1]; xlabel="step", ylabel="rel. error", title="Momentum conservation error", yscale=log10
    )
    lines!(ax_P, collect(steps), max.(P_err, 1e-18); color=:green, linewidth=2)

    ax_I = Axis(fig[4, 1]; xlabel="step", ylabel="iter", title="Inner-iteration count")
    lines!(ax_I, 1:(p.N_STEPS), iter_history; color=:black, linewidth=2)

    ax_R = Axis(
        fig[5, 1]; xlabel="step", ylabel="‖r‖", title="Picard fixed-point residual ‖G(v) − v‖₂", yscale=log10
    )
    lines!(ax_R, 1:(p.N_STEPS), max.(res_history, 1e-30); color=:purple, linewidth=2)

    ax_F = Axis(
        fig[6, 1];
        xlabel="step",
        ylabel="‖f_s − f_p‖₂",
        title="Histogram-based projection error  ‖f_s − f_p‖₂",
    )
    lines!(ax_F, 1:(p.N_STEPS), fp_l2_history; color=:orange, linewidth=2)

    ax_N = Axis(fig[7, 1]; xlabel="step", ylabel="∫max(−f_s,0)", title="Negative-part L¹ of f_s")
    lines!(ax_N, 1:(p.N_STEPS), neg_history; color=:darkred, linewidth=2)

    png_name = "dashboard_$(suffix).png"
    save(png_name, fig)
    println("Saved $png_name")
    return png_name
end

function run_simulation(p::SimParameters; resume=nothing)
    print_summary(p)
    Random.seed!(p.seed)

    ws = build_workspace(p)
    println("Workspace: n_dofs=$(ws.n_dofs)  n_elements=$(ws.n_elements)")

    start_step = 0
    local v_particles, w_particles, f_coeffs
    local entropy_history, energy_history, momentum_history
    local iter_history, res_history, fp_l2_history, neg_history

    if resume !== nothing
        ckpt = load_checkpoint(p.suffix, resume)
        start_step = ckpt.step
        v_particles = ckpt.v_particles
        w_particles = ckpt.w_particles
        f_coeffs = ckpt.f_coeffs
        entropy_history = ckpt.entropy_history
        energy_history = ckpt.energy_history
        momentum_history = ckpt.momentum_history
        iter_history = ckpt.iter_history
        res_history = ckpt.res_history
        fp_l2_history = ckpt.fp_l2_history
        neg_history = ckpt.neg_history
        copy!(Random.default_rng(), ckpt.rng_state)

        size(v_particles, 1) == p.N_PARTICLES ||
            error("Checkpoint N_PARTICLES=$(size(v_particles,1)) ≠ preset N_PARTICLES=$(p.N_PARTICLES)")
        length(f_coeffs) == ws.n_dofs ||
            error("Checkpoint n_dofs=$(length(f_coeffs)) ≠ workspace n_dofs=$(ws.n_dofs); mesh changed?")
        start_step < p.N_STEPS || error("Checkpoint step=$start_step ≥ N_STEPS=$(p.N_STEPS); nothing to do")

        println("Resuming from step $start_step (running through $(p.N_STEPS))")
    else
        v_particles = sample_initial_velocities(p, Random.default_rng())
        w_particles = fill(1.0 / p.N_PARTICLES, p.N_PARTICLES)
        f_coeffs = zeros(ws.n_dofs)
        l2_project!(ws, f_coeffs, v_particles, w_particles)

        entropy_history = Float64[]
        energy_history = Float64[]
        momentum_history = NTuple{2,Float64}[]
        iter_history = Int[]
        res_history = Float64[]
        fp_l2_history = Float64[]
        neg_history = Float64[]

        f_s0 = build_field(ws, f_coeffs)
        push!(entropy_history, compute_entropy(ws, f_s0))
        push!(energy_history, compute_energy(v_particles, w_particles))
        push!(momentum_history, compute_momentum(v_particles, w_particles))
        println("Initial  H_h = $(entropy_history[end])")
        println("Initial  E   = $(energy_history[end])")
        println("Initial  P   = $(momentum_history[end])")
        fs_minus_fp_l2 = compute_fs_minus_fp_l2(ws, f_s0, v_particles, w_particles)
        println("Initial  ‖f_s − f_p‖₂ = " * string(fs_minus_fp_l2))
        neg_part = compute_negative_part_l1(ws, f_s0)
        println("Initial  ∫max(−f_s,0) = " * string(neg_part))
    end

    v1 = copy(v_particles)
    v_mid = similar(v_particles)
    g = zeros(p.N_PARTICLES, 2)
    dot_v = zeros(p.N_PARTICLES, 2)
    f_buf = zeros(ws.n_dofs)

    Gv = zeros(p.N_PARTICLES, 2)
    r_curr = zeros(p.N_PARTICLES, 2)
    r_prev = zeros(p.N_PARTICLES, 2)
    Gv_prev = zeros(p.N_PARTICLES, 2)
    v_old_buf = zeros(p.N_PARTICLES, 2)
    ΔF = zeros(2 * p.N_PARTICLES, p.m_anderson)
    ΔG = zeros(2 * p.N_PARTICLES, p.m_anderson)

    snapshot_steps = Set(0:(p.snap_every):(p.N_STEPS))
    push!(snapshot_steps, p.N_STEPS)

    cons_csv = "conservation_history_$(p.suffix).csv"
    snap_csv = "particle_snapshots_$(p.suffix).csv"

    if start_step == 0
        save_fs_snapshot(ws, p.suffix, 0, f_coeffs)
        plot_fs_diagnostics(ws, f_coeffs, p.suffix, 0)

        cons_io = open(cons_csv, "w")
        println(
            cons_io, "step,time,entropy,energy,momentum_1,momentum_2," * "iter,residual,fp_minus_fs,neg_part"
        )
        println(
            cons_io,
            "0,0.0,$(entropy_history[1]),$(energy_history[1])," *
            "$(momentum_history[1][1]),$(momentum_history[1][2])," *
            "0,0.0,$fs_minus_fp_l2,$neg_part",
        )
        flush(cons_io)

        snap_io = open(snap_csv, "w")
        println(snap_io, "step,time,particle_idx,v1,v2")
        for i in axes(v_particles, 1)
            println(snap_io, "0,0.0,$i,$(v_particles[i, 1]),$(v_particles[i, 2])")
        end
        flush(snap_io)

        save_checkpoint(
            p.suffix,
            0,
            v_particles,
            w_particles,
            f_coeffs,
            entropy_history,
            energy_history,
            momentum_history,
            iter_history,
            res_history,
            fp_l2_history,
            neg_history,
            copy(Random.default_rng()),
        )
    else
        cons_io = open(cons_csv, "a")
        snap_io = open(snap_csv, "a")
    end

    for step in (start_step + 1):(p.N_STEPS)
        # Initial guess: explicit-Euler step using current f_s
        l2_project!(ws, f_buf, v_particles, w_particles)
        eval_loggrad_at_particles!(ws, g, v_particles, f_buf)
        n_h, U1, U2, Q = compute_moments(v_particles, w_particles)
        A1, A2, B = compute_drift_multipliers(v_particles, w_particles, g, n_h, U1, U2, Q)
        compute_LB_velocity!(dot_v, v_particles, g, A1, A2, B, p.nu)
        @. v1 = v_particles + p.DT * dot_v

        iter, res_final, n_rs = step_anderson!(
            ws,
            v1,
            v_particles,
            w_particles,
            p.DT,
            v_mid,
            f_buf,
            g,
            dot_v,
            Gv,
            r_curr,
            r_prev,
            Gv_prev,
            v_old_buf,
            ΔF,
            ΔG;
            m=p.m_anderson,
            max_iter=p.max_iter,
            tol=p.tol,
            abs_floor=p.abs_floor,
            stag_window=p.stag_window,
            stag_rel_tol=p.stag_rel_tol,
            damp_decay_start=p.damp_decay_start,
            damp_decay_factor=p.damp_decay_factor,
            damping=p.damping,
            use_anderson=p.use_anderson,
            verbose=(step <= 3),
        )
        v_particles .= v1

        l2_project!(ws, f_coeffs, v_particles, w_particles)
        f_s = build_field(ws, f_coeffs)

        push!(entropy_history, compute_entropy(ws, f_s))
        push!(energy_history, compute_energy(v_particles, w_particles))
        push!(momentum_history, compute_momentum(v_particles, w_particles))
        push!(iter_history, iter)
        push!(res_history, res_final)
        push!(fp_l2_history, compute_fs_minus_fp_l2(ws, f_s, v_particles, w_particles))
        push!(neg_history, compute_negative_part_l1(ws, f_s))

        let t = step * p.DT, P = momentum_history[end]
            println(
                cons_io,
                "$step,$t,$(entropy_history[end]),$(energy_history[end])," *
                "$(P[1]),$(P[2]),$iter,$res_final," *
                "$(fp_l2_history[end]),$(neg_history[end])",
            )
            flush(cons_io)
        end

        if step in snapshot_steps
            save_fs_snapshot(ws, p.suffix, step, f_coeffs)
            plot_fs_diagnostics(ws, f_coeffs, p.suffix, step)
            let t = step * p.DT
                for i in axes(v_particles, 1)
                    println(snap_io, "$step,$t,$i,$(v_particles[i, 1]),$(v_particles[i, 2])")
                end
                flush(snap_io)
            end
            save_checkpoint(
                p.suffix,
                step,
                v_particles,
                w_particles,
                f_coeffs,
                entropy_history,
                energy_history,
                momentum_history,
                iter_history,
                res_history,
                fp_l2_history,
                neg_history,
                copy(Random.default_rng()),
            )
        end

        step % p.snap_every == 0 && println(
            "Step $step/$(p.N_STEPS)  iter=$iter  rs=$n_rs" *
            "  ‖r‖=$(round(res_final; sigdigits=3))" *
            "  ‖f_s−f_p‖=$(round(fp_l2_history[end]; sigdigits=4))" *
            "  neg=$(round(neg_history[end]; sigdigits=4))" *
            "  H=$(round(entropy_history[end]; digits=6))" *
            "  E=$(round(energy_history[end]; sigdigits=10))",
        )
    end

    close(cons_io)
    close(snap_io)
    println("Saved $cons_csv")
    println("Saved $snap_csv")

    plot_run_dashboard(
        ws,
        entropy_history,
        energy_history,
        momentum_history,
        iter_history,
        res_history,
        fp_l2_history,
        neg_history,
        p.suffix,
    )

    return (;
        entropy_history,
        energy_history,
        momentum_history,
        iter_history,
        res_history,
        fp_l2_history,
        neg_history,
        label=(p.use_anderson ? "Anderson(m=$(p.m_anderson))" : "Picard"),
    )
end

function main(args=ARGS)
    if isempty(args)
        preset = "parameters_LB2D_v3.jl"
        overrides = String[]
    else
        preset = args[1]
        overrides = collect(String, args[2:end])
    end
    isfile(preset) || error("Preset file not found: $preset")

    resume = nothing
    overrides = filter(overrides) do tok
        if startswith(tok, "--resume=")
            val = tok[(length("--resume=") + 1):end]
            resume = (val == "auto") ? :auto : parse(Int, val)
            return false
        end
        return true
    end

    println("Loading preset: $preset")
    params_loaded = include(joinpath(@__DIR__, preset))
    p = parse_overrides(params_loaded::SimParameters, overrides)

    res = run_simulation(p; resume=resume)
    if isempty(res.iter_history)
        println("\n--- No new steps run (already at N_STEPS) ---")
    else
        a = sum(res.iter_history) / length(res.iter_history)
        println("\n--- Inner-iter summary ---")
        println(
            "$(res.label):  avg=$(round(a; digits=2))  max=$(maximum(res.iter_history))" *
            "  steps=$(length(res.iter_history))",
        )
    end
end

main()
