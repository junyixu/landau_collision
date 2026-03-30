include("MantisWrappers.jl")
using .MantisWrappers
using GLMakie
using Random

include("parameters.jl");
println("V ∈ [$V_MIN, $V_MAX],  p=$P_DEG, k=$K_REG,  $N_ELEM×$N_ELEM elements")
println("N_particles=$N_PARTICLES,  σ₁=$σ₁, σ₂=$σ₂,  Δt=$DT,  N_steps=$N_STEPS")


function main()

    # Initialize particles from a Maxwellian distribution
    Random.seed!(42)
    v_particles = zeros(N_PARTICLES, 2)
    v_particles[:, 1] .= σ₁ * randn(N_PARTICLES)
    v_particles[:, 2] .= σ₂ * randn(N_PARTICLES)
    w_particles = fill(1.0 / N_PARTICLES, N_PARTICLES)
    f_coeffs = zeros(n_dofs)

    l2_project!(f_coeffs, v_particles, w_particles)
    f_s = build_field(f_coeffs)        # f_s(v) = Σ_i f_i φ_i(v)

    entropy_history = Float64[]
    push!(entropy_history, compute_entropy(f_s))
    println("Initial entropy S_h = $(entropy_history[end])")
    r_vec = zeros(n_dofs)
    L_vec = zeros(n_dofs)
    G = zeros(N_PARTICLES, 2)
    dot_v = zeros(N_PARTICLES, 2)

    snapshot_steps = [0, N_STEPS ÷ 4, N_STEPS ÷ 2, N_STEPS]
    snapshots = Dict{Int,Matrix{Float64}}()
    snapshots[0] = copy(v_particles)



    # Time-stepping loop
    for step in 1:N_STEPS
        compute_r!(r_vec, f_s)
        L_vec .= M_lu \ r_vec
        compute_G!(G, v_particles, L_vec)
        compute_collision!(dot_v, v_particles, w_particles, G)
        v_particles .+= DT .* dot_v

        l2_project!(f_coeffs, v_particles, w_particles)
        f_s = build_field(f_coeffs)        # f_s(v) = Σ_i f_i φ_i(v)

        push!(entropy_history, compute_entropy(f_s))

        if step in snapshot_steps
            snapshots[step] = copy(v_particles)
        end

        step % 25 == 0 && println("Step $step/$N_STEPS  S = $(round(entropy_history[end]; digits=6))")
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

        # H-function evolution: H_h = -S_h (Boltzmann H-theorem: dH/dt ≤ 0)
        n_rows = cld(length(snap_keys), 2)
        H_history = -entropy_history
        ax_H = Axis(fig[n_rows+1, 1:2];
            xlabel="time step", ylabel="H_h = -S_h",
            title="Boltzmann H-function (should decrease monotonically)")
        lines!(ax_H, 0:length(H_history)-1, H_history; color=:red, linewidth=2)
        save("landau_collision_2d.png", fig)
    end
end

main()
