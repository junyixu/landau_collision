#! /usr/bin/env -S julia --color=yes --startup-file=no
# Visualize only the initial L² projection of f_s(v₁, v₂).

include("MantisWrappers.jl")
using .MantisWrappers
using GLMakie
using Random

include("parameters.jl")


function main()

    begin # initialize particles and compute L² projection
        Random.seed!(42)
        v_particles = [σ₁ * randn(N_PARTICLES) σ₂ * randn(N_PARTICLES)]
        w_particles = fill(1.0 / N_PARTICLES, N_PARTICLES)
        f_coeffs = zeros(n_dofs)
        l2_project!(f_coeffs, v_particles, w_particles)
        field = build_field(f_coeffs) # f_s(v) = Σᵢ fᵢ φᵢ(v)
    end


    nv = 201
    v_grid = LinRange(V_MIN + 1e-10, V_MAX - 1e-10, nv)
    F = evaluate_on_grid(field, v_grid)

    println("f range: [$(minimum(F)), $(maximum(F))]")

    fig = Figure(; size=(600, 500))
    ax = Axis(fig[1, 1];
        title="Initial L² projection of f_s  (N=$(N_PARTICLES), $(N_ELEM)×$(N_ELEM) elems)",
        xlabel="v₁", ylabel="v₂", aspect=DataAspect())
    hm = heatmap!(ax, v_grid, v_grid, F'; colormap=:inferno)
    Colorbar(fig[1, 2], hm)
    xlims!(ax, V_MIN, V_MAX)
    ylims!(ax, V_MIN, V_MAX)

    save("initial_fs.png", fig)
    println("Saved initial_fs.png")
end

main()
