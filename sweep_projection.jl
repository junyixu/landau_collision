#! /usr/bin/env -S julia --color=yes --startup-file=no
# Sweep over N_ELEM × N_PARTICLES and visualize the initial L² projection of f_s.
# Rows = N_ELEM values, Columns = N_PARTICLES values.

using Mantis
using GLMakie
using LinearAlgebra
using Random

const V_MIN, V_MAX = -5.0, 5.0
const P_DEG = 2
const K_REG = 1
const σ₁, σ₂ = 1.0, 0.3

const N_ELEM_VALUES      = [20, 30, 40]
const N_PARTICLES_VALUES = [10_000, 30_000, 50_000, 70_000, 100_000]

# Build all FEM structures for a given n_elem.
# Returns a NamedTuple with everything needed for l2_project! and evaluation.
function build_context(n_elem)
    bp         = LinRange(V_MIN, V_MAX, n_elem + 1)
    geo_1d     = Geometry.CartesianGeometry((bp,))
    B_1d       = FunctionSpaces.BSplineSpace(geo_1d, P_DEG, K_REG)
    TP         = FunctionSpaces.TensorProductSpace((B_1d, B_1d), Geometry.CartesianGeometry)
    X⁰         = Forms.FormSpace(0, TP, "f")
    n_dofs     = Forms.get_num_basis(X⁰)
    geo_2d     = Forms.get_geometry(X⁰)
    n_elements = Geometry.get_num_elements(geo_2d)
    lin_idx    = LinearIndices((n_elem, n_elem))

    qrule  = Quadrature.tensor_product_rule((P_DEG + 1, P_DEG + 1), Quadrature.gauss_legendre)
    dΩ     = Quadrature.StandardQuadrature(qrule, n_elements)
    f_zero = Forms.AnalyticalFormField(0, x -> [zeros(size(x, 1))], geo_2d, "0")
    wfi    = Assemblers.WeakFormInputs(X⁰, f_zero)
    v⁰     = Assemblers.get_test_form(wfi)
    u⁰     = Assemblers.get_trial_form(wfi)
    M, _   = Assemblers.assemble(Assemblers.WeakForm(((∫(v⁰ ∧ ★(u⁰), dΩ),),), ((0,),), wfi))
    M_lu   = lu(M)

    return (; bp, X⁰, n_dofs, M_lu, lin_idx)
end

function locate(ctx, v1, v2)
    (v1 <= V_MIN || v1 >= V_MAX || v2 <= V_MIN || v2 >= V_MAX) && return nothing
    i  = searchsortedlast(ctx.bp, v1)
    j  = searchsortedlast(ctx.bp, v2)
    h1 = ctx.bp[i+1] - ctx.bp[i]
    h2 = ctx.bp[j+1] - ctx.bp[j]
    ξ1 = (v1 - ctx.bp[i]) / h1
    ξ2 = (v2 - ctx.bp[j]) / h2
    elem_id = ctx.lin_idx[i, j]
    xi = Points.CartesianPoints(([ξ1], [ξ2]))
    return (; elem_id, xi)
end

function l2_project(ctx, v_parts, w_parts)
    rhs = zeros(ctx.n_dofs)
    for α in axes(v_parts, 1)
        loc = locate(ctx, v_parts[α, 1], v_parts[α, 2])
        isnothing(loc) && continue
        evals, indices = Forms.evaluate(ctx.X⁰, loc.elem_id, loc.xi)
        for (j, gidx) in enumerate(indices[1])
            rhs[gidx] += w_parts[α] * evals[1][1, j]
        end
    end
    return ctx.M_lu \ rhs
end

function evaluate_on_grid(ctx, coeffs, v_grid)
    field = Forms.build_form_field(ctx.X⁰, coeffs)
    nv = length(v_grid)
    F  = zeros(nv, nv)
    for (j, v2) in enumerate(v_grid)
        for (i, v1) in enumerate(v_grid)
            loc = locate(ctx, v1, v2)
            isnothing(loc) && continue
            fv, _ = Forms.evaluate(field, loc.elem_id, loc.xi)
            F[i, j] = fv[1][1]
        end
    end
    return F
end

function main()
    Random.seed!(42)
    nv     = 101
    v_grid = LinRange(V_MIN + 1e-10, V_MAX - 1e-10, nv)

    n_rows = length(N_ELEM_VALUES)
    n_cols = length(N_PARTICLES_VALUES)

    # First pass: compute all F matrices and find global color range
    grids = Matrix{Matrix{Float64}}(undef, n_rows, n_cols)
    for (row, n_elem) in enumerate(N_ELEM_VALUES)
        print("Building context for N_ELEM=$n_elem ... ")
        ctx = build_context(n_elem)
        println("done  ($(ctx.n_dofs) dofs)")
        for (col, n_particles) in enumerate(N_PARTICLES_VALUES)
            v_parts = hcat(σ₁ * randn(n_particles), σ₂ * randn(n_particles))
            w_parts = fill(1.0 / n_particles, n_particles)
            coeffs  = l2_project(ctx, v_parts, w_parts)
            F       = evaluate_on_grid(ctx, coeffs, v_grid)
            println("  N_PARTICLES=$n_particles  f ∈ [$(round(minimum(F); digits=5)), $(round(maximum(F); digits=5))]")
            grids[row, col] = F
        end
    end

    global_max = maximum(maximum.(grids))
    crange = (-global_max, global_max)

    # Second pass: plot with unified colorrange
    fig = Figure(; size=(380 * n_cols + 80, 380 * n_rows))
    for (row, n_elem) in enumerate(N_ELEM_VALUES)
        for (col, n_particles) in enumerate(N_PARTICLES_VALUES)
            ax = Axis(fig[row, col];
                title="$(n_elem)×$(n_elem) elems, N=$(n_particles)",
                xlabel="v₁", ylabel="v₂", aspect=DataAspect(),
                titlesize=13)
            hm = heatmap!(ax, collect(v_grid), collect(v_grid), grids[row, col]';
                colormap=Reverse(:RdBu), colorrange=crange)
            xlims!(ax, V_MIN, V_MAX)
            ylims!(ax, V_MIN, V_MAX)
        end
    end
    Colorbar(fig[1:n_rows, n_cols+1]; colormap=Reverse(:RdBu), limits=crange, width=20)

    save("sweep_projection.png", fig)
    println("\nSaved sweep_projection.png")
end

main()
