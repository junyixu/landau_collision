# # MantisWrappers (1D)
#
# Wraps Mantis FEM for 1D velocity-space B-spline projection used by the
# conservative Lenard-Bernstein collision operator (Jeyakumar et al. 2024).
#
# All FEM state held in a single `Workspace` struct constructed from a
# `SimParameters` instance. No module-level globals depend on parameters.

module MantisWrappers

using Mantis
using LinearAlgebra
using LinearAlgebra: mul!, lu

include("Parameters.jl")

# ## Bézier-extraction scratch buffer (1D)
struct ParticleBuf
    B::Vector{Float64}
    dB::Vector{Float64}
    phi::Vector{Float64}
    dphi::Vector{Float64}
end
ParticleBuf(p_deg::Int) = ParticleBuf(zeros(p_deg+1), zeros(p_deg+1),
                                       zeros(p_deg+1), zeros(p_deg+1))

# ## Workspace
struct Workspace{G, X, L, Q, E}
    p::SimParameters

    bp::Vector{Float64}

    geo_1d::G
    X⁰::X
    n_dofs::Int
    n_elements::Int

    # Mass-matrix LU factorization
    M_lu::L

    # Quadrature rule (1D, used for entropy / negative-part integration)
    qrule_integrate::Q

    # Bézier extraction cache
    ext1d::Vector{E}
    basis_start_1d::Vector{Int}

    # Per-particle scratch
    pbuf::ParticleBuf
    lp_vals::Vector{Float64}
    lp_gids::Vector{Int}
    G_vals::Vector{Float64}
    G_dxi::Vector{Float64}
    G_gids::Vector{Int}
end

function build_workspace(p::SimParameters)
    bp = p.bp

    geo_1d = Geometry.CartesianGeometry((bp,))
    B_1d   = FunctionSpaces.BSplineSpace(geo_1d, p.P_DEG, p.K_REG)
    X⁰     = Forms.FormSpace(0, B_1d, "f")

    n_dofs     = Forms.get_num_basis(X⁰)
    n_elements = Geometry.get_num_elements(geo_1d)

    M_lu = let
        qrule = Quadrature.tensor_product_rule((p.P_DEG + 1,), Quadrature.gauss_legendre)
        dΩ = Quadrature.StandardQuadrature(qrule, n_elements)
        f_zero = Forms.AnalyticalFormField(0, x -> [zeros(size(x, 1))], geo_1d, "0")
        wfi = Assemblers.WeakFormInputs(X⁰, f_zero)
        v⁰ = Assemblers.get_test_form(wfi)
        u⁰ = Assemblers.get_trial_form(wfi)
        M_expr = ∫(v⁰ ∧ ★(u⁰), dΩ)
        M_wf = Assemblers.WeakForm(((M_expr,),), ((0,),), wfi)
        M, _ = Assemblers.assemble(M_wf)
        lu(M)
    end

    qrule_integrate = Quadrature.tensor_product_rule((p.N_QUAD,), Quadrature.gauss_legendre)

    n_elem = length(bp) - 1
    ext1d = [FunctionSpaces.get_extraction(B_1d, e, 1)[1] for e in 1:n_elem]
    basis_start_1d = [first(FunctionSpaces.get_basis_indices(B_1d, e)) for e in 1:n_elem]

    pbuf = ParticleBuf(p.P_DEG)
    nloc = p.P_DEG + 1
    lp_vals = zeros(nloc)
    lp_gids = zeros(Int, nloc)
    G_vals  = zeros(nloc)
    G_dxi   = zeros(nloc)
    G_gids  = zeros(Int, nloc)

    return Workspace(p, bp,
                     geo_1d, X⁰, n_dofs, n_elements,
                     M_lu, qrule_integrate,
                     ext1d, basis_start_1d,
                     pbuf,
                     lp_vals, lp_gids,
                     G_vals, G_dxi, G_gids)
end

# ## Particle location (1D)
struct ParticleLocation
    elem_id::Int
    xi::Points.CartesianPoints
    h::Float64
end

function locate_particle(ws::Workspace, v)
    (v <= ws.bp[1] || v >= ws.bp[end]) && return nothing
    i = searchsortedlast(ws.bp, v)
    h = ws.bp[i+1] - ws.bp[i]
    ξ = (v - ws.bp[i]) / h
    return ParticleLocation(i, Points.CartesianPoints(([ξ],)), h)
end

# ## Evaluation wrappers
evaluate(ws::Workspace, ff::Forms.AbstractFormField, e::Int) =
    Forms.evaluate(ff, e, ws.qrule_integrate.nodes)
evaluate(ws::Workspace, fs::Forms.AbstractFormSpace, elem_id::Int) =
    Forms.evaluate(fs, elem_id, ws.qrule_integrate.nodes)
evaluate(ws::Workspace, e::Int) = evaluate(ws, ws.X⁰, e)
evaluate(ws::Workspace, fs::Forms.AbstractFormSpace, loc::ParticleLocation) =
    Forms.evaluate(fs, loc.elem_id, loc.xi)
evaluate(ws::Workspace, ff::Forms.AbstractFormField, loc::ParticleLocation) =
    Forms.evaluate(ff, loc.elem_id, loc.xi)
evaluate(ws::Workspace, loc::ParticleLocation) = evaluate(ws, ws.X⁰, loc)

build_field(ws::Workspace, coeffs) = Forms.build_form_field(ws.X⁰, coeffs)
element_measure(ws::Workspace, e) = Geometry.get_element_measure(ws.geo_1d, e)

# ## Bernstein basis evaluation (degree p, point ξ ∈ [0,1])
function _bernstein_eval!(B::AbstractVector, dB::AbstractVector, p::Int, ξ::Float64)
    one_minus = 1.0 - ξ
    @inbounds for i in 0:p
        B[i+1] = binomial(p, i) * ξ^i * one_minus^(p-i)
    end
    if p == 0
        fill!(dB, 0.0)
    else
        @inbounds for i in 0:p
            bm1_left  = i > 0 ? binomial(p-1, i-1) * ξ^(i-1) * one_minus^(p-i)   : 0.0
            bm1_right = i < p ? binomial(p-1, i)   * ξ^i     * one_minus^(p-1-i) : 0.0
            dB[i+1] = p * (bm1_left - bm1_right)
        end
    end
    return nothing
end

# Fill `vals` and `gids` with φ_j(ξ) and global DOF indices on `loc`.
function fast_eval_particle!(ws::Workspace,
                              vals::AbstractVector, gids::AbstractVector,
                              loc::ParticleLocation)
    p_deg = ws.p.P_DEG
    ξ = loc.xi[1][1]
    i = loc.elem_id

    C = ws.ext1d[i]
    s = ws.basis_start_1d[i]

    _bernstein_eval!(ws.pbuf.B, ws.pbuf.dB, p_deg, ξ)
    mul!(ws.pbuf.phi, transpose(C), ws.pbuf.B)

    p1 = p_deg + 1
    @inbounds for j in 1:p1
        vals[j] = ws.pbuf.phi[j]
        gids[j] = s + j - 1
    end
    return p1
end

# Values + canonical-coord gradient ∂φ/∂ξ. Physical: ∂/∂v = (∂/∂ξ) / h.
function fast_eval_particle_grad!(ws::Workspace,
                                   vals::AbstractVector,
                                   dvals_dxi::AbstractVector,
                                   gids::AbstractVector,
                                   loc::ParticleLocation)
    p_deg = ws.p.P_DEG
    ξ = loc.xi[1][1]
    i = loc.elem_id

    C = ws.ext1d[i]
    s = ws.basis_start_1d[i]

    _bernstein_eval!(ws.pbuf.B, ws.pbuf.dB, p_deg, ξ)
    mul!(ws.pbuf.phi,  transpose(C), ws.pbuf.B)
    mul!(ws.pbuf.dphi, transpose(C), ws.pbuf.dB)

    p1 = p_deg + 1
    @inbounds for j in 1:p1
        vals[j]      = ws.pbuf.phi[j]
        dvals_dxi[j] = ws.pbuf.dphi[j]
        gids[j]      = s + j - 1
    end
    return p1
end

# Sample f_s on a 1D grid for plots. Points outside the domain → 0.
function evaluate_on_grid(ws::Workspace, field::Forms.FormField, v_grid::AbstractVector)
    F = zeros(length(v_grid))
    for (i, v) in enumerate(v_grid)
        loc = locate_particle(ws, v)
        isnothing(loc) && continue
        fv, _ = evaluate(ws, field, loc)
        F[i] = fv[1][1]
    end
    return F
end

export SimParameters, parse_overrides, print_summary, sample_initial_velocities
export Workspace, build_workspace
export ParticleLocation, locate_particle, evaluate, build_field, element_measure
export fast_eval_particle!, fast_eval_particle_grad!
export evaluate_on_grid

include("functions.jl")

export l2_project!, compute_entropy
export eval_fs_dfs_at_particles!, compute_moments, compute_A1A2, compute_LB_velocity!
export compute_negative_part_l1, compute_fs_minus_fp_l2

end # module
