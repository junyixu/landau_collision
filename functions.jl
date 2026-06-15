# 2D conservative Lenard-Bernstein physics + diagnostics.
# All take `ws::Workspace` as first argument.
#
# Drift (paper eq 23/30, generalised to 2D velocity space):
#   v̇_α = -ν ( ∇f_s(v_α)/f_s(v_α) + A + B v_α )
# with A ∈ ℝ² (momentum multipliers) and B ∈ ℝ (energy multiplier) chosen so the
# discrete momentum Σ w_α v̇_α and energy Σ w_α v_α·v̇_α are conserved exactly.

# Floor on f_s used in 1/f_s evaluation; f_s = 0 is a hard singularity.
const FS_FLOOR = 1e-30

# Limiter on the per-component log-density gradient g = ∇f_s/f_s. The physical
# |∇log f| for a Gaussian is bounded by v_max/σ_min² (≈ 24 for the v3 IC); an
# L²-spline Gibbs undershoot can instead drive f_s(v_α) → 0⁺ and make the raw
# ratio explode (~1e30), blowing up the implicit solve. Clamping g to ±G_MAX
# removes that singularity without distorting the bulk; conservation is still
# exact because the multipliers are solved from the (clamped) g.
const G_MAX = 100.0

# ## L² projection of weighted Dirac measure onto X⁰
function l2_project!(ws::Workspace, f_coeffs, v_parts, w_parts)
    rhs = zeros(ws.n_dofs)
    nloc = (ws.p.P_DEG + 1)^2
    for α in axes(v_parts, 1)
        loc = locate_particle(ws, v_parts[α, 1], v_parts[α, 2])
        isnothing(loc) && continue
        fast_eval_particle!(ws, ws.lp_vals, ws.lp_gids, loc)
        @inbounds for j in 1:nloc
            rhs[ws.lp_gids[j]] += w_parts[α] * ws.lp_vals[j]
        end
    end
    f_coeffs .= ws.M_lu \ rhs
    return nothing
end

# ## Entropy H_h = -∫ f_s log f_s dv  (sign convention: dH/dt ≥ 0)
function compute_entropy(ws::Workspace, field::Forms.FormField)
    S = 0.0
    for e in 1:ws.n_elements
        jac = element_measure(ws, e)
        fv, _ = evaluate(ws, field, e)
        for q in eachindex(ws.qrule_integrate.weights)
            f_val = fv[1][q]
            if f_val > FS_FLOOR
                S -= f_val * log(f_val) * ws.qrule_integrate.weights[q] * jac
            end
        end
    end
    return S
end

# ## Eval the clamped log-density gradient  g_α = ∇f_s(v_α)/f_s(v_α)  at every
# particle. Each component is limited to ±G_MAX (see G_MAX note). Out-of-domain
# particles → g = 0 (their drift becomes A + B v, keeping the moment-
# conservation algebra consistent).
function eval_loggrad_at_particles!(ws::Workspace, g::AbstractMatrix,
                                    v_parts, f_coeffs)
    nloc = (ws.p.P_DEG + 1)^2
    @inbounds for α in axes(v_parts, 1)
        loc = locate_particle(ws, v_parts[α, 1], v_parts[α, 2])
        if isnothing(loc)
            g[α, 1] = 0.0
            g[α, 2] = 0.0
            continue
        end
        fast_eval_particle_grad!(ws, ws.G_vals, ws.G_dxi1, ws.G_dxi2, ws.G_gids, loc)
        inv_h1 = 1.0 / loc.h1
        inv_h2 = 1.0 / loc.h2
        f = 0.0; d1 = 0.0; d2 = 0.0
        for j in 1:nloc
            c = f_coeffs[ws.G_gids[j]]
            f  += c * ws.G_vals[j]
            d1 += c * ws.G_dxi1[j] * inv_h1
            d2 += c * ws.G_dxi2[j] * inv_h2
        end
        invf = 1.0 / max(abs(f), FS_FLOOR)
        g[α, 1] = clamp(d1 * invf, -G_MAX, G_MAX)
        g[α, 2] = clamp(d2 * invf, -G_MAX, G_MAX)
    end
    return nothing
end

# ## Raw particle moments (sums, not per-particle averages):
#   n  = Σ w_α            (zeroth)
#   U1 = Σ w_α v_α1,  U2 = Σ w_α v_α2   (first, by component)
#   Q  = Σ w_α |v_α|²    (second, trace)
function compute_moments(v_parts, w_parts)
    n = 0.0; U1 = 0.0; U2 = 0.0; Q = 0.0
    @inbounds for α in axes(v_parts, 1)
        w = w_parts[α]
        a = v_parts[α, 1]; b = v_parts[α, 2]
        n  += w
        U1 += w * a
        U2 += w * b
        Q  += w * (a^2 + b^2)
    end
    return n, U1, U2, Q
end

# ## Drift multipliers A = (A1, A2), B  (2D generalisation of paper eq 30).
# Solves the 3×3 symmetric system enforcing momentum + energy conservation:
#   [ n   0   U1 ] [A1]     [ Sg1 ]
#   [ 0   n   U2 ] [A2]  = -[ Sg2 ]
#   [ U1  U2  Q  ] [B ]     [ P   ]
# where g_α (clamped log-gradient),  Sg = Σ w_α g_α,  P = Σ w_α v_α·g_α.
function compute_drift_multipliers(v_parts, w_parts, g, n, U1, U2, Q)
    Sg1 = 0.0; Sg2 = 0.0; P = 0.0
    @inbounds for α in axes(v_parts, 1)
        g1 = g[α, 1]; g2 = g[α, 2]
        w = w_parts[α]
        Sg1 += w * g1
        Sg2 += w * g2
        P   += w * (v_parts[α, 1] * g1 + v_parts[α, 2] * g2)
    end
    Mmat = [n 0.0 U1; 0.0 n U2; U1 U2 Q]
    rhs  = [-Sg1, -Sg2, -P]
    sol  = Mmat \ rhs
    return sol[1], sol[2], sol[3]   # A1, A2, B
end

# ## LB velocity update  v̇_α = -ν ( ∇f_s/f_s + A + B v_α )
# Sign: paper eq (23) ∂_t f = ∇·(a f), a = ν(∇f/f + A + B v); continuity forces
# v̇ = -a. Pure-heat limit (A=B=0) gives v̇ = -ν ∇f/f, drift down log-density
# gradient = diffusion, H_h increases (matches the 1D sign fix).
function compute_LB_velocity!(dot_v, v_parts, g, A1, A2, B, ν)
    @inbounds for α in axes(v_parts, 1)
        dot_v[α, 1] = -ν * (g[α, 1] + A1 + B * v_parts[α, 1])
        dot_v[α, 2] = -ν * (g[α, 2] + A2 + B * v_parts[α, 2])
    end
    return nothing
end

# ##############################################################################
# Diagnostics (2D)
# ##############################################################################

# (1) Negative-part L¹ norm of f_s:    ∫ max(-f_s, 0) dv
function compute_negative_part_l1(ws::Workspace, field::Forms.FormField)
    neg = 0.0
    for e in 1:ws.n_elements
        jac = element_measure(ws, e)
        fv, _ = evaluate(ws, field, e)
        for q in eachindex(ws.qrule_integrate.weights)
            f_val = fv[1][q]
            if f_val < 0.0
                neg += (-f_val) * ws.qrule_integrate.weights[q] * jac
            end
        end
    end
    return neg
end

# (2) ‖f_s − f_p‖₂ with f_p the element-constant histogram density.
function compute_fs_minus_fp_l2(ws::Workspace, field::Forms.FormField,
                                 v_parts::AbstractMatrix, w_parts::AbstractVector)
    elem_mass = zeros(ws.n_elements)
    for α in axes(v_parts, 1)
        loc = locate_particle(ws, v_parts[α, 1], v_parts[α, 2])
        isnothing(loc) && continue
        elem_mass[loc.elem_id] += w_parts[α]
    end

    sumsq = 0.0
    for e in 1:ws.n_elements
        jac = element_measure(ws, e)
        fp_e = elem_mass[e] / jac
        fv, _ = evaluate(ws, field, e)
        for q in eachindex(ws.qrule_integrate.weights)
            f_val = fv[1][q]
            d = f_val - fp_e
            sumsq += d^2 * ws.qrule_integrate.weights[q] * jac
        end
    end
    return sqrt(sumsq)
end
