# 1D conservative Lenard-Bernstein physics + diagnostics.
# All take `ws::Workspace` as first argument.

# Floor on f_s used in 1/f_s evaluation. Particles in regions where f_s drops
# below this floor get f_s clamped; mostly catches dying tails of bi-Maxwellian
# valleys and Gibbs undershoots. Paper §5.5 also notes f_s = 0 is a hard
# singularity of the operator.
const FS_FLOOR = 1e-30

# ## L² projection of weighted Dirac measure onto X⁰
function l2_project!(ws::Workspace, f_coeffs, v_parts, w_parts)
    rhs = zeros(ws.n_dofs)
    nloc = ws.p.P_DEG + 1
    for α in eachindex(v_parts)
        loc = locate_particle(ws, v_parts[α])
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
# Paper §5 uses S = ∫ f log f, so H_h = -S; same numerical content.
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

# ## Eval f_s and f_s' at every particle location.
# Out-of-domain particles → f = FS_FLOOR, df = 0 (their contribution to a(v,f_s)
# becomes A1 + A2*v which keeps moment-conservation algebra consistent).
function eval_fs_dfs_at_particles!(ws::Workspace,
                                    f_vals::AbstractVector, df_vals::AbstractVector,
                                    v_parts, f_coeffs)
    nloc = ws.p.P_DEG + 1
    @inbounds for α in eachindex(v_parts)
        loc = locate_particle(ws, v_parts[α])
        if isnothing(loc)
            f_vals[α] = FS_FLOOR
            df_vals[α] = 0.0
            continue
        end
        fast_eval_particle_grad!(ws, ws.G_vals, ws.G_dxi, ws.G_gids, loc)
        inv_h = 1.0 / loc.h
        f = 0.0; df = 0.0
        for j in 1:nloc
            c = f_coeffs[ws.G_gids[j]]
            f  += c * ws.G_vals[j]
            df += c * ws.G_dxi[j] * inv_h
        end
        f_vals[α]  = max(f, FS_FLOOR)
        df_vals[α] = df
    end
    return nothing
end

# ## Particle moments  n_h = Σ w_α,  n_h u_h = Σ w_α v_α,  n_h ε_h = Σ w_α v_α²
function compute_moments(v_parts, w_parts)
    n = 0.0; nu = 0.0; ne = 0.0
    @inbounds for α in eachindex(v_parts)
        wα = w_parts[α]; vα = v_parts[α]
        n  += wα
        nu += wα * vα
        ne += wα * vα^2
    end
    return n, nu/n, ne/n   # returns (n_h, u_h, ε_h)
end

# ## Discrete A1, A2  (paper eq 30)
# Inputs:
#   v_parts, w_parts   : particle velocities + weights
#   f_vals, df_vals    : f_s(v_α), f_s'(v_α)
#   n_h, u_h, eps_h    : particle moments
function compute_A1A2(v_parts, w_parts, f_vals, df_vals, n_h, u_h, eps_h)
    s1 = 0.0; s2 = 0.0
    @inbounds for α in eachindex(v_parts)
        ratio = df_vals[α] / f_vals[α]
        wα = w_parts[α]
        s1 += wα * (u_h * v_parts[α] - eps_h) * ratio
        s2 += wα * (u_h - v_parts[α]) * ratio
    end
    denom = n_h * eps_h - n_h * u_h^2     # = n_h * (ε_h - u_h²)
    A1 = s1 / denom
    A2 = s2 / denom
    return A1, A2
end

# ## LB velocity update  v̇_α = -ν (f_s'(v_α)/f_s(v_α) + A1 + A2 v_α)
# Sign rationale: paper eq (23) is ∂_t f = ∂_v(a f) with a = ν(f'/f + A1 + A2 v).
# Continuity ∂_t f + ∂_v(v̇ f) = 0 forces v̇ = -a. Paper eq (24) drops the minus
# (typo); we follow the physical sign. Sanity: pure-heat limit (A1=A2=0) gives
# v̇ = -ν f'/f, particles drift down log-density gradient = diffusion. With the
# wrong (paper) sign particles concentrate toward peak (anti-diffusion) and H_h
# decreases — matches what we observed in the smoke test before this fix.
function compute_LB_velocity!(dot_v, v_parts, f_vals, df_vals, A1, A2, ν)
    @inbounds for α in eachindex(v_parts)
        dot_v[α] = -ν * (df_vals[α] / f_vals[α] + A1 + A2 * v_parts[α])
    end
    return nothing
end

# ##############################################################################
# Diagnostics (1D)
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
                                 v_parts::AbstractVector, w_parts::AbstractVector)
    elem_mass = zeros(ws.n_elements)
    for α in eachindex(v_parts)
        loc = locate_particle(ws, v_parts[α])
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
