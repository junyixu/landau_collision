function l2_project!(f_coeffs, v_parts, w_parts)
    rhs = zeros(n_dofs)
    for α in axes(v_parts, 1)
        loc = locate_particle(v_parts[α, 1], v_parts[α, 2])
        isnothing(loc) && continue

        # φ_k(v_α): evaluate all basis functions at particle position
        evals, indices = evaluate(X⁰, loc)

        # b_k += w_α · φ_k(v_α)
        for (j, gidx) in enumerate(indices[1])
            rhs[gidx] += w_parts[α] * evals[1][1, j]
        end
    end
    # M f = b
    f_coeffs .= M_lu \ rhs
end
function compute_entropy(coeffs)
    field = build_field(coeffs)        # f_s(v) = Σ_i f_i φ_i(v)
    S = 0.0
    for e in 1:n_elements
        jac = element_measure(e)       # |J_e|
        fv, _ = evaluate(field, e)
        for q in eachindex(wq)
            f_val = fv[1][q]           # f_s(v_q^e)
            if f_val > 1e-30
                # S -= f_s log(f_s) · w_q · |J_e|
                S -= f_val * log(f_val) * wq[q] * jac
            end
        end
    end
    return S
end
function compute_r!(r, coeffs)
    fill!(r, 0.0)
    field = build_field(coeffs)
    for e in 1:n_elements
        jac = element_measure(e)
        fv, _ = evaluate(field, e)                         # f_s(v_q^e)
        evals, indices = evaluate(X⁰, e)                  # φ_i(v_q^e)
        for q in eachindex(wq)
            f_val = fv[1][q]
            integrand = f_val > 1e-30 ? (1 + log(f_val)) : 0.0
            for (j, gidx) in enumerate(indices[1])
                # r_i += φ_i(v_q) · (1 + log f_s(v_q)) · w_q · |J_e|
                r[gidx] += integrand * evals[1][q, j] * wq[q] * jac
            end
        end
    end
end
function compute_G!(G, v_parts, L_vec)
    fill!(G, 0.0)
    for α in axes(v_parts, 1)
        loc = locate_particle(v_parts[α, 1], v_parts[α, 2])
        isnothing(loc) && continue

        # nderivatives=1 → local_basis[2][d][1] = ∂φ/∂ξ_d
        local_basis, indices = evaluate_basis_derivatives(loc, 1)
        dφ_dξ1 = local_basis[2][1][1]   # (n_pts × n_basis)
        dφ_dξ2 = local_basis[2][2][1]

        for (j, gidx) in enumerate(indices[1])
            # G_α += L_k · ∇φ_k(v_α),  chain rule: ∂φ/∂v_d = (∂φ/∂ξ_d) / h_d
            G[α, 1] += L_vec[gidx] * dφ_dξ1[1, j] / loc.h1
            G[α, 2] += L_vec[gidx] * dφ_dξ2[1, j] / loc.h2
        end
    end
end
function compute_collision!(dot_v, v_parts, w_parts, G)
    fill!(dot_v, 0.0)
    N = size(v_parts, 1)
    for γ in 1:N
        vγ1, vγ2 = v_parts[γ, 1], v_parts[γ, 2]
        (vγ1 <= V_MIN || vγ1 >= V_MAX || vγ2 <= V_MIN || vγ2 >= V_MAX) && continue
        Gγ1, Gγ2 = G[γ, 1], G[γ, 2]
        acc1, acc2 = 0.0, 0.0
        for α in 1:N
            γ == α && continue
            vα1, vα2 = v_parts[α, 1], v_parts[α, 2]
            (vα1 <= V_MIN || vα1 >= V_MAX || vα2 <= V_MIN || vα2 >= V_MAX) && continue

            # Δv = v_γ - v_α
            d1 = vγ1 - vα1
            d2 = vγ2 - vα2
            dist2 = d1^2 + d2^2
            dist2 < 1e-24 && continue
            dist = sqrt(dist2)

            # g = G_γ - G_α
            g1 = Gγ1 - G[α, 1]
            g2 = Gγ2 - G[α, 2]

            # [U·g]_i = (g_i - Δv̂_i (Δv̂·g)) / |Δv|
            dv_dot_g = (d1 * g1 + d2 * g2) / dist2    # (Δv·g) / |Δv|²
            inv_dist = 1.0 / dist                       # 1/|Δv|
            acc1 += w_parts[α] * (g1 - d1 * dv_dot_g) * inv_dist
            acc2 += w_parts[α] * (g2 - d2 * dv_dot_g) * inv_dist
        end
        dot_v[γ, 1] = acc1
        dot_v[γ, 2] = acc2
    end
end
