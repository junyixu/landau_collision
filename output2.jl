begin
include("MantisWrappers.jl")
using .MantisWrappers
using LinearAlgebra
using GLMakie
using Random
end;
include("parameters.jl");
println("V ∈ [$V_MIN, $V_MAX],  p=$P_DEG, k=$K_REG,  $N_ELEM×$N_ELEM elements")
println("N_particles=$N_PARTICLES,  σ₁=$σ₁, σ₂=$σ₂,  Δt=$DT,  N_steps=$N_STEPS")
Random.seed!(42)
v_particles = zeros(N_PARTICLES, 2)
v_particles[:, 1] .= σ₁ * randn(N_PARTICLES)
v_particles[:, 2] .= σ₂ * randn(N_PARTICLES)
w_particles = fill(1.0 / N_PARTICLES, N_PARTICLES)
function l2_project!(f_coeffs, v_parts, w_parts)
    rhs = zeros(n_dofs)
    for α in axes(v_parts, 1)
        loc = locate_particle(v_parts[α, 1], v_parts[α, 2])
        isnothing(loc) && continue

        # φ_k(v_α): evaluate all basis functions at particle position
        xi = make_point(loc.ξ1, loc.ξ2)
        evals, indices = evaluate(X⁰, loc.elem_id, xi)

        # b_k += w_α · φ_k(v_α)
        for (j, gidx) in enumerate(indices[1])
            rhs[gidx] += w_parts[α] * evals[1][1, j]
        end
    end
    # M f = b
    f_coeffs .= M_lu \ rhs
end
f_coeffs = zeros(n_dofs)
l2_project!(f_coeffs, v_particles, w_particles)
function compute_entropy(coeffs)
    field = build_field(coeffs)        # f_s(v) = Σ_i f_i φ_i(v)
    xi_q, wq = quadrature_nodes_weights()
    S = 0.0
    for e in 1:n_elements
        jac = element_measure(e)       # |J_e|
        fv, _ = evaluate(field, e, xi_q)
        for q in eachindex(wq)
            f_val = fv[1][q]           # f_s(v_q^e)
            f2 = f_val^2
            if f2 > 1e-30
                # S -= f_s log(f_s) · w_q · |J_e|
                S -= f_val * 0.5*log(f2) * wq[q] * jac
            end
        end
    end
    return S
end
entropy_history = Float64[]
push!(entropy_history, compute_entropy(f_coeffs))
println("Initial entropy S_h = $(entropy_history[end])")
function compute_r!(r, coeffs)
    fill!(r, 0.0)
    field = build_field(coeffs)
    xi_q, wq = quadrature_nodes_weights()
    for e in 1:n_elements
        jac = element_measure(e)
        fv, _ = evaluate(field, e, xi_q)                  # f_s(v_q^e)
        evals, indices = evaluate(X⁰, e, xi_q)           # φ_i(v_q^e)
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
        xi = make_point(loc.ξ1, loc.ξ2)

        # nderivatives=1 → local_basis[2][d][1] = ∂φ/∂ξ_d
        local_basis, indices = evaluate_basis_derivatives(loc.elem_id, xi, 1)
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
r_vec = zeros(n_dofs)
L_vec = zeros(n_dofs)
G = zeros(N_PARTICLES, 2)
dot_v = zeros(N_PARTICLES, 2)

snapshot_steps = [0, N_STEPS ÷ 4, N_STEPS ÷ 2, N_STEPS]
snapshots = Dict{Int,Matrix{Float64}}()
snapshots[0] = copy(v_particles)

for step in 1:N_STEPS
    compute_r!(r_vec, f_coeffs)
    L_vec .= M_lu \ r_vec
    compute_G!(G, v_particles, L_vec)
    compute_collision!(dot_v, v_particles, w_particles, G)
    v_particles .+= DT .* dot_v
    l2_project!(f_coeffs, v_particles, w_particles)
    push!(entropy_history, compute_entropy(f_coeffs))

    if step in snapshot_steps
        snapshots[step] = copy(v_particles)
    end

    step % 25 == 0 && println("Step $step/$N_STEPS  S = $(round(entropy_history[end]; digits=6))")
end
begin
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
ax_H = Axis(fig[n_rows + 1, 1:2];
    xlabel="time step", ylabel="H_h = -S_h",
    title="Boltzmann H-function (should decrease monotonically)")
lines!(ax_H, 0:length(H_history)-1, H_history; color=:red, linewidth=2)
save("landau_collision_2d_2.png", fig)
end;
