"""
Discrete Gradient Method — Lotka-Volterra example

System:  ẋ = x(α - βy),  ẏ = y(-γ + δx)
First integral: I(x,y) = δx - γ ln(x) + βy - α ln(y)
Skew-gradient form: ż = S(z) ∇I(z),  S^T = -S

We use the Gonzalez midpoint discrete gradient + midpoint S̃.
The resulting scheme is implicit and second-order.
"""

using LinearAlgebra

# --- Parameters ---
const α = 2.0 / 3.0
const β = 4.0 / 3.0
const γ = 1.0
const δ = 1.0

"""
# --- First integral and its gradient ---
"""
function I_func(z)
    x, y = z
    return δ * x - γ * log(x) + β * y - α * log(y)
end

function ∇I(z)
    x, y = z
    return [δ - γ / x, β - α / y]
end

"""
# --- Skew matrix S(z) such that f(z) = S(z) ∇I(z) ---
For Lotka-Volterra:  S = [0  -xy;  xy  0]
Verify: S ∇I = [-xy(β - α/y), xy(δ - γ/x)] = [-xβy + αx, xyδ - γy]
              = [x(α - βy), y(δx - γ)]  ✓
"""
function S_matrix(z)
    x, y = z
    return [0.0 -x*y;
        x*y 0.0]
end

"""
# Gonzalez midpoint discrete gradient ---
∇̄I(z, z') = ∇I(ẑ) + [I(z') - I(z) - ∇I(ẑ)⋅(z'-z)] / |z'-z|² * (z'-z)
where ẑ = (z + z')/2
"""
function discrete_gradient_gonzalez(z0, z1)
    ẑ = 0.5 * (z0 + z1)
    Δz = z1 - z0
    g = ∇I(ẑ)
    ΔI = I_func(z1) - I_func(z0)
    correction = (ΔI - dot(g, Δz)) / dot(Δz, Δz)
    return g + correction * Δz
end

"""
# One step of the discrete gradient method ---
Solve: (z1 - z0)/Δt = S̃(z0,z1) ⋅ ∇̄I(z0,z1)
where S̃ = S((z0+z1)/2)
This is implicit → use fixed-point iteration
"""
function step_discrete_gradient(z0, Δt; maxiter=50, tol=1e-12)
    z1 = copy(z0)  # initial guess

    for _ in 1:maxiter
        ẑ = 0.5 * (z0 + z1)
        S̃ = S_matrix(ẑ)
        ḡ = discrete_gradient_gonzalez(z0, z1)
        z1_new = z0 + Δt * S̃ * ḡ

        if norm(z1_new - z1) < tol
            return z1_new
        end
        z1 = z1_new
    end

    @warn "Fixed-point iteration did not converge"
    return z1
end

"""
For comparison: explicit RK4 ---
"""
function rk4_step(z, Δt)
    f(z) = [z[1] * (α - β * z[2]),
        z[2] * (-γ + δ * z[1])]
    k1 = f(z)
    k2 = f(z + 0.5Δt * k1)
    k3 = f(z + 0.5Δt * k2)
    k4 = f(z + Δt * k3)
    return z + (Δt / 6) * (k1 + 2k2 + 2k3 + k4)
end

"""
Run both methods and compare integral preservation ---
"""
function main()
    z0 = [1.0, 1.0]
    Δt = 0.05
    T = 100.0
    N = round(Int, T / Δt)

    I0 = I_func(z0)

    # Discrete gradient method
    z_dg = copy(z0)
    I_err_dg = Float64[]

    # RK4
    z_rk = copy(z0)
    I_err_rk = Float64[]

    for _ in 1:N
        z_dg = step_discrete_gradient(z_dg, Δt)
        push!(I_err_dg, abs(I_func(z_dg) - I0))

        z_rk = rk4_step(z_rk, Δt)
        push!(I_err_rk, abs(I_func(z_rk) - I0))
    end

    t = Δt * (1:N)

    println("=== After T = $T ===")
    println("Discrete gradient |ΔI| = $(I_err_dg[end])  (should be ~machine epsilon)")
    println("RK4               |ΔI| = $(I_err_rk[end])  (drifts)")
    println()
    println("Max |ΔI| discrete gradient: $(maximum(I_err_dg))")
    println("Max |ΔI| RK4:              $(maximum(I_err_rk))")
end

main()
