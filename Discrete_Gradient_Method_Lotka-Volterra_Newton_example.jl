"""
Discrete Gradient Method — Lotka-Volterra example (Newton solver)

System:  ẋ = x(α - βy),  ẏ = y(-γ + δx)
First integral: I(x,y) = δx - γ ln(x) + βy - α ln(y)
Skew-gradient form: ż = S(z) ∇I(z),  S^T = -S
"""

using LinearAlgebra
using ForwardDiff

const α = 2.0 / 3.0
const β = 4.0 / 3.0
const γ_ = 1.0
const δ_ = 1.0

function I_func(z)
    x, y = z
    return δ_ * x - γ_ * log(x) + β * y - α * log(y)
end

function ∇I(z)
    x, y = z
    return [δ_ - γ_ / x, β - α / y]
end

function S_matrix(z)
    x, y = z
    return [0.0 -x*y;
        x*y 0.0]
end

function discrete_gradient_gonzalez(z0, z1)
    ẑ = 0.5 * (z0 + z1)
    Δz = z1 - z0
    g = ∇I(ẑ)
    nrm2 = dot(Δz, Δz)
    if nrm2 < 1e-30
        return g
    end
    ΔI = I_func(z1) - I_func(z0)
    correction = (ΔI - dot(g, Δz)) / nrm2
    return g + correction * Δz
end

function residual(z1, z0, Δt)
    ẑ = 0.5 * (z0 + z1)
    S̃ = S_matrix(ẑ)
    ḡ = discrete_gradient_gonzalez(z0, z1)
    return z1 - z0 - Δt * S̃ * ḡ
end

function step_discrete_gradient(z0, Δt; maxiter=20, tol=1e-13)
    f0 = S_matrix(z0) * ∇I(z0)
    z1 = z0 + Δt * f0  # explicit Euler initial guess

    for k in 1:maxiter
        F = residual(z1, z0, Δt)
        if norm(F) < tol
            return z1
        end
        J = ForwardDiff.jacobian(z -> residual(z, z0, Δt), z1)
        z1 = z1 - J \ F
    end

    @warn "Newton did not converge, residual = $(norm(residual(z1, z0, Δt)))"
    return z1
end

function rk4_step(z, Δt)
    f(z) = [z[1] * (α - β * z[2]),
        z[2] * (-γ_ + δ_ * z[1])]
    k1 = f(z)
    k2 = f(z + 0.5Δt * k1)
    k3 = f(z + 0.5Δt * k2)
    k4 = f(z + Δt * k3)
    return z + (Δt / 6) * (k1 + 2k2 + 2k3 + k4)
end

function main()
    z0 = [1.0, 1.0]
    Δt = 0.05
    T = 100.0
    N = round(Int, T / Δt)
    I0 = I_func(z0)

    z_dg = copy(z0)
    z_rk = copy(z0)
    max_err_dg = 0.0
    max_err_rk = 0.0

    for n in 1:N
        z_dg = step_discrete_gradient(z_dg, Δt)
        err_dg = abs(I_func(z_dg) - I0)
        max_err_dg = max(max_err_dg, err_dg)

        z_rk = rk4_step(z_rk, Δt)
        err_rk = abs(I_func(z_rk) - I0)
        max_err_rk = max(max_err_rk, err_rk)
    end

    println("=== After T = $T, Δt = $Δt, N = $N ===")
    println("Discrete gradient  final |ΔI| = $(abs(I_func(z_dg) - I0))")
    println("Discrete gradient  max   |ΔI| = $max_err_dg")
    println("RK4                final |ΔI| = $(abs(I_func(z_rk) - I0))")
    println("RK4                max   |ΔI| = $max_err_rk")
end

main()
