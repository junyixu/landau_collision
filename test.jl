include("MantisWrappers.jl")
using .MantisWrappers
loc = locate_particle(0.3, -0.7)
# 旧路径
evals_old, idx_old = MantisWrappers.evaluate(loc)
lb_old, idx_old2 = MantisWrappers.evaluate_basis_derivatives(loc, 1)
# 新路径
v = zeros((P_DEG+1)^2); g = zeros(Int, (P_DEG+1)^2)
MantisWrappers.fast_eval_particle!(v, g, loc)
@assert maximum(abs, v .- vec(evals_old[1])) < 1e-12
@assert Set(g) == Set(idx_old[1])
