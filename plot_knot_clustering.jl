# Detect secular particle clustering onto grid knots in long LB runs.
# Reads particle_snapshots_<tag>.csv (cols: step,time,particle_idx,v) + mesh bp.
# Metrics:
#   (1) fine histogram of final-step particle v, knot positions overlaid
#   (2) mean distance to nearest knot vs uniform-random null (Monte-Carlo)
#   (3) phase variable θ = frac((v-knot0)/h) histogram — flat if no clustering,
#       peaked at 0/1 if particles pile on knots, peaked at 0.5 if avoid knots.
# Usage: julia --project=. plot_knot_clustering.jl <tag> <bp_preset.jl>
include("MantisWrappers.jl")
using .MantisWrappers
using DelimitedFiles, Printf, Random, Statistics
import CairoMakie as M
M.activate!()

length(ARGS) >= 2 || error("usage: plot_knot_clustering.jl <tag> <bp_preset.jl>")
tag = ARGS[1]; preset = ARGS[2]
include(preset)            # defines PARAMS
bp = PARAMS.bp
h_unif = bp[2] - bp[1]     # nominal spacing (uniform meshes)

snap = "runs/$tag/particle_snapshots_$tag.csv"
isfile(snap) || (snap = "particle_snapshots_$tag.csv")
isfile(snap) || error("missing particle snapshots for $tag")

hdr = split(strip(readline(snap)), ",")
A = readdlm(snap, ',', Float64; skipstart=1)
cs=findfirst(==("step"),hdr); cv=findfirst(==("v"),hdr)
steps = unique(A[:,cs])
last_step = maximum(steps)
vf = A[A[:,cs].==last_step, cv]
v0 = A[A[:,cs].==minimum(steps), cv]
@printf "tag=%s  N=%d  steps %d..%d\n" tag length(vf) Int(minimum(steps)) Int(last_step)

# nearest-knot distance metric
nearest_dist(v, knots) = [minimum(abs.(x .- knots)) for x in v]
df = nearest_dist(vf, bp)
# null: uniform-random in [bp[1],bp[end]] gives E[dist]; but particles are Maxwellian
# so compare to particles drawn from SAME marginal shuffled in phase. Use Monte-Carlo:
# resample N velocities from a Maxwellian matching vf moments, measure dist.
mu = mean(vf); sig = std(vf)
rng = MersenneTwister(1)
null_means = Float64[]
for _ in 1:200
    vr = mu .+ sig .* randn(rng, length(vf))
    vr = clamp.(vr, bp[1]+1e-6, bp[end]-1e-6)
    push!(null_means, mean(nearest_dist(vr, bp)))
end
obs = mean(df); nm = mean(null_means); ns = std(null_means)
z = (obs - nm) / ns
@printf "mean dist to knot: obs=%.5f  null=%.5f±%.5f  z=%+.2f  (z<0 cluster, z>0 avoid)\n" obs nm ns z

# phase variable on uniform mesh
θ = mod.((vf .- bp[1]) ./ h_unif, 1.0)

fig = M.Figure(size=(1200,900))
ax1 = M.Axis(fig[1,1], xlabel="v", ylabel="count",
    title="$tag  final-step particle histogram (red bars = knots, h=$(round(h_unif;digits=3)))")
M.hist!(ax1, vf, bins=200, color=(:steelblue,0.7))
M.vlines!(ax1, bp, color=:red, linewidth=0.5, alpha=0.5)
M.xlims!(ax1, -6, 6)

ax2 = M.Axis(fig[2,1], xlabel="θ = frac((v-v0)/h)", ylabel="count",
    title="phase histogram — flat=no clustering, peak@0/1=on-knot pile, peak@0.5=avoid")
M.hist!(ax2, θ, bins=40, color=(:purple,0.7))
M.hlines!(ax2, [length(vf)/40], color=:black, linestyle=:dash)  # uniform expectation

ax3 = M.Axis(fig[3,1], xlabel="v", ylabel="density",
    title="IC (gray) vs final (blue) — overall relaxation to Maxwellian")
M.hist!(ax3, v0, bins=120, color=(:gray,0.5), normalization=:pdf, label="t=0")
M.hist!(ax3, vf, bins=120, color=(:steelblue,0.6), normalization=:pdf, label="final")
vv = range(-6,6,length=400)
M.lines!(ax3, vv, (1/(sig*sqrt(2pi))).*exp.(-(vv.-mu).^2 ./(2sig^2)),
    color=:red, linewidth=2, label="Maxwellian fit")
M.xlims!(ax3, -6, 6); M.axislegend(ax3, position=:rt)

out = "knot_clustering_$tag.png"
M.save(out, fig); println("Saved $out")
