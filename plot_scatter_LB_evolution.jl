#! /usr/bin/env -S julia --color=yes --startup-file=no
# Montage of LB particle scatter across ALL snapshot steps in one figure, to see
# the time evolution (initial → intermediate → final) of any banding/clustering.
# Reads particle_snapshots_<suffix>.csv once; one bulk-zoom panel per snap step.
#
#   julia --project=. plot_scatter_LB_evolution.jl [suffix]
#
# Default suffix=LB2D_v3. Also dumps a standalone step-0 (initial-condition) PNG.

using CairoMakie
using DelimitedFiles

suffix = length(ARGS) >= 1 ? ARGS[1] : "LB2D_v3"

# v3 baseline mesh breakpoints (overlay)
bp1 = [-6.0; -5.0; LinRange(-4.0, 4.0, 17); 5.0; 6.0]
bp2 = [-6.0; LinRange(-2.5, 2.5, 26); 6.0]

snap_csv = "particle_snapshots_$(suffix).csv"
isfile(snap_csv) || error("Snapshot CSV not found: $snap_csv")

data, _ = readdlm(snap_csv, ','; header=true)
steps_all = Int.(@view data[:, 1])
uniq_steps = sort(unique(steps_all))
println("Snapshot steps: ", uniq_steps)

# ---- standalone step-0 (initial condition) two-panel figure -----------------
function plot_single(step)
    mask = steps_all .== step
    v1 = Float64.(data[mask, 4]); v2 = Float64.(data[mask, 5])
    t  = Float64(data[findfirst(mask), 2]); N = length(v1)
    fig = Figure(; size=(1500, 700))
    ax = Axis(fig[1, 1]; xlabel="v₁", ylabel="v₂", aspect=DataAspect(),
        title="LB scatter  (suffix=$suffix, step=$step, t=$t, N=$N)")
    vlines!(ax, bp1; color=(:gray, 0.35), linewidth=0.5)
    hlines!(ax, bp2; color=(:gray, 0.35), linewidth=0.5)
    scatter!(ax, v1, v2; markersize=2, color=(:navy, 0.25))
    xlims!(ax, bp1[1], bp1[end]); ylims!(ax, bp2[1], bp2[end])
    ax2 = Axis(fig[1, 2]; xlabel="v₁", ylabel="v₂", aspect=DataAspect(),
        title="bulk zoom  v₁∈[-4,4], v₂∈[-2.5,2.5]")
    vlines!(ax2, bp1; color=(:gray, 0.4), linewidth=0.6)
    hlines!(ax2, bp2; color=(:gray, 0.4), linewidth=0.6)
    scatter!(ax2, v1, v2; markersize=3, color=(:navy, 0.3))
    xlims!(ax2, -4, 4); ylims!(ax2, -2.5, 2.5)
    name = "scatter_$(suffix)_step$(lpad(step, 5, '0')).png"
    save(name, fig); println("Saved $name  (N=$N)")
end

plot_single(0)   # initial-condition standalone

# ---- evolution montage: bulk-zoom panel per snapshot step -------------------
nstep = length(uniq_steps)
ncol  = 5
nrow  = cld(nstep, ncol)
fig = Figure(; size=(300 * ncol, 300 * nrow))
Label(fig[0, 1:ncol],
      "LB particle-scatter evolution  (suffix=$suffix, bulk zoom v₁∈[-4,4] v₂∈[-2.5,2.5])";
      fontsize=18, tellwidth=false)

for (i, step) in enumerate(uniq_steps)
    r = cld(i, ncol); c = mod1(i, ncol)
    mask = steps_all .== step
    v1 = Float64.(data[mask, 4]); v2 = Float64.(data[mask, 5])
    t  = Float64(data[findfirst(mask), 2])
    ax = Axis(fig[r, c]; aspect=DataAspect(),
              title="step=$step  t=$(round(t; digits=3))",
              titlesize=12)
    vlines!(ax, bp1; color=(:gray, 0.3), linewidth=0.4)
    hlines!(ax, bp2; color=(:gray, 0.3), linewidth=0.4)
    scatter!(ax, v1, v2; markersize=1.5, color=(:navy, 0.25))
    xlims!(ax, -4, 4); ylims!(ax, -2.5, 2.5)
    hidedecorations!(ax; label=false)
end

rowsize!(fig.layout, 0, Fixed(40))
mont = "scatter_evolution_$(suffix).png"
save(mont, fig)
println("Saved $mont  ($nstep panels)")
