# 1D f_s overlay across mesh sweep variants at selected snapshot steps.
# Usage:
#   julia --project=. plot_mesh_sweep_compare.jl [steps]
# steps default = "0,500,2000,6250"
include("MantisWrappers.jl")
using .MantisWrappers
using CairoMakie
CairoMakie.activate!()

steps = length(ARGS) >= 1 ? parse.(Int, split(ARGS[1], ',')) : [0, 500, 2000, 6250]
tags  = ["LB_mesh_align", "LB_mesh_offset", "LB_mesh_fine", "LB_mesh_peakdense"]
colors = Dict("LB_mesh_align" => :tomato,
              "LB_mesh_offset" => :steelblue,
              "LB_mesh_fine" => :seagreen,
              "LB_mesh_peakdense" => :purple)

function parse_header(path)
    bp = Float64[]; ndof = 0; data_start = 0
    open(path) do io
        for (i, ln) in enumerate(eachline(io))
            if startswith(ln, "# bp=")
                bp = parse.(Float64, split(ln[length("# bp=")+1:end], ','))
            elseif startswith(ln, "# n_dofs=")
                ndof = parse(Int, ln[length("# n_dofs=")+1:end])
            elseif ln == "coeff"
                data_start = i + 1
                break
            end
        end
    end
    return bp, ndof, data_start
end

function load_coeffs(path, ndof, ds)
    c = Vector{Float64}(undef, ndof)
    open(path) do io
        for _ in 1:ds-1; readline(io); end
        for k in 1:ndof
            c[k] = parse(Float64, readline(io))
        end
    end
    c
end

# Build per-tag Workspace + cache evaluations on common dense grid
ε = 1e-6
vg = range(-5.0+ε, 5.0-ε; length=1200)  # zoom on bulk; bi-Max support

per_tag_fields = Dict{String, Dict{Int, Vector{Float64}}}()
per_tag_bp = Dict{String, Vector{Float64}}()
for tag in tags
    snap0 = "fs_snapshot_$(tag)_step$(lpad(steps[1], 5, '0')).csv"
    isfile(snap0) || error("missing $snap0")
    bp, ndof, _ = parse_header(snap0)
    per_tag_bp[tag] = bp
    p = SimParameters(bp=bp)
    ws = build_workspace(p)
    fdict = Dict{Int, Vector{Float64}}()
    for s in steps
        path = "fs_snapshot_$(tag)_step$(lpad(s, 5, '0')).csv"
        isfile(path) || (println("skip missing $path"); continue)
        _, _, ds = parse_header(path)
        c = load_coeffs(path, ndof, ds)
        fld = build_field(ws, c)
        F = evaluate_on_grid(ws, fld, vg)
        fdict[s] = F
        println("tag=$tag step=$s  fmin=$(minimum(F))  fmax=$(maximum(F))")
    end
    per_tag_fields[tag] = fdict
end

# 4-panel: one per step. Overlay 4 mesh variants in each panel.
fig = Figure(; size=(1300, 900))
for (k, s) in enumerate(steps)
    row = (k - 1) ÷ 2 + 1
    col = (k - 1) % 2 + 1
    ax = Axis(fig[row, col];
        xlabel="v", ylabel="f_s(v)",
        title="step=$s   t=$(round(s*8e-4; digits=3))")
    for tag in tags
        haskey(per_tag_fields[tag], s) || continue
        F = per_tag_fields[tag][s]
        lines!(ax, vg, F; label=replace(tag, "LB_mesh_"=>""),
               color=colors[tag], linewidth=1.5)
    end
    hlines!(ax, [0.0]; color=:black, linewidth=0.5, linestyle=:dash)
    if k == 1
        axislegend(ax; position=:rt, framevisible=true)
    end
end

Label(fig[0, 1:2],
    "f_s mesh sweep — bi-Max IC (peaks ±2)";
    fontsize=18, tellwidth=false)

out = "mesh_sweep_overlay.png"
save(out, fig)
println("Saved $out")

# Zoom on right peak v ∈ [1.0, 3.0] to expose honeycomb modulation
fig2 = Figure(; size=(1300, 900))
for (k, s) in enumerate(steps)
    row = (k - 1) ÷ 2 + 1
    col = (k - 1) % 2 + 1
    ax = Axis(fig2[row, col];
        xlabel="v", ylabel="f_s(v)",
        title="step=$s   peak-zoom  v∈[1,3]")
    for tag in tags
        haskey(per_tag_fields[tag], s) || continue
        F = per_tag_fields[tag][s]
        mask = (vg .>= 1.0) .& (vg .<= 3.0)
        lines!(ax, vg[mask], F[mask]; label=replace(tag, "LB_mesh_"=>""),
               color=colors[tag], linewidth=1.5)
        # mark mesh knots in this zoom window
        bp = per_tag_bp[tag]
        bp_in = bp[(bp .>= 1.0) .& (bp .<= 3.0)]
        vlines!(ax, bp_in; color=colors[tag], linewidth=0.3, alpha=0.35)
    end
    if k == 1
        axislegend(ax; position=:rt, framevisible=true)
    end
end
Label(fig2[0, 1:2],
    "f_s peak zoom v∈[1,3] (vertical bars = knot positions)";
    fontsize=18, tellwidth=false)

out2 = "mesh_sweep_peakzoom.png"
save(out2, fig2)
println("Saved $out2")
