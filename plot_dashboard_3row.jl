# 3-row dashboard from conservation_history: H_h + energy/momentum rel. error.
# Dedup keeps LAST occurrence per step (restart segments overlap).
#   julia --project=. plot_dashboard_3row.jl [tag] [out_suffix]
using GLMakie, DelimitedFiles
GLMakie.activate!()

# PPT-sized fonts
update_theme!(
    Axis = (titlesize=28, xlabelsize=28, ylabelsize=28,
            xticklabelsize=22, yticklabelsize=22),
)

tag = length(ARGS) >= 1 ? ARGS[1] : "sq_d04"
out = length(ARGS) >= 2 ? ARGS[2] : tag
csv = "conservation_history_$(tag).csv"

# --- piecewise time model: dt 0.001 -> 0.005 at step 11626 ---
const SWITCH = 11626
const DT1 = 0.001
const DT2 = 0.005
time_at(s) = s <= SWITCH ? s * DT1 : SWITCH * DT1 + (s - SWITCH) * DT2

raw, _ = readdlm(csv, ',', Float64, '\n'; header=true)
keep = Dict{Int,Int}()
for (i, s) in enumerate(raw[:, 1]); keep[Int(s)] = i; end     # last wins
idx = [keep[s] for s in sort(collect(keys(keep)))]
M = raw[idx, :]
step = Int.(M[:, 1])
S    = M[:, 3]
E    = M[:, 4]
m1   = M[:, 5]; m2 = M[:, 6]

t = time_at.(step)
E0 = E[1]; P10 = m1[1]; P20 = m2[1]
E_err = abs.(E .- E0) ./ abs(E0)
P_err = hypot.(m1 .- P10, m2 .- P20) ./ max(hypot(P10, P20), 1e-30)

fig = Figure(; size=(1700, 1100))
ax1 = Axis(fig[1, 1]; ylabel=L"S_h/S_0",
    title=L"\mathrm{Entropy}\ S_h/S_0")
ax2 = Axis(fig[2, 1]; ylabel=L"\mathrm{rel.\ error}",
    title=L"\mathrm{Energy\ conservation\ error}", yscale=log10)
ax3 = Axis(fig[3, 1]; xlabel=L"t", ylabel=L"\mathrm{rel.\ error}",
    title=L"\mathrm{Momentum\ conservation\ error}", yscale=log10)
lines!(ax1, t, S ./ S[1]; color=:red, linewidth=2)
lines!(ax2, t, max.(E_err, 1e-18); color=:blue, linewidth=2)
lines!(ax3, t, max.(P_err, 1e-18); color=:green, linewidth=2)

# merge time axis: share x, single label/ticks at bottom
linkxaxes!(ax1, ax2, ax3)
hidexdecorations!(ax1; grid=false)
hidexdecorations!(ax2; grid=false)
rowgap!(fig.layout, 12)

png = "dashboard_$(out)_3row.png"
save(png, fig)
println("Saved $png  (steps $(step[1])..$(step[end]), $(length(step)) pts)")
