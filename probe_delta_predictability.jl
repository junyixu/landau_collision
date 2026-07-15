# Go/no-go probe: how much of the Euler-predictor gap δ is predictable from
# state features alone? Ridge regression (closed form, no training) on the
# features from warmstart_nn.jl, evaluated per time window.
#
#   δ = (v^{n+1,*} − vⁿ − Δt·v̇ⁿ)/Δt²   from consecutive dump records.
#
# Reported "decades" = log10(rms‖δ‖ / rms‖δ − δ̂‖) — the initial-residual
# decades a perfect-deployment linear model would save. Rule of thumb from
# the plan: linear probe ≥ 0.5 dec with headroom, or ≥ 1 dec ⇒ train the MLP;
# below ⇒ stop (state features carry no usable signal, an MLP won't save it).
#
# Run as:
#   julia --project=. probe_delta_predictability.jl training_dump_<suffix>.bin \
#         [n_windows] [samples_per_pair]
include("warmstart_nn.jl")

length(ARGS) >= 1 || error("usage: probe_delta_predictability.jl <dump.bin> [n_windows] [samples_per_pair]")
dump_file = ARGS[1]
n_windows = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 3
n_per_pair = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 2000
λ_ridge = 1e-6

# Header first: dt is needed while streaming.
N_dump, n_dofs, dt = open(read_dump_header, dump_file)

# Breakpoints are not stored in the dump; read them from the matching fs
# snapshot header (step 0) in cwd, or fall back to the sq_d04 preset mesh.
function load_breakpoints(dump_file)
    m = match(r"training_dump_(.+)\.bin", basename(dump_file))
    if m !== nothing
        snap = "fs_snapshot_$(m.captures[1])_step0000.csv"
        if isfile(snap)
            bp1 = Float64[]; bp2 = Float64[]
            for ln in eachline(snap)
                startswith(ln, "# bp1=") && (bp1 = parse.(Float64, split(ln[8:end], ',')))
                startswith(ln, "# bp2=") && (bp2 = parse.(Float64, split(ln[8:end], ',')))
                isempty(bp1) || isempty(bp2) || return bp1, bp2
            end
        end
    end
    @warn "breakpoints not found next to dump; using sq_d04 preset mesh"
    bp = [-6.0; -5.0; collect(LinRange(-4.0, 4.0, 21)); 5.0; 6.0]
    return bp, copy(bp)
end
const BP1, BP2 = load_breakpoints(dump_file)

# Streaming pass: collect (features, δ) samples with one record lookbehind.
# Weights are uniform (w = 1/N) in these runs — needed for the temperature
# moments inside the feature builder.
X_all = Matrix{Float32}[]
Y_all = Matrix{Float32}[]
steps = Int[]

prev = nothing
rng = Random.Xoshiro(7)
foreach_dump_record(dump_file) do step, v, dot_v, G, L
    global prev
    if prev !== nothing && step == prev.step + 1
        N = size(v, 1)
        w = fill(1.0 / N, N)
        idx = rand(rng, 1:N, min(n_per_pair, N))
        vp  = Float64.(prev.v);  dvp = Float64.(prev.dot_v)
        δ = Matrix{Float32}(undef, 2, length(idx))
        @inbounds for (j, α) in enumerate(idx)
            δ[1, j] = (Float64(v[α, 1]) - vp[α, 1] - dt * dvp[α, 1]) / dt^2
            δ[2, j] = (Float64(v[α, 2]) - vp[α, 2] - dt * dvp[α, 2]) / dt^2
        end
        Xfull = build_features(vp, dvp, Float64.(prev.G), w, BP1, BP2, dt)
        push!(X_all, Xfull[:, idx])
        push!(Y_all, δ)
        push!(steps, prev.step)
    end
    prev = (; step, v, dot_v, G)
end

isempty(X_all) && error("no consecutive record pairs in $dump_file")
println("dump: N=$N_dump  n_dofs=$n_dofs  dt=$dt  pairs=$(length(steps))  " *
        "samples=$(sum(size(x, 2) for x in X_all))")

# Ridge fit + evaluation on a sample set (matrices n_feat × S, 2 × S).
function ridge_decades(X::Matrix{Float32}, Y::Matrix{Float32})
    S = size(X, 2)
    Xa = vcat(Float64.(X), ones(1, S))          # bias row
    A  = Xa * Xa' + λ_ridge * S * I
    B  = Xa * Float64.(Y)'
    β  = A \ B                                   # (n_feat+1) × 2
    R  = Float64.(Y) - (Xa' * β)'
    rms_y = sqrt(sum(abs2, Y) / S)
    rms_r = sqrt(sum(abs2, R) / S)
    return log10(rms_y / rms_r), rms_y, rms_r
end

# Global fit
Xg = hcat(X_all...); Yg = hcat(Y_all...)
dec, rms_y, rms_r = ridge_decades(Xg, Yg)
println("\nGLOBAL   rms‖δ‖=$(round(rms_y; sigdigits=3))  " *
        "residual=$(round(rms_r; sigdigits=3))  decades=$(round(dec; digits=2))")

# Per-window fits (relaxation phases behave differently)
edges = round.(Int, range(1, length(X_all) + 1; length=n_windows + 1))
for wdw in 1:n_windows
    lo, hi = edges[wdw], edges[wdw+1] - 1
    Xw = hcat(X_all[lo:hi]...); Yw = hcat(Y_all[lo:hi]...)
    d, ry, rr = ridge_decades(Xw, Yw)
    println("window $wdw  steps $(steps[lo])–$(steps[hi])  " *
            "rms‖δ‖=$(round(ry; sigdigits=3))  residual=$(round(rr; sigdigits=3))  " *
            "decades=$(round(d; digits=2))")
end

println("\ngo/no-go: ≥0.5 dec (linear, headroom for MLP) or ≥1 dec ⇒ train; " *
        "below ⇒ stop.")
