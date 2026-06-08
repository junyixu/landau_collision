# Parameters.jl — 1D Lenard-Bernstein configuration.
#
# Workflow:
#   1. Preset files (parameters_LB_*.jl) build a `SimParameters` bound to
#      `PARAMS`. Each preset directly supplies the breakpoint vector `bp`.
#   2. main_LB.jl picks ARGS[1] as preset, --key=val tokens override scalars.
#   3. Vector fields (`bp`) not CLI-overridable — edit the preset.
#
# Initial-condition selector:
#   ic_type = "shifted" : v ~ N(ic_mu, ic_sigma)               (paper §5.2)
#   ic_type = "bimax"   : 50/50 mixture N(±ic_sep, ic_sigma)   (paper §5.3)
#   ic_type = "uniform" : v ~ U(-ic_L, ic_L)                   (paper §5.4)

Base.@kwdef struct SimParameters
    # Velocity-space breakpoints (1D, possibly non-uniform).
    bp::Vector{Float64} = collect(LinRange(-10.0, 10.0, 42))

    # B-spline space
    P_DEG::Int = 3
    K_REG::Int = 2
    N_QUAD::Int = 6

    # Particles
    N_PARTICLES::Int = 1000

    # IC selector + parameters (only the relevant ones for the chosen type are used)
    ic_type::String = "bimax"
    ic_mu::Float64    = 2.0       # shifted: mean
    ic_sigma::Float64 = 1.0       # shifted / bimax: std
    ic_sep::Float64   = 2.0       # bimax:   centers at ±ic_sep
    ic_L::Float64     = 2.0       # uniform: half-width

    # Collision frequency
    nu::Float64 = 1.0

    # Time integration (implicit midpoint)
    DT::Float64 = 8e-4
    N_STEPS::Int = 6250            # default bi-Max final time = 5.0

    # Implicit-solver knobs
    use_anderson::Bool = true
    damping::Float64   = 0.7
    m_anderson::Int    = 8
    tol::Float64       = 1e-12
    max_iter::Int      = 2000
    abs_floor::Float64       = 1e-10
    stag_window::Int         = 50
    stag_rel_tol::Float64    = 0.01
    damp_decay_start::Int    = 200
    damp_decay_factor::Float64 = 0.5

    # Snapshot cadence (every snap_every steps + final)
    snap_every::Int = 250

    # Run identifier
    suffix::String = "LB_bimax"

    # RNG seed
    seed::Int = 42
end

function _parse_override_token(tok::AbstractString)
    startswith(tok, "--") || return nothing
    eq = findfirst('=', tok)
    eq === nothing && error("CLI override $tok must use --key=value form")
    key = Symbol(tok[3:eq-1])
    val_str = tok[eq+1:end]
    return key, val_str
end

function _coerce(::Type{T}, s::AbstractString) where {T}
    if T === Bool
        s in ("true", "1")  && return true
        s in ("false", "0") && return false
        error("Cannot parse $s as Bool")
    elseif T <: AbstractString
        return String(s)
    elseif T <: AbstractVector
        error("Vector parameter `bp` is not CLI-overridable; edit the preset file")
    else
        return parse(T, s)
    end
end

function parse_overrides(p::SimParameters, args)
    isempty(args) && return p
    fields = fieldnames(SimParameters)
    field_type = Dict(f => fieldtype(SimParameters, f) for f in fields)
    overrides = Dict{Symbol, Any}()
    for tok in args
        parsed = _parse_override_token(tok)
        parsed === nothing && continue
        key, val_str = parsed
        haskey(field_type, key) || error("Unknown parameter: $key")
        overrides[key] = _coerce(field_type[key], val_str)
    end
    return SimParameters(; (f => get(overrides, f, getfield(p, f)) for f in fields)...)
end

function print_summary(p::SimParameters)
    n_cells = length(p.bp) - 1
    dv_min, dv_max = extrema(diff(p.bp))
    println("==== SimParameters (LB 1D) ====")
    println("v ∈ [$(p.bp[1]), $(p.bp[end])]  cells=$n_cells  Δv ∈ [$dv_min, $dv_max]")
    println("P_DEG=$(p.P_DEG)  K_REG=$(p.K_REG)  N_QUAD=$(p.N_QUAD)")
    println("N_PARTICLES=$(p.N_PARTICLES)  seed=$(p.seed)")
    println("ic_type=$(p.ic_type)  μ=$(p.ic_mu)  σ=$(p.ic_sigma)  sep=$(p.ic_sep)  L=$(p.ic_L)")
    println("ν=$(p.nu)  DT=$(p.DT)  N_STEPS=$(p.N_STEPS)  T_final=$(p.DT * p.N_STEPS)")
    println("solver=$(p.use_anderson ? "Anderson(m=$(p.m_anderson))" : "Picard")" *
            "  damping=$(p.damping)  tol=$(p.tol)  max_iter=$(p.max_iter)")
    println("suffix=$(p.suffix)")
    println("===============================")
end

# Sample initial particle velocities according to `ic_type`. Returns Vector{Float64}.
function sample_initial_velocities(p::SimParameters, rng=Random.default_rng())
    if p.ic_type == "shifted"
        return p.ic_mu .+ p.ic_sigma .* randn(rng, p.N_PARTICLES)
    elseif p.ic_type == "bimax"
        v = Vector{Float64}(undef, p.N_PARTICLES)
        for i in 1:p.N_PARTICLES
            sign_i = rand(rng) < 0.5 ? -1.0 : 1.0
            v[i] = sign_i * p.ic_sep + p.ic_sigma * randn(rng)
        end
        return v
    elseif p.ic_type == "uniform"
        return p.ic_L .* (2 .* rand(rng, p.N_PARTICLES) .- 1.0)
    else
        error("Unknown ic_type: $(p.ic_type)")
    end
end
