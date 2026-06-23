# sq_d04 — square mesh (bp1 = bp2), inner Δ = 0.4, fresh anisotropic run.
# Coarser-resolution counterpart of the dense square (sq_dense_from2000, inner
# Δ = 0.2): same symmetric square layout [-6,-5, inner[-4,4], 5,6] with no giant
# coarse outer cell, but the inner region at Δ = 0.4 (LinRange(-4,4,21)) instead
# of 0.2. Anisotropic IC (σ1=4/3, σ2=0.5) → isotropic, Gonzalez + ½·log f²,
# fresh 0 → N. Use to probe how the residual negative ring scales with inner
# square-mesh resolution.
PARAMS = SimParameters(
    bp1 = [-6.0; -5.0; LinRange(-4.0, 4.0, 21); 5.0; 6.0],   # Δinner=0.4
    bp2 = [-6.0; -5.0; LinRange(-4.0, 4.0, 21); 5.0; 6.0],   # = bp1
    P_DEG=2, K_REG=1, N_QUAD=6,
    N_PARTICLES=40_000,
    σ1=4/3, σ2=0.5,
    DT=0.001, N_STEPS=2000,
    use_anderson=true,
    use_gonzalez=true,
    use_logsq=true,
    damping=0.7, m_anderson=8,
    tol=1e-12, max_iter=2000,
    abs_floor=1e-10,
    stag_window=30, stag_rel_tol=0.1,
    damp_decay_start=200, damp_decay_factor=0.5,
    suffix="sq_d04",
    seed=42,
)
