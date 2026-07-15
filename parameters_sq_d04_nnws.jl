# sq_d04_nnws — sq_d04 square mesh with per-step training dump for the NN
# warm-start pipeline. Same physics/mesh/solver as parameters_sq_d04.jl
# (square mesh bp1=bp2, inner Δ=0.4, anisotropic IC σ1=4/3 σ2=0.5 → isotropic,
# Gonzalez + ½·log f²), extended to 5000 steps. dump_training streams
# (v, v̇, G, L_vec) per step to training_dump_sq_d04_nnws.bin (~1 MB/step ⇒
# ~4.7 GB total); δ labels are built offline from consecutive records.
PARAMS = SimParameters(
    bp1 = [-6.0; -5.0; LinRange(-4.0, 4.0, 21); 5.0; 6.0],   # Δinner=0.4
    bp2 = [-6.0; -5.0; LinRange(-4.0, 4.0, 21); 5.0; 6.0],   # = bp1
    P_DEG=2, K_REG=1, N_QUAD=6,
    N_PARTICLES=40_000,
    σ1=4/3, σ2=0.5,
    DT=0.001, N_STEPS=5000,
    use_anderson=true,
    use_gonzalez=true,
    use_logsq=true,
    damping=0.7, m_anderson=8,
    tol=1e-12, max_iter=2000,
    abs_floor=1e-10,
    stag_window=30, stag_rel_tol=0.1,
    damp_decay_start=200, damp_decay_factor=0.5,
    dump_training=true,
    suffix="sq_d04_nnws",
    seed=42,
)
