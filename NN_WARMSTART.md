# NN warm start for the implicit Landau solve

分支 `iso-plainmid-probe-rclone`。目标：在 explicit Euler 预测子之上叠加一个
逐粒子 MLP 修正，给 Anderson 不动点求解一个更准的初值：

$$v^{(0)} = v^n + \Delta t\,\dot v^n + \Delta t^2\,\hat\delta_\theta$$

求解器与收敛解完全不动（守恒性质零风险），初值更准 = 迭代更少。当前
sq_d04 冒烟测试早期段平均 ~45 iter/步；若砍半，墙钟近似砍半。

设计由实测锁定（bimodal 分支 25k 步数据）：

- Euler 预测子**优于** previous-step 和外推（实测外推初值差 1.6–2.7×；
  轨迹时间上只有 C¹——粒子穿越样条 knot 时 ∇²φ 跳变，τ_eff ≈ Δt），
  所以保留 Euler 当基座，NN 只学余差 δ，**禁用历史特征**。
- 标签 δ = (v^{n+1,*} − vⁿ − Δt·v̇ⁿ)/Δt²，v^{n+1,*} 即下一步收敛解。
- 特征只用当前状态（v, v̇, G, 单元内相对坐标 ξ, knot 穿越预警, 单元尺寸,
  全局温度 T₁/T₂, |v|，共 15 维）——穿越预警是外推原理上拿不到的信息。

## 改动清单

| 文件 | 改动 |
|---|---|
| `warmstart_nn.jl` **(新)** | 特征构建器（单一来源，probe/训练/推理共用）；手写 tanh MLP（15→h→h→h→2）+ 反向传播（纯 stdlib，热循环零新依赖）；`WarmstartModel` 权重+归一化统计的存取（Serialization）；训练 dump 二进制格式读写（定长记录、逐步 flush、crash-safe） |
| `Parameters.jl` | 新字段：`warmstart::Symbol=:euler`（`:euler`/`:nn`）、`nn_weights::String`、`dump_training::Bool=false`；`_coerce` 支持 Symbol（CLI 可覆写） |
| `main_Gonzalez.jl` | `step_anderson!` 返回值加第 4 项 `nrm_r0`（初值残差，三个 return 点）；conservation CSV 尾部加 `r0` 列（旧列序不变，老脚本不破）；预测子块后写 dump 记录（在 NN 修正**之前**，保证标签纯净）；`:nn` 分支调用 `nn_warmstart_correct!`（逐粒子 clamp ≤ 3×mean‖Δt·v̇‖ 兜底）；dump io 开/关 |
| `parameters_sq_d04_nnws.jl` **(新)** | sq_d04 同参数 + `N_STEPS=5000` + `dump_training=true`，suffix=`sq_d04_nnws`（~1 MB/步 ⇒ 全程 ~4.7 GB dump） |
| `probe_delta_predictability.jl` **(新)** | go/no-go 探针：流式读 dump，ridge 回归拟合 δ，按时间窗报告可压掉的数量级 |
| `train_warmstart.jl` **(新)** | 离线训练：流式采样、按时间块切 train/val（防相邻步泄漏）、手写 Adam、按 val 存最优模型 |

NN 无状态 ⇒ checkpoint 格式未动，resume 天然兼容。

## 测试状态（诚实记录）

- ✅ 单元冒烟：MLP 反向传播过有限差分校验；dump 读写往返；特征有限性。
- ✅ 集成冒烟（12 步）：`r0` 列在写、dump ~1 MB/步、probe 端到端跑通。
- ⚠️ `train_warmstart.jl` 与 `--warmstart=nn` 部署路径**尚未端到端跑过**
  （冒烟被中断）；首次真跑前先用小参数验证（见下 Step 3 的 smoke 变体）。

## 使用流程

### Step 1 — 数据采集跑（>5 min ⇒ systemd）

```bash
systemd-run --user \
    --unit=junyi-task-julia-Gonzalez-diagnostics-nnwsdump \
    --working-directory="$PWD" \
    --setenv=JULIA_NUM_THREADS=8 \
    julia --project=. main_Gonzalez.jl parameters_sq_d04_nnws.jl
# 跟踪:
journalctl --user -u junyi-task-julia-Gonzalez-diagnostics-nnwsdump.service -f --no-pager
```

产出：`training_dump_sq_d04_nnws.bin`（可中途 kill，记录逐步 flush）+
带 `r0` 列的 `conservation_history_sq_d04_nnws.csv`（这就是 baseline 初值
质量曲线）。

### Step 2 — go/no-go 探针（几分钟，前台）

```bash
julia --project=. probe_delta_predictability.jl training_dump_sq_d04_nnws.bin
# 可选: [n_windows=3] [samples_per_pair=2000]
```

判据：线性探针 ≥0.5 dec（MLP 有加成空间）或 ≥1 dec ⇒ 训练；
低于 ⇒ 停（状态特征无信号，MLP 救不了），转投 GPU 碰撞和。

### Step 3 — 训练（systemd）

```bash
# 先小参数 smoke（~1 min，前台）:
julia --project=. train_warmstart.jl training_dump_sq_d04_nnws.bin \
    nn_smoke.jls --epochs=2 --hidden=32 --max_samples=100000

# 真训练:
systemd-run --user \
    --unit=junyi-task-julia-Gonzalez-diagnostics-nnwstrain \
    --working-directory="$PWD" \
    --setenv=JULIA_NUM_THREADS=8 \
    julia --project=. train_warmstart.jl training_dump_sq_d04_nnws.bin \
        nn_warmstart_sq_d04.jls --epochs=30 --hidden=128
```

关注输出的 `val_decades`——部署后初值能压掉的数量级预估。

### Step 4 — 部署 + A/B

```bash
# NN warm start 跑:
systemd-run --user \
    --unit=junyi-task-julia-Gonzalez-diagnostics-nnwsab \
    --working-directory="$PWD" \
    --setenv=JULIA_NUM_THREADS=8 \
    julia --project=. main_Gonzalez.jl parameters_sq_d04_nnws.jl \
        --dump_training=false --warmstart=nn \
        --nn_weights=nn_warmstart_sq_d04.jls --suffix=sq_d04_nn
```

对比两条 CSV：`r0` 列（初值残差，直接度量）、`iter` 列直方图、墙钟。
守恒列应在求解器噪声内一致（收敛根未变）。

### 快速冒烟（改代码后回归用）

```bash
RCLONE_UPLOAD=0 JULIA_NUM_THREADS=8 julia --project=. \
    main_Gonzalez.jl parameters_sq_d04_nnws.jl --N_STEPS=12 --suffix=nnws_smoke
```

## Dump 二进制格式

头部：`"NNWSDMP1"` + N::Int64 + n_dofs::Int64 + dt::Float64。
记录：step::Int64 + v,v̇,G（各 2N Float32，列主序）+ L_vec（n_dofs Float64）。
定长记录可 seek；`foreach_dump_record` 流式读，不整载内存。
breakpoints 不入 dump——probe/训练从同目录 `fs_snapshot_<suffix>_step0000.csv`
头部读取（缺失则回退 sq_d04 预设网格并告警）。
