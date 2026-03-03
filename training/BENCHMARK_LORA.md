# LoRA Fine-Tuning Benchmark — M4 Pro (Mac Mini)

## Hardware
- **Machine**: Mac Mini (M4 Pro, 14-core CPU / 20-core GPU / 16-core Neural Engine)
- **RAM**: 48GB unified
- **macOS**: 15.x (Sequoia)
- **Model**: stories110M (12-layer, 768-dim, 12-head, 32k vocab)

## What This Is

CPU-side LoRA (Low-Rank Adaptation) bolted onto the existing ANE training pipeline. Freezes all pretrained weights and trains only small rank-4 matrices on Wq and Wv projections. The LoRA matrices are merged into frozen weights on CPU before each ANE kernel compile: `W_eff = W_frozen + (alpha/rank) * B @ A`.

**No changes to MIL generators, ANE compile/eval, or IOSurface pipeline.** Two new files (`lora.h`, `train_lora.m`), one Makefile line.

## Parameters

| | Full Training | LoRA |
|---|---|---|
| Trainable params | 109,529,856 (438 MB) | 147,456 (576 KB) |
| Reduction | — | **744x fewer params** |
| Optimizer memory | ~1.7 GB (Adam m+v) | ~2.3 MB (Adam m+v) |
| Checkpoint size | 438 MB | 1.7 MB |

- LoRA rank: 4
- Alpha: 4.0
- Targets: Wq, Wv per layer (standard LoRA)
- Optimizer: Adam (lr=1e-4, β1=0.9, β2=0.999)
- Gradient accumulation: 10 steps per batch
- Sequence length: 256

## Throughput (10-step batch, M4 Pro)

| Metric | train_large | train_lora | Delta |
|---|---|---|---|
| Compile time | 3310 ms | 3301 ms | ~same |
| Train time (10 steps) | 747 ms | 737 ms | ~same |
| ms/step | 74.7 | 73.7 | ~same |
| Compile overhead | 79.4% | 81.3% | ~same |

Per-step training time is effectively identical — the LoRA merge (`cblas_sgemm` for 768×4 × 4×768, 12 layers × 2 targets) adds <1ms total, negligible against ~70ms ANE eval per step.

**Compile overhead dominates.** Both approaches spend ~80% of wall time recompiling ANE kernels. This is the nature of the current weight-baked-into-kernel architecture. The proposed weights-as-tensors approach (issue #18) would eliminate this for LoRA entirely, since merged weights change every batch.

## Convergence (1000 steps / 100 batches)

Fine-tuning stories110M on the same TinyStories distribution it was pretrained on. This is deliberately a conservative test — LoRA should show minimal improvement since the model already fits this distribution.

```
=== Loss Trend (1000 steps) ===
First 10 batches avg: 3.7481    Last 10: 3.7142    Δ: -0.034
First 25 batches avg: 3.7478    Last 25: 3.7125    Δ: -0.035
Overall avg: 3.7239
Min batch loss: 3.4982 (batch 46)
Max batch loss: 3.9634 (batch 74)
Linear regression: loss = -0.0004/batch (R² = 0.02)
```

R² is low because batch losses have high variance (random 256-token windows from a 20M-token shard). The weak downward trend is expected — the model already fits this distribution, so LoRA has little to learn. A proper LoRA use case (adapting to a new domain) would show much larger loss reduction.

## Gradient Flow Verification

The strongest proof that LoRA is working correctly:

```
=== LoRA Parameter Norms (1000 steps) ===
|A|: 13.7915 → 14.0839  (Δ = +0.292, +2.1%)
|B|:  0.0262 →  1.2913  (49.3x growth from near-zero init)

|B| quartiles: Q1=0.314  Q2=0.618  Q3=0.957  Q4=1.291
```

- **B starts at zero** (by design — ensures model starts at pretrained baseline)
- **B grows 49x** over 1000 steps: gradients are flowing through the chain rule correctly
- **B growth is linear** (quartiles roughly evenly spaced): consistent gradient magnitude throughout training
- **A grows slowly** (+2.1%): expected because dA ∝ B^T·dW, and B starts near-zero so A gets tiny initial gradients
- Compile time grows ~6% over 1000 steps as merged weights diverge from initial values

## Identity Property

With fresh initialization (B=0), the first forward pass produces **identical loss** to running the pretrained model through train_large:

```
train_lora step 0: loss = 3.7212
train_large step 0: loss = 3.7212
```

This confirms `W_eff = W_frozen + 0 = W_frozen` — the merge is mathematically correct.

## Checkpoint Round-Trip

Save at step 20, resume with `--steps 40`:

```
Phase 1 (steps 0-19):  |A|=13.7918  |B|=0.0414
Phase 2 (steps 20-29): |A|=13.7922  |B|=0.0552  (smooth continuation)
Phase 2 (steps 30-39): |A|=13.7926  |B|=0.0679  (continued growth)
```

Parameter norms continue smoothly across checkpoint boundaries. Adam optimizer state (m, v, t) is preserved.

## Caveats

1. **This is a CPU-merge approach.** LoRA matrices live on CPU and are merged into weights before ANE kernel compilation. This means recompile every batch, same as full training.
2. **Compile overhead is the bottleneck.** ~80% of wall time is ANE kernel compilation. With weights-as-tensors (#18), LoRA could skip recompilation entirely.
3. **Same-distribution fine-tuning shows minimal loss improvement.** A proper LoRA use case would be adapting to a new domain (medical text, code, dialogue style) where you'd see much larger loss reduction.
4. **Rank 4 is minimal.** Higher ranks (8, 16) would learn faster but use more memory. Rank 4 is conservative for a 768-dim model.
5. **Only Wq and Wv are adapted.** Standard LoRA targets. Adding Wk, Wo, or FFN projections would increase capacity.

## Files

- `lora.h` — Data structures, merge, gradient extraction, checkpoint I/O (~164 lines)
- `train_lora.m` — Training loop, identical forward/backward to train_large (~507 lines)
- Makefile — +1 target

No modifications to existing files.
