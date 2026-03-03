# ANE Training — LoRA Fine-Tuning on Apple Neural Engine, No Recompilation

> **Fork of [maderix/ANE](https://github.com/maderix/ANE)** — the original project that proved backpropagation on the Apple Neural Engine is possible. All credit for the foundational reverse engineering, the private API discovery, the MIL code generation, and the single-layer training prototype goes to [@maderix](https://github.com/maderix). Read his excellent writeups: [Part 1](https://maderix.substack.com/p/inside-the-m4-apple-neural-engine), [Part 2](https://maderix.substack.com/p/inside-the-m4-apple-neural-engine-615).
>
> This fork extends the original with **LoRA fine-tuning** and a **dynamic weight pipeline that eliminates ANE recompilation entirely**.

---

## What's new in this fork

The original repo proved something remarkable: you can train neural networks on Apple's Neural Engine by reverse-engineering private APIs. But it had a fundamental bottleneck — **weights are baked into ANE programs at compile time**. Every time weights change, you recompile. The ANE compiler leaks memory and crashes after ~119 compilations. The workaround was `exec()` — kill the process, restart, reload from checkpoint, recompile, keep training. Every 10 steps.

**This fork solves that problem.** We found a way to pass weights at runtime without recompilation, then built LoRA fine-tuning on top of it.

### The headline numbers

| What | Original repo | This fork |
|------|--------------|-----------|
| Model | Single transformer layer | **12-layer, 109M-param Stories110M** |
| Training data | Synthetic/random | **Real tokenized TinyStories (20M tokens)** |
| Recompilation | Every 10 steps + `exec()` restart | **Never. Compile once, train forever.** |
| LoRA fine-tuning | No | **Yes — 147K trainable params, 2.9 MB checkpoints** |
| Compile limit (~119) | Hit it, must `exec()` restart | **Eliminated** |
| Parameter efficiency | Full 109M params | **744x reduction with LoRA** |
| Checkpoint size | 438 MB (full weights) | **2.9 MB (LoRA adapters only)** |
| Optimizer memory | ~1.7 GB | **~2.3 MB** |
| Python bridge | No | **Yes — libane_bridge.dylib for ctypes** |

### Why this matters

The original project's `exec()` restart was a clever hack, but it meant ANE training was a research demo — you couldn't just let it run. Compile overhead dominated wall time (75-82%), and every restart burned seconds reloading state.

**With the dynamic weight pipeline, compile cost drops to a one-time 333ms.** The ANE just trains. No restarts, no compile limits, no process cycling. Combined with LoRA, you get parameter-efficient fine-tuning that produces tiny 2.9 MB adapter checkpoints — small enough to swap at inference time, share over a network, or store per-user.

This is the difference between "we proved it's possible" and "you can actually use this."

---

## How we eliminated recompilation

We tried five approaches. Four failed. The failures are documented here because they reveal real ANE hardware constraints that anyone working with these APIs needs to know.

**Attempt 1: Weight file swap.** Overwrite the weight blob on disk, unload and reload the model. Result: ANE ignores the file change — weights are baked at compile time and the reload serves the cached program. Confirmed on M4 and M5. **Dead end.**

**Attempt 2: `weightsBuffer` IOSurface.** The `_ANERequest` API accepts a `weightsBuffer` parameter. We filled it with new weights and passed it at eval time. Result: output unchanged. The parameter serves some other internal purpose. **Dead end.**

**Attempt 3: Multiple dynamic IOSurfaces.** Pass weights as separate IOSurface inputs alongside activations. Result: ANE rejects it with `status=0x1d` — the hardware only accepts a single dynamic input tensor. Code preserved in `stories_flex.h`. **Dead end.**

**Attempt 4: Partial recompile.** Only recompile kernels whose weights changed (LoRA targets Wq/Wv — 24 of 72 kernels). Result: 2x speedup, but still recompiles, still hits the ~119 limit, still needs `exec()` every 70 steps. **Partial win, not the answer.**

**Attempt 5: Pack weights into the spatial dimension.** The ANE takes input as `[1, channels, 1, spatial]`. We made the spatial dimension wider and packed weight matrices into the extra columns:

```
[1, DIM, 1, SEQ + weight_cols]
         ^^^^          ^^^^^^^^^^^^^
     activations       weights stuffed here
         sp[0:256]     sp[256:256+768]
```

The MIL graph uses `slice_by_size` to carve out the weight region, then `matmul` to multiply. The ANE has no idea the extra columns are weights — it just sees a wider input tensor.

**Result: 9 kernels compiled once at startup. Zero recompilation. No compile limit. No `exec()` restart. Train forever.**

This is the key insight. Everything else — LoRA, the performance gains, the tiny checkpoints — builds on top of it.

## LoRA on the Neural Engine

With recompilation eliminated, LoRA becomes practical on the ANE.

**How it works:** Freeze all 109M pretrained weights. Train only small rank-4 matrices (A and B) on the Wq and Wv projections — 147,456 parameters total. Before each forward pass, merge on CPU: `W_eff = W_frozen + (alpha/rank) * B @ A`. This adds <1ms. The merged weights go into the IOSurface spatial dimension, the ANE runs the same 9 kernels, gradients flow back, and only the tiny LoRA matrices get updated.

**What we skip:** 5 of 7 weight gradient GEMMs per layer. The backward pass still runs all ANE kernels (you need activation gradients for backprop), but the expensive CPU-side `cblas_sgemm` calls for frozen weight gradients (dW1, dW2, dW3, dWo, dWk) are eliminated entirely.

**Verified correct:**
- Identity property: with fresh LoRA (B=0), loss matches pretrained model exactly at step 0
- Gradient flow: |B| grows 49x over 1000 steps from near-zero init — consistent, linear growth
- Checkpoint round-trip: parameter norms continue smoothly across save/resume boundaries

| | Full Training | LoRA |
|---|---|---|
| Trainable params | 109,529,856 (438 MB) | 147,456 (576 KB) |
| Reduction | — | **744x fewer** |
| Optimizer memory | ~1.7 GB | **~2.3 MB** |
| Checkpoint | 438 MB | **2.9 MB** |
| dW GEMMs/layer | 7 | **2** |
| Recompilation | Never (dynamic) | **Never** |

---

## Quick start

macOS 15+, Apple Silicon (tested on M4, M4 Pro, M5). No external dependencies.

```bash
git clone https://github.com/solatticus/ANE.git && cd ANE/training
bash download_data.sh                     # pretokenized TinyStories (~41 MB)

# Dynamic LoRA — the recommended path
cd training_dynamic && make train_lora
./train_lora --steps 1000 --lr 1e-4       # fine-tune on your Neural Engine

# Dynamic full training (all 109M params)
make train && ./train --scratch --steps 1000

# Dashboard (live loss curves, power, memory)
pip install blessed psutil numpy
sudo python3 ../dashboard.py --dynamic
```

**CLI flags:**
- `--steps N` — training steps (default 10000)
- `--lr F` — learning rate (default 3e-4)
- `--model PATH` — pretrained weights file
- `--ckpt PATH` — checkpoint file
- `--resume` / `--scratch` — resume from checkpoint or start fresh

The original static pipelines are also preserved:
```bash
# Original static baseline (for comparison)
make train_large && ./train_large --steps 100

# Static + ANE extras (PR#19 — classifier/softmax/rmsnorm on ANE)
make train_large_ane && ./train_large_ane --steps 100
```

## Python bridge

`bridge/libane_bridge.dylib` — wraps all ANE private APIs into C functions callable from Python via ctypes:

```c
int ane_bridge_init(void);
ANEKernelHandle *ane_bridge_compile(mil_text, mil_len, weight_data, weight_len, ...);
ANEKernelHandle *ane_bridge_compile_multi_weights(mil_text, mil_len, names, datas, lens, n, ...);
bool ane_bridge_eval(ANEKernelHandle *kernel);
void ane_bridge_write_input(kernel, idx, data, bytes);
void ane_bridge_read_output(kernel, idx, data, bytes);
void ane_bridge_free(ANEKernelHandle *kernel);
uint8_t *ane_bridge_build_weight_blob(src, rows, cols, out_len);
```

Handles IOSurface lifecycle, ARC memory management, compile retry logic (100ms backoff on load failure), and compile count tracking for `exec()` budgeting.

---

## Performance

### This fork vs original — 20 steps on M4 Pro

| | Original static | This fork: Dynamic | This fork: Dynamic LoRA |
|---|---|---|---|
| **Wall time** | **10.1s** | **~2.6s** | **~2.8s** |
| Compile time | 7.6s (75%) | 0.4s (15%) | 0.3s (11%) |
| Training time | 2.1s | 2.2s | 2.5s |
| ms/step | 106.7 | 111 | 125.6 |
| Recompile | Every 10 steps | **Never** | **Never** |
| exec() restart | Yes | **No** | **No** |
| Speedup (wall) | 1x | **3.9x** | **3.6x** |

The per-step training time is similar — the ANE kernels do the same work. The difference is that the original spends 75% of wall time recompiling, and that compounds: at 100 steps, 1000 steps, 10000 steps, the dynamic pipeline just keeps pulling ahead.

### LoRA: static vs dynamic pipeline

| | Static LoRA | Dynamic LoRA |
|---|---|---|
| Compile | 3,400ms per batch | **333ms once** |
| ms/step | ~400 | **125.6** |
| exec() restart | Every 10 steps | **Never** |
| Speedup | 1x | **3.2x** |

### Single-layer optimization history (from original repo)

| Optimization | ms/step | ANE util |
|---|---|---|
| Baseline (vDSP transpose) | 33.5 | 3.1% |
| Channel-first layout | 20.3 | 5.2% |
| vDSP vectorized RMSNorm | 14.2 | 7.4% |
| GCD async cblas overlap | 11.4 | 9.2% |
| ANE RMSNorm fusion | 11.4 | 9.2% |
| Wo^T fusion (7->6 kernels) | 11.4 | 9.2% |
| Deferred cblas wait | **9.3** | **11.2%** |

---

## ANE hardware findings

Discoveries from probing M4, M4 Pro, and M5 that informed the design:

- **Weights are immutable after compile.** File swap, unload/reload, `weightsBuffer` IOSurface — none of them change the output. Weights are baked. This is why the dynamic spatial packing approach was necessary.
- **Single dynamic input only.** ANE rejects multiple IOSurface inputs with `status=0x1d`. You get one input tensor. This is why we pack weights into the spatial dimension of that one tensor.
- **~119 compile limit per process.** The ANE compiler leaks resources. The original repo works around this with `exec()` restart. The dynamic pipeline eliminates it by compiling once.
- **SDPA ignores causal mask.** ANE hardware ignores `attn_mask` in SDPA operations. Must decompose into Q@K^T, mask+softmax, scores@V as separate ops.
- **QoS has no effect.** All values 0-63 produce identical latency. ANE runs at fixed frequency.
- **Chaining API exists.** `_ANEChainingRequest` with loopback support — could enable multi-layer pipelining without CPU round-trips. Unexplored.
- **67 private classes** discovered. Many unexplored (`_ANEDeviceController`, `_ANESharedEvents`, `_ANEProgramForEvaluation`).

## Architecture details

### 6 ANE kernels per training step (per layer)

| Kernel | Function |
|--------|----------|
| `kFwdAttn` | RMSNorm + QKV projection + SDPA + output projection |
| `kFwdFFN` | RMSNorm + SwiGLU FFN (W1, W3, SiLU, W2) |
| `kFFNBwd` | FFN backward (W2^T + SiLU_bwd + W1^T + W3^T) |
| `kSdpaBwd1` | Wo^T + SDPA backward part 1 (dV, probs, dp) |
| `kSdpaBwd2` | SDPA backward part 2 (softmax grad, dQ, dK) |
| `kQKVb` | QKV backward (Wq^T + Wk^T + Wv^T -> dx) |

In the dynamic pipeline, these 6 kernel *types* become 9 compiled kernels (some split for the spatial packing layout) shared across all 12 layers. Weights are swapped via IOSurface writes between layers — same kernels, different data.

### Key optimizations
- **Channel-first layout** — `[1,C,1,S]` everywhere, matches ANE IOSurface format, zero transpose
- **Dynamic weight packing** — weights as extra spatial columns in single IOSurface
- **vDSP vectorized math** — RMSNorm 10x faster, cross-entropy 8x faster
- **NEON fp16<->fp32** — ARM intrinsics for IOSurface data transfer
- **GCD async cblas** — dW sgemms overlap with ANE eval on background queue
- **Deferred cblas wait** — pushed into next step's forward pass
- **ANE RMSNorm fusion** — folded into forward kernels as MIL ops
- **Wo^T fusion** — output projection backward merged into SDPA backward
- **Forward taps** — Q, K, V, scores exposed via concat, no CPU recompute
- **Vocab compaction** — 32K -> 9.2K active tokens, 3.5x less classifier work

## File structure

```
├── api_exploration.m           # ANE API discovery (from original)
├── inmem_basic.m               # In-memory MIL compilation PoC (from original)
├── inmem_bench.m               # ANE dispatch latency benchmarks (from original)
├── inmem_peak.m                # Peak TFLOPS measurement (from original)
├── sram_bench.m                # SRAM bandwidth probing (from original)
├── sram_probe.m                # SRAM size/layout exploration (from original)
├── bridge/                     # NEW — Python bridge
│   ├── ane_bridge.h            # C-callable ANE bridge API
│   ├── ane_bridge.m            # Bridge implementation (ObjC -> C)
│   └── libane_bridge.dylib     # Compiled shared library
└── training/
    ├── ane_runtime.h           # ANE private API wrapper (from original)
    ├── ane_mil_gen.h           # MIL generation helpers (from original)
    ├── model.h                 # Weight init and blob builders (from original)
    ├── forward.h               # Forward pass MIL generators (from original)
    ├── backward.h              # Backward pass MIL generators (from original)
    ├── stories_config.h        # Stories110M config/structs (from original)
    ├── stories_io.h            # IOSurface I/O, NEON conversion (from original)
    ├── stories_mil.h           # Static MIL generators (from original)
    ├── stories_cpu_ops.h       # vDSP RMSNorm, cross-entropy, Adam (from original)
    ├── train_large.m           # Static baseline trainer (from original)
    ├── train_large_ane.m       # Static + ANE extras (from original, PR#19)
    ├── ane_classifier.h        # ANE classifier/softmax (from original, PR#19)
    ├── ane_rmsnorm_bwd.h       # ANE RMSNorm backward (from original, PR#19)
    ├── dashboard.py            # TUI dashboard (from original)
    ├── stories_flex.h          # NEW — failed multi-IOSurface attempt (preserved)
    ├── lora.h                  # NEW — LoRA adapter: merge, gradients, checkpoint
    ├── train_lora.m            # NEW — LoRA on static pipeline
    ├── train_lora_flex.m       # NEW — LoRA with partial recompile
    ├── BENCHMARK_LORA.md       # NEW — LoRA benchmark results
    └── training_dynamic/       # NEW — dynamic weight pipeline
        ├── config.h            # Model config
        ├── io.h                # IOSurface I/O + dynamic weight packing
        ├── cpu_ops.h           # CPU ops with vocab compaction
        ├── mil_dynamic.h       # MIL generators using slice_by_size
        ├── lora.h              # LoRA adapter (dynamic pipeline)
        ├── train.m             # Dynamic full training (9 kernels, zero recompile)
        ├── train_lora.m        # Dynamic LoRA (the whole point of this fork)
        └── SESSION.md          # Development session notes
```

## Limitations

- **ANE utilization** — still below peak theoretical TFLOPS; element-wise ops and CPU round-trips between layers limit throughput
- **fp16 only** — ANE operates in fp16; accumulation to fp32 on CPU adds transfer overhead
- **Single model architecture** — hardcoded for Stories110M (Llama2); other architectures need MIL generator changes
- **macOS only** — private frameworks, Apple Silicon only
- **Private APIs** — may break on any macOS update; tested on 15.x (Sequoia)

## Disclaimer

This project uses Apple's private, undocumented APIs (`_ANEClient`, `_ANECompiler`, `_ANEInMemoryModelDescriptor`). These APIs are not covered by any public stability guarantee and may change or break with any macOS update. This is independent research into Apple Neural Engine architecture, using APIs discovered through runtime introspection for research and educational purposes under fair use and interoperability provisions (see *Sega v. Accolade*, 1992; DMCA sec. 1201(f)). No Apple proprietary code or binaries are included in this repository. This project is not affiliated with or endorsed by Apple Inc. Use at your own risk.

## License

MIT — see [LICENSE](LICENSE)

---

*Original research by [@maderix](https://github.com/maderix). Fork extensions by [@solatticus](https://github.com/solatticus), built with Claude as an AI coding assistant.*
