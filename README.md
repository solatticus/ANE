# ANE Training — Backpropagation on Apple Neural Engine

Training neural networks directly on Apple's Neural Engine (ANE) via reverse-engineered private APIs. No CoreML training APIs, no Metal, no GPU — pure ANE compute.

## TL;DR

Every Mac with Apple Silicon has a chip called the **Neural Engine** (ANE). It's a 15.8 TFLOPS accelerator — faster than most laptops' GPUs — but Apple only lets you use it for *running* AI models, not *training* them. That's a software restriction, not a hardware one.

**This project removes that restriction.**

We reverse-engineered Apple's private APIs and figured out how to run the full training loop — forward pass, backward pass, weight updates — directly on the Neural Engine. No GPU, no Metal, no CoreML training APIs. Just the ANE doing what Apple said it couldn't.

**What works today:**
- Train a 109M-parameter transformer (12 layers, same architecture as Llama 2) on the ANE
- LoRA fine-tuning with 2.9 MB checkpoints — adapts a pretrained model with 744x fewer parameters
- Zero-recompile pipeline — compile once, train forever (we tried 5 different approaches before finding one that worked)
- Python bridge — use the ANE from Python via a shared library, no Objective-C required
- Real training data (TinyStories, 20M tokens), real loss curves, real checkpoints

**What it means:**
- Every Mac is a training machine, not just an inference box
- NPUs in general (Qualcomm, Intel, etc.) are likely training-capable too — the barrier is software everywhere
- LoRA fine-tuning on-device could enable personal model adaptation without cloud GPUs

**Try it yourself** (macOS 15+, Apple Silicon):
```bash
git clone https://github.com/solatticus/ANE.git && cd ANE/training
bash download_data.sh                     # grab training data (~41 MB)
cd training_dynamic && make train_lora    # build the trainer
./train_lora --steps 100 --lr 1e-4       # fine-tune on your Mac's Neural Engine
```

---

## Project Scope & Intent

I'm genuinely grateful for all the attention this project has received — I never expected a weekend research hack to blow up like this. Thank you to everyone who starred, forked, ran benchmarks on their own hardware, and shared the work. It means a lot.

That said, I want to set clear expectations about what this project is and isn't.

This is a **research project**, not a production framework.

The goal was to demonstrate that **training on the Apple Neural Engine — and potentially other NPUs — is possible**, and that the barrier has always been software support, not hardware capability. The ANE is a remarkably capable piece of silicon that Apple restricts to inference-only use through CoreML. This project bypasses that restriction using reverse-engineered private APIs to show what's possible when you give the hardware a chance.

### What this project is

- A proof of concept for ANE training via `_ANEClient` and `_ANECompiler` private APIs
- A set of benchmarks documenting real ANE performance characteristics (throughput, power, SRAM behavior)
- A reference for anyone exploring direct ANE access outside CoreML
- Research code that I update when I find something interesting

### What this project is not

- A maintained framework or library
- A replacement for CoreML, MLX, llama.cpp, or any production inference stack
- A path to training large models on consumer hardware (yet)

### On the hype

Some coverage of this project has overstated its implications. To be clear:

- Training works, but utilization is low with significant engineering challenges remaining
- Many element-wise operations still fall back to CPU
- This does **not** replace GPU training for anything beyond small research models today

The honest results — including all limitations — are documented in the accompanying articles:
- [Part 1: Reverse Engineering](https://maderix.substack.com/p/inside-the-m4-apple-neural-engine)
- [Part 2: Benchmarks](https://maderix.substack.com/p/inside-the-m4-apple-neural-engine-615)

### Fork it, build on it

This is MIT licensed for a reason. Everyone now has access to AI-assisted development tools that can adapt and extend code in hours. If this project is useful to you — take it, modify it, build something better. If you do something cool with it, I'd love to hear about it.

---

## What This Is

A from-scratch implementation of transformer training (forward + backward pass) running on the ANE in Apple Silicon. The ANE is a 15.8 TFLOPS (M4) inference accelerator that Apple does not expose for training. This project reverse-engineers the `_ANEClient` / `_ANECompiler` private APIs and the MIL (Model Intermediate Language) format to run custom compute graphs — including backpropagation — directly on ANE hardware.

**Current results (M4 Pro, Stories110M — 12 layers, 109M params):**
- 92-125 ms/step depending on pipeline and training mode
- 12-layer Llama2-architecture transformer (dim=768, hidden=2048, 12 heads, 32K vocab)
- Real tokenized training data (TinyStories, 20M tokens)
- LoRA fine-tuning: 147K trainable params, 2.9 MB checkpoints, 744x parameter reduction
- Zero-recompile dynamic pipeline: compile 9 kernels once, train forever
- Adam optimizer, gradient accumulation, checkpoint/resume across `exec()` restarts
- Python bridge API (`libane_bridge.dylib`) for using ANE from Python via ctypes

## The Road to Zero Recompile

The biggest engineering challenge in ANE training isn't the math — it's the compilation bottleneck. Weights are baked into ANE programs as constants at compile time. Every time weights change, you recompile. The ANE compiler leaks resources and dies after ~119 compilations per process. We tried everything to eliminate this.

### What we tried and what broke

**Attempt 1: Weight file swap.** Overwrite the weight blob on disk, unload and reload the model. Result: ANE ignores the file change. Weights are baked at compile time and the reload serves the cached program. **Dead end.**

**Attempt 2: `weightsBuffer` IOSurface.** The `_ANERequest` API accepts a `weightsBuffer` parameter. We filled it with new weights and passed it at eval time. Result: output unchanged. The parameter likely serves a different internal purpose. **Dead end.**

**Attempt 3: Multiple dynamic IOSurfaces.** Pass weights as separate IOSurface inputs alongside activations (`stories_flex.h`). Result: ANE rejects it with `status=0x1d`. The hardware only supports a single dynamic input tensor. **Dead end.**

**Attempt 4: Partial recompile.** Only recompile the kernels whose weights changed (LoRA targets: Wq/Wv). Result: works — 2x speedup over full recompile. But still hits the compile limit, still needs `exec()` restart every 70 steps. **Partial win, not the answer.**

### The breakthrough: weights as spatial columns

**Attempt 5: Pack weights into the input tensor's spatial dimension.**

The ANE takes a single input as `[1, channels, 1, spatial]`. What if we made the spatial dimension wider than the sequence length, and stuffed weight matrices into the extra columns?

Layout: `[1, DIM, 1, SEQ + weight_cols]` — activations in `sp[0:SEQ]`, weights packed sequentially in `sp[SEQ:]`.

The MIL graph uses `slice_by_size` to carve out the weight region and `matmul` to multiply. The ANE doesn't know the extra columns are weights — it just sees a wider tensor.

**Result: 9 kernels compiled once at startup. Zero recompilation. No compile limit. No `exec()` restart. The dynamic weight pipeline.**

This is the key insight that makes practical ANE training possible.

## Architecture

### Single-layer prototype (6 kernels per step)

| Kernel | Function | Weights |
|--------|----------|---------|
| `kFwdAttn` | RMSNorm + QKV projection + SDPA + output projection | Wq, Wk, Wv, Wo, rms1, mask |
| `kFwdFFN` | RMSNorm + SwiGLU FFN (W1, W3, SiLU, W2) | W1, W2, W3, rms2 |
| `kFFNBwd` | FFN backward (W2^T + SiLU_bwd + W1^T + W3^T) | W2^T, W1^T, W3^T |
| `kSdpaBwd1` | Wo^T + SDPA backward part 1 (dV, probs, dp) | Wo^T, mask |
| `kSdpaBwd2` | SDPA backward part 2 (softmax grad, dQ, dK) | — |
| `kQKVb` | QKV backward (Wq^T + Wk^T + Wv^T -> dx) | Wq^T, Wk^T, Wv^T |

### Stories110M (12 layers, 109M params)

Four training pipelines, each with different tradeoffs:

| Pipeline | ms/step | Compile | Recompile? | Trainable params |
|----------|---------|---------|------------|------------------|
| Static baseline (`train_large`) | 106.7 | 7.6s/restart | Every 10 steps | 109M (full) |
| Static + ANE extras (`train_large_ane`) | 91.8 | 9.6s/restart | Every 10 steps | 109M (full) |
| Dynamic (`training_dynamic/train`) | 111 | 0.4s once | **Never** | 109M (full) |
| Dynamic LoRA (`training_dynamic/train_lora`) | 125.6 | 0.3s once | **Never** | **147K (LoRA)** |

CPU handles: RMSNorm backward, residual connections, loss computation, dW gradient accumulation (cblas_sgemm), Adam optimizer.

### Key optimizations
- **Channel-first layout** — `[1,C,1,S]` everywhere, matches ANE IOSurface format, zero transpose overhead
- **Dynamic weight packing** — weights as extra spatial columns in single IOSurface
- **vDSP vectorized math** — RMSNorm (10x faster), cross-entropy (8x faster)
- **NEON fp16<->fp32** — ARM intrinsics for fast IOSurface data transfer
- **GCD async cblas** — dW gradient sgemms overlap with ANE eval on background queue
- **Deferred cblas wait** — pushed into next step's forward pass for maximum overlap
- **ANE RMSNorm fusion** — folded into forward kernels as MIL ops
- **Wo^T fusion** — output projection backward merged into SDPA backward kernel
- **Forward taps** — intermediates (Q, K, V, scores) exposed via concat, avoiding CPU recompute
- **Vocab compaction** — 32K -> 9.2K active tokens, 3.5x reduction in classifier compute

## LoRA Fine-Tuning

Low-Rank Adaptation with rank-4 matrices on Wq and Wv projections. Works on both static and dynamic pipelines.

| | Full Training | LoRA |
|---|---|---|
| Trainable params | 109,529,856 (438 MB) | 147,456 (576 KB) |
| Reduction | — | **744x fewer params** |
| Optimizer memory | ~1.7 GB (Adam m+v) | ~2.3 MB |
| Checkpoint size | 438 MB | 2.9 MB |
| dW GEMMs per layer | 7 | 2 (Wq, Wv only) |

CPU-side merge: `W_eff = W_frozen + (alpha/rank) * B @ A` — adds <1ms per step. B initialized to zero ensures the model starts at the pretrained baseline (identity property verified: loss matches pretrained exactly at step 0). Gradient flow verified over 1000 steps: |B| grows 49x from near-zero init, consistent linear growth.

On the dynamic pipeline, LoRA eliminates 5 of 7 weight gradient GEMMs per layer *and* never recompiles — best of both worlds.

## Python Bridge

`bridge/libane_bridge.dylib` — C-callable shared library wrapping all ANE private APIs, designed for Python ctypes:

```c
int ane_bridge_init(void);
ANEKernelHandle *ane_bridge_compile(mil_text, mil_len, weight_data, weight_len, ...);
bool ane_bridge_eval(ANEKernelHandle *kernel);
void ane_bridge_write_input(kernel, idx, data, bytes);
void ane_bridge_read_output(kernel, idx, data, bytes);
void ane_bridge_free(ANEKernelHandle *kernel);
```

Handles IOSurface lifecycle, ARC memory management, compile retry logic, and the full compile->load->eval->unload flow.

## File Structure

```
├── api_exploration.m           # Initial ANE API discovery and reverse engineering
├── inmem_basic.m               # In-memory MIL compilation proof-of-concept
├── inmem_bench.m               # ANE dispatch latency benchmarks
├── inmem_peak.m                # Peak TFLOPS measurement (stacked 2048x2048 matmul)
├── sram_bench.m                # ANE SRAM bandwidth probing
├── sram_probe.m                # SRAM size/layout exploration
├── bridge/
│   ├── ane_bridge.h            # C-callable ANE bridge API
│   ├── ane_bridge.m            # Bridge implementation (ObjC -> C)
│   ├── libane_bridge.dylib     # Compiled shared library
│   └── Makefile
└── training/
    ├── ane_runtime.h           # ANE private API wrapper
    ├── ane_mil_gen.h           # MIL program generation helpers
    ├── model.h                 # Weight init and blob builders
    ├── forward.h               # Forward pass MIL generators
    ├── backward.h              # Backward pass MIL generators
    ├── stories_config.h        # Static pipeline: config, structs, alloc
    ├── stories_io.h            # Static pipeline: IOSurface I/O, NEON, compile
    ├── stories_mil.h           # Static pipeline: MIL generators (6 kernel types)
    ├── stories_cpu_ops.h       # Static pipeline: vDSP RMSNorm, cross-entropy, Adam
    ├── stories_flex.h          # Failed attempt: weights as separate IOSurfaces
    ├── ane_classifier.h        # ANE classifier fwd (32K conv), softmax
    ├── ane_rmsnorm_bwd.h       # ANE RMSNorm backward kernel
    ├── lora.h                  # LoRA adapter: merge, gradient extract, checkpoint
    ├── train.m                 # Minimal training loop (early prototype)
    ├── tiny_train.m            # 2-layer tiny model training
    ├── train_large.m           # Static baseline: 72 kernels, recompile every 10 steps
    ├── train_large_ane.m       # Static + ANE extras: classifier/softmax/rmsnorm on ANE
    ├── train_lora.m            # LoRA on static pipeline
    ├── train_lora_flex.m       # LoRA with partial recompile (2x faster, still limited)
    ├── dashboard.py            # TUI dashboard: loss curve, power, CPU, memory graphs
    ├── download_data.sh        # Download pretokenized TinyStories from HuggingFace
    ├── test_*.m                # Unit tests for individual kernels
    ├── Makefile
    └── training_dynamic/
        ├── config.h            # Model config (DIM=768, HIDDEN=2048, etc.)
        ├── io.h                # IOSurface I/O + dynamic weight packing
        ├── cpu_ops.h           # CPU ops: RMSNorm, cross-entropy, Adam, vocab compaction
        ├── mil_dynamic.h       # MIL generators using slice_by_size for dynamic weights
        ├── lora.h              # LoRA adapter (dynamic pipeline version)
        ├── train.m             # Dynamic full training: 9 kernels, zero recompile
        ├── train_lora.m        # Dynamic LoRA: zero recompile + parameter efficient
        └── Makefile
```

## Building

Requires macOS 15+ on Apple Silicon (tested on M4, M4 Pro, M5).

```bash
# Download training data (pretokenized TinyStories, ~41 MB)
cd training && bash download_data.sh

# Static baseline
make train_large && ./train_large --steps 100

# Static + ANE extras (14% faster per step)
make train_large_ane && ./train_large_ane --steps 100

# Dynamic pipeline — zero recompile (recommended)
cd training_dynamic && make train
./train --scratch --steps 1000        # full training from random init
./train --steps 1000                  # resume from checkpoint

# Dynamic LoRA — zero recompile + parameter efficient (recommended for fine-tuning)
cd training_dynamic && make train_lora
./train_lora --steps 1000 --lr 1e-4   # fine-tune pretrained weights

# LoRA on static pipeline (for comparison)
make train_lora && ./train_lora --steps 100 --lr 1e-4
```

**CLI flags (all pipelines):**
- `--steps N` — training steps (default 10000)
- `--lr F` — learning rate (default 3e-4)
- `--model PATH` — pretrained weights file
- `--ckpt PATH` — checkpoint file
- `--resume` — resume from checkpoint
- `--scratch` — train from random initialization
- `--no-ane-extras` — (train_large_ane only) fall back to CPU classifier/softmax

**Dashboard:**
```bash
pip install blessed psutil numpy
sudo python3 dashboard.py              # static pipeline
sudo python3 dashboard.py --dynamic    # dynamic pipeline
```

## How It Works

1. **MIL generation** — Objective-C constructs MIL program text at runtime: convolutions for linear layers, matmul for attention, softmax, element-wise ops. Dynamic pipeline adds `slice_by_size` to extract weight regions from the spatial dimension.
2. **In-memory compilation** — `_ANEInMemoryModelDescriptor` compiles MIL text + weight blobs directly to ANE programs. No disk `.mlmodelc` needed.
3. **IOSurface I/O** — Tensors passed via IOSurface shared memory in `[1, C, 1, S]` format (fp16). Dynamic pipeline packs weights into the spatial dimension: `[1, C, 1, SEQ + weight_cols]`.
4. **Weight handling** — Static pipeline: weights baked as BLOBFILE constants, recompiled when changed. Dynamic pipeline: weights written to IOSurface at runtime, zero recompilation.
5. **Gradient flow** — Forward taps expose intermediates; backward kernels compute dx (activation gradients) on ANE; dW (weight gradients) on CPU via cblas_sgemm. LoRA mode skips 5 of 7 dW GEMMs per layer.

## ANE Hardware Findings

Through systematic probing (M4, M4 Pro, M5):

- **Compile limit:** ~119 compilations per process before the compiler leaks out. Workaround: `exec()` restart with checkpoint, or use the dynamic pipeline.
- **SDPA causal masking:** ANE hardware ignores `attn_mask` in SDPA ops. Must decompose into Q@K^T, mask+softmax, scores@V as separate operations.
- **Weight immutability:** Once compiled, weights cannot be changed via file swap, unload/reload, or `weightsBuffer`. They are baked at compile time.
- **Single dynamic input:** ANE rejects multiple dynamic IOSurface inputs (`status=0x1d`). One input tensor only.
- **QoS has no effect:** All QoS values 0-63 produce identical latency. ANE runs at fixed frequency.
- **Chaining API exists:** `_ANEChainingRequest` supports loopback execution (output->input) — unexplored but promising for multi-layer pipelining.
- **67 private classes** discovered via runtime introspection, many unexplored.

## Performance

### Single-layer prototype optimization history

| Optimization | ms/step | ANE util |
|---|---|---|
| Baseline (vDSP transpose) | 33.5 | 3.1% |
| Channel-first layout | 20.3 | 5.2% |
| vDSP vectorized RMSNorm | 14.2 | 7.4% |
| GCD async cblas overlap | 11.4 | 9.2% |
| ANE RMSNorm fusion | 11.4 | 9.2% |
| Wo^T fusion (7->6 kernels) | 11.4 | 9.2% |
| Deferred cblas wait | **9.3** | **11.2%** |

### Stories110M (12 layers) — 20-step comparison

| | Static | Static + ANE | Dynamic | Dynamic LoRA |
|---|---|---|---|---|
| Wall time | 10.1s | 11.7s | **~2.6s** | **~2.8s** |
| Compile | 7.6s (75%) | 9.6s (82%) | 0.4s (15%) | 0.3s (11%) |
| ms/step | 106.7 | 91.8 | 111 | 125.6 |
| Recompile? | Every 10 steps | Every 10 steps | **Never** | **Never** |

### LoRA training efficiency

| Metric | Static LoRA | Dynamic LoRA |
|---|---|---|
| Compile | 3400ms/batch | **333ms once** |
| ms/step | ~400 | **125.6** |
| exec() restart | Every 10 steps | **Never** |
| Speedup vs static | 1x | **3.2x** |

## Limitations

- **ANE utilization** — Still well below peak theoretical TFLOPS. Element-wise ops, CPU round-trips between layers, and data transfer overhead limit throughput.
- **fp16 only** — ANE operates in fp16. Accumulation to fp32 happens on CPU, adding transfer overhead.
- **Single model architecture** — Currently hardcoded for Stories110M (Llama2). Adapting to other architectures requires modifying MIL generators.
- **macOS only** — Uses private frameworks that only exist on macOS/Apple Silicon.
- **Private APIs** — May break with any macOS update. Tested on macOS 15.x (Sequoia).

## Disclaimer

This project uses Apple's private, undocumented APIs (`_ANEClient`, `_ANECompiler`, `_ANEInMemoryModelDescriptor`). These APIs are not covered by any public stability guarantee and may change or break with any macOS update. This is independent research into Apple Neural Engine architecture, using APIs discovered through runtime introspection for research and educational purposes under fair use and interoperability provisions (see *Sega v. Accolade*, 1992; DMCA sec. 1201(f)). No Apple proprietary code or binaries are included in this repository. This project is not affiliated with or endorsed by Apple Inc. Use at your own risk.

## License

MIT — see [LICENSE](LICENSE)

---

*Built by a human + Claude, one weekend at a time.*
