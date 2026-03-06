# ANE Training — Backpropagation on Apple Neural Engine

Training neural networks directly on Apple's Neural Engine (ANE) via reverse-engineered private APIs. No CoreML training APIs, no Metal, no GPU — pure ANE compute.

## Project Scope & Intent

I'm genuinely grateful for all the attention this project has received — I never expected a weekend research hack to blow up like this. Thank you to everyone who starred, forked, ran benchmarks on their own hardware, and shared the work. It means a lot.

That said, I want to set clear expectations about what this project is and isn't.

This is a **research project**, not a production framework.

The goal was to demonstrate that **training on the Apple Neural Engine — and potentially other NPUs — is possible**, and that the barrier has always been software support, not hardware capability. The ANE is a remarkably capable piece of silicon that Apple restricts to inference-only use through CoreML. This project bypasses that restriction using reverse-engineered private APIs to show what's possible when you give the hardware a chance.

### What This Project Is

- A proof of concept for ANE training via `_ANEClient` and `_ANECompiler` private APIs
- A set of benchmarks documenting real ANE performance characteristics (throughput, power, SRAM behavior)
- A reference for anyone exploring direct ANE access outside CoreML
- Research code that I update when I find something interesting

### What This Project Is Not

- A maintained framework or library
- A replacement for CoreML, MLX, llama.cpp, or any production inference stack
- A path to training large models on consumer hardware (yet)

### On The Hype

Some coverage of this project has overstated its implications. To be clear:

- Training works, but utilization is low (~5-9% of peak) with significant engineering challenges remaining
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

## Results

### Stories110M — Original Pipeline (12-layer, dim=768, 109M params)

- Static pipeline: **91 ms/step** (M3 Ultra), **106 ms/step** (M4)
- Dynamic pipeline: **110 ms/step**, no recompilation
- 72 ANE kernels per step (static), 9 shared kernels (dynamic)
- All forward and backward dx passes on ANE, dW gradients on CPU (Accelerate cblas)
- Adam optimizer, gradient accumulation, checkpoint/resume via exec() restart

### Qwen2.5-3B — Annie Pipeline (36-layer, dim=2048, 3B params)

First 3B parameter model trained on Apple Neural Engine:

- **Model**: Qwen2.5-3B (36 layers, 2048-dim, GQA 16/2 heads, 151K vocab)
- **Method**: LoRA rank 8 on Wq/Wv — 1.84M trainable params (0.06% of 3B)
- **Kernels**: 4 dynamic kernels shared across all 36 layers (vs. per-layer in static pipeline)
- **Speed**: ~135s/step, 324 ANE evals/step (144 forward + 180 backward)
- **Hardware**: M4 Pro, 24GB unified memory, 16-core ANE
- **Stability**: Zero crashes, zero ANE errors across extended training runs
- **Loss scaling**: 256x FP16 loss scaling required — ANE backward matmul products flush to zero without it

Key advances over the original pipeline:
- **Dynamic weight pipeline at scale** — 4 compiled kernels handle arbitrary layer count (original: per-layer kernels, hard-limited by ~239 compile ceiling)
- **GQA (Grouped Query Attention)** — asymmetric QKV projections. Original only supported MHA
- **151K vocabulary** — classifier and embedding at full Qwen2.5 vocab (original: 32K)
- **FP16 loss scaling** — global 256x scaling on dlogits, divided out before Adam update. Without this, gradients die in ANE's FP16 backward matmuls

## Architecture

### Static Pipeline (Stories110M)

6 ANE kernels per step:

| Kernel | Function | Weights |
|--------|----------|---------|
| `kFwdAttn` | RMSNorm + QKV projection + SDPA + output projection | Wq, Wk, Wv, Wo, rms1, mask |
| `kFwdFFN` | RMSNorm + SwiGLU FFN (W1, W3, SiLU, W2) | W1, W2, W3, rms2 |
| `kFFNBwd` | FFN backward (W2^T + SiLU_bwd + W1^T + W3^T) | W2^T, W1^T, W3^T |
| `kSdpaBwd1` | Wo^T + SDPA backward part 1 (dV, probs, dp) | Wo^T, mask |
| `kSdpaBwd2` | SDPA backward part 2 (softmax grad, dQ, dK) | — |
| `kQKVb` | QKV backward (Wq^T + Wk^T + Wv^T -> dx) | Wq^T, Wk^T, Wv^T |

### Dynamic Pipeline (Annie / Qwen2.5-3B)

4 shared kernels, weights passed via IOSurface at runtime:

| Kernel | Function | Dynamic Weights |
|--------|----------|-----------------|
| `qkvProj` | QKV projection (dim -> dim + 2*kv_dim) | Wq^T, Wk^T, Wv^T packed as spatial columns |
| `dimToHidden` | FFN up-projection (dim -> hidden_dim) | W1^T, W3^T packed |
| `hiddenToDim` | FFN down-projection (hidden_dim -> dim) | W2^T packed |
| `dimToDim` | Output projection (dim -> dim) | Wo^T packed |

CPU handles: RMSNorm (forward + backward), SDPA (forward + backward), residual connections, loss computation, dW gradient accumulation (cblas_sgemm), LoRA merge, Adam optimizer.

Key optimizations:
- **Channel-first CPU layout** — matches ANE IOSurface `[1,C,1,S]` format, eliminates all transpose overhead
- **vDSP vectorized RMSNorm** — 10x faster than naive (6.7ms -> 0.7ms)
- **GCD async cblas overlap** — dW gradient sgemms run in parallel with ANE evals on a serial dispatch queue
- **Deferred cblas wait** — wait pushed into next step's forward pass for maximum overlap
- **ANE RMSNorm fusion** — RMSNorm folded into forward kernels as MIL ops (reduce_sum + pow + mul)
- **Forward taps** — Q, K, V, attention scores, hidden states exposed via concat outputs, avoiding CPU recompute
- **FP16 loss scaling** — 256x scaling prevents gradient underflow in ANE backward matmuls
- **exec() restart** — bypasses ~239 ANE compile limit per process (static pipeline only)

## File Structure

```
├── api_exploration.m               # Initial ANE API discovery
├── inmem_basic.m                   # In-memory MIL compilation proof-of-concept
├── inmem_bench.m                   # ANE dispatch latency benchmarks
├── inmem_peak.m                    # Peak TFLOPS measurement (2048x2048 matmul)
├── sram_bench.m                    # ANE SRAM bandwidth probing
├── sram_probe.m                    # SRAM size/layout exploration
└── training/
    ├── ane_runtime.h               # ANE private API wrapper (compile, eval, IOSurface)
    ├── ane_mil_gen.h               # MIL program generation helpers
    ├── model.h                     # Model weight initialization and blob builders
    ├── forward.h                   # Forward pass MIL generators (static)
    ├── backward.h                  # Backward pass MIL generators (static)
    ├── lora.h                      # LoRA adapter (static pipeline)
    ├── stories_config.h            # Stories110M model config
    ├── train_large.m               # Main training loop (static pipeline)
    ├── test_*.m                    # Unit tests for individual kernels
    ├── Makefile
    ├── training_dynamic/           # Dynamic weight pipeline (stories110M)
    │   ├── config.h                # Model config + compile limits
    │   ├── mil_dynamic.h           # Dynamic MIL kernel generators
    │   ├── io.h                    # IOSurface weight staging helpers
    │   ├── cpu_ops.h               # CPU operations (RMSNorm, loss, Adam)
    │   ├── lora.h                  # LoRA adapter (dynamic pipeline)
    │   ├── train.m                 # Full training loop
    │   └── Makefile
    └── annie/                      # Qwen2.5-3B pipeline (LoRA fine-tuning)
        ├── config.h                # Qwen2.5-3B architecture config + GQA dims
        ├── mil_dynamic.h           # Dynamic MIL generators (4 shared kernels)
        ├── forward.h               # Forward pass (36 layers, GQA)
        ├── backward.h              # Backward pass (FP16 loss scaling)
        ├── cpu_ops.h               # CPU ops (RMSNorm, cross-entropy, Adam)
        ├── io.h                    # IOSurface I/O for large tensors
        ├── lora.h                  # LoRA rank-8 on Wq/Wv
        ├── train_lora.m            # Main training loop
        ├── convert_weights.py      # HuggingFace safetensors -> ANE binary
        ├── tokenize_data.py        # Conversation JSONL -> tokenized binary
        └── Makefile
```

## Building

Requires macOS 15+ on Apple Silicon.

```bash
# Stories110M (static pipeline)
cd training && make train_large
./train_large

# Stories110M (dynamic pipeline)
cd training/training_dynamic && make train
./train

# Qwen2.5-3B (Annie — requires converted weights)
cd training/annie && make train_lora
# Convert weights first:
python convert_weights.py --model Qwen/Qwen2.5-3B --output qwen3b_weights.bin
# Tokenize training data:
python tokenize_data.py --input conversations.jsonl --output annie_train_data.bin
# Train:
./train_lora --steps 1000 --lr 1e-4 --accum 10
```

No external dependencies for C code. Uses only system frameworks + private ANE APIs resolved at runtime via `objc_msgSend`. Python scripts require `transformers` and `safetensors`.

## Training Data

For Stories110M: pretokenized TinyStories data. Download with:
```bash
cd training && bash download_data.sh
```

For Annie/Qwen2.5-3B: conversation JSONL with query/response pairs. See `training/annie/tokenize_data.py`.

## How It Works

1. **MIL generation** — Objective-C code constructs MIL program text at runtime, specifying convolutions (for linear layers), matmul (for attention), softmax, element-wise ops
2. **In-memory compilation** — `_ANEInMemoryModelDescriptor` compiles MIL text + weight blobs directly to ANE programs, no disk mlmodelc needed
3. **IOSurface I/O** — Input/output tensors passed via IOSurface shared memory in `[1, channels, 1, spatial]` format (fp16)
4. **Weight delivery** — Static pipeline bakes weights as BLOBFILE constants (recompile per update); dynamic pipeline passes weights as extra spatial columns in IOSurface input (no recompile)
5. **Gradient flow** — Forward taps expose intermediates needed for backward; backward kernels compute dx (input gradients) on ANE; dW (weight gradients) computed on CPU via cblas
6. **FP16 loss scaling** — ANE operates in FP16 internally. Gradient products (~8e-5 x 0.036) underflow to zero. Loss scaling (256x on dlogits, divided out before optimizer) keeps gradients alive

## Limitations

- **SDPA causal masking** — ANE hardware ignores `attn_mask` in SDPA ops; causal attention is decomposed into separate Q@K^T -> mask+softmax -> scores@V
- **~239 compile limit** — ANE compiler leaks resources; worked around via `exec()` restart with checkpoint (static pipeline) or dynamic weight pipeline (no recompile)
- **Low utilization** — Training sustains ~1-2 TFLOPS out of 15.8+ peak due to CPU fallbacks and I/O overhead
- **Training speed at scale** — 3B model runs ~135s/step on M4 Pro. Practical for validation, not for large-scale training. GPU training recommended for production fine-tuning, with ANE for inference deployment

## Performance History

### Stories110M (Static Pipeline)

| Optimization | ms/step | ANE util |
|---|---|---|
| Baseline (vDSP transpose) | 33.5 | 3.1% |
| Channel-first layout | 20.3 | 5.2% |
| vDSP vectorized RMSNorm | 14.2 | 7.4% |
| GCD async cblas overlap | 11.4 | 9.2% |
| ANE RMSNorm fusion | 11.4 | 9.2% |
| Wo^T fusion (7->6 kernels) | 11.4 | 9.2% |
| Deferred cblas wait | **9.3** | **11.2%** |

### Qwen2.5-3B (Annie Pipeline)

| Metric | Value |
|---|---|
| Compile time (one-time) | 470ms (4 kernels) |
| Step time | ~135s |
| ANE evals/step | 324 (144 fwd + 180 bwd) |
| Weight load | 12.3GB (FP32) |
| Memory footprint | ~22GB peak |
| Compile count | 4 (constant, independent of layer count) |

## Disclaimer

This project uses Apple's private, undocumented APIs (`_ANEClient`, `_ANECompiler`, `_ANEInMemoryModelDescriptor`). These APIs are not covered by any public stability guarantee and may change or break with any macOS update. This is independent research into Apple Neural Engine architecture, using APIs discovered through runtime introspection for research and educational purposes under fair use and interoperability provisions (see *Sega v. Accolade*, 1992; DMCA 1201(f)). No Apple proprietary code or binaries are included in this repository. This project is not affiliated with or endorsed by Apple Inc. Use at your own risk.

## License

MIT — see [LICENSE](LICENSE)
